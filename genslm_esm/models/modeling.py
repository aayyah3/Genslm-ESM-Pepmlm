"""Adapted PyTorch ESM model."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import cast

import einops
import torch
import torch.nn.functional as F
from einops import rearrange
from einops import repeat
from torch import nn
from transformers import logging
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func  # type: ignore
    from flash_attn.bert_padding import pad_input  # type:ignore
    from flash_attn.bert_padding import unpad_input  # type:ignore
    from flash_attn.ops.triton.rotary import (  # type:ignore
        apply_rotary as apply_triton_rotary,
    )

    is_flash_attn_available = True
except (ImportError, RuntimeError):
    pad_input = None
    unpad_input = None
    apply_triton_rotary = None
    flash_attn_varlen_qkvpacked_func = None
    is_flash_attn_available = False

try:
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
except (ImportError, RuntimeError):
    comm, rank, size = None, None, None


# from esm.utils.constants.models import ESMC_600M
from genslm_esm.models.configuration import ContrastiveEsmCConfig

logger = logging.get_logger(__name__)

# ESMC pad token id
ESMC_PAD_TOKEN_ID = 1


def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1),
            '... d two -> ... (d two)',
            two=2,
        )


def apply_rotary_emb_torch(x, cos, sin, interleaved=False, _inplace=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    seqlen = x.size(1)
    cos = cos[:seqlen]
    sin = sin[:seqlen]
    cos = repeat(cos, 's d -> s 1 (2 d)')
    sin = repeat(sin, 's d -> s 1 (2 d)')
    return torch.cat(
        [
            x[..., :ro_dim] * cos
            + rotate_half(x[..., :ro_dim], interleaved) * sin,
            x[..., ro_dim:],
        ],
        dim=-1,
    )


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        self.device = device

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None
        self.reset_parameters()

    def reset_parameters(self):
        inv_freq = self._compute_inv_freq(self.device)
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        arange = torch.arange(
            0,
            self.dim,
            2,
            device=self.device,
            dtype=torch.float32,
        )
        scale = (
            (arange + 0.4 * self.dim) / (1.4 * self.dim)
            if self.scale_base is not None
            else None
        )
        self.register_buffer('scale', scale)

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(
                    0,
                    self.dim,
                    2,
                    device=device,
                    dtype=torch.float32,
                )
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(
                    seqlen,
                    device=device,
                    dtype=self.inv_freq.dtype,
                )  # pyright: ignore[reportArgumentType, reportCallIssue]
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)  # pyright: ignore[reportArgumentType]

            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(  # pyright: ignore[reportCallIssue]
                        seqlen,
                        dtype=self.scale.dtype,  # pyright: ignore[reportArgumentType]
                        device=self.scale.device,  # pyright: ignore[reportArgumentType]
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(
                    -1,
                )
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        self._update_cos_sin_cache(
            q.shape[1] + seqlen_offset,
            device=q.device,
            dtype=q.dtype,
        )
        assert self._cos_cached is not None
        assert self._sin_cached is not None
        if self.scale is None:
            return (
                apply_rotary_emb_torch(
                    q,
                    self._cos_cached[seqlen_offset:],
                    self._sin_cached[seqlen_offset:],
                    self.interleaved,
                    True,  # inplace=True
                ),
                apply_rotary_emb_torch(
                    k,
                    self._cos_cached[seqlen_offset:],
                    self._sin_cached[seqlen_offset:],
                    self.interleaved,
                    True,  # inplace=True
                ),
            )  # type: ignore
        else:
            assert False


class TritonRotaryEmbedding(RotaryEmbedding):
    def forward(
        self,
        qkv: torch.Tensor,
        cu_seqlens,
        max_seqlen,
    ) -> torch.Tensor:
        """
        qkv: (n, 3, nheads, headdim)
        cu_seqlens: cumulative sequence lengths
        max_seqlen: max sequence length
        """
        self._update_cos_sin_cache(
            max_seqlen,
            device=qkv.device,
            dtype=qkv.dtype,
        )
        assert self._cos_cached is not None
        assert self._sin_cached is not None

        assert apply_triton_rotary is not None
        # In-place modification
        apply_triton_rotary(
            qkv[:, 0],
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )
        apply_triton_rotary(
            qkv[:, 1],
            self._cos_cached,
            self._sin_cached,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            inplace=True,
        )

        return qkv


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_head = self.d_model // self.n_heads
        self.layernorm_qkv = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 3, bias=bias),
        )
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if qk_layernorm:
            self.q_ln = nn.LayerNorm(d_model, bias=bias)
            self.k_ln = nn.LayerNorm(d_model, bias=bias)
        else:
            self.q_ln = nn.Identity()
            self.k_ln = nn.Identity()

        self.rotary = RotaryEmbedding(d_model // n_heads)

    def _apply_rotary(self, q: torch.Tensor, k: torch.Tensor):
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x, seq_id):
        qkv_BLD3 = self.layernorm_qkv(x)
        query_BLD, key_BLD, value_BLD = torch.chunk(qkv_BLD3, 3, dim=-1)
        query_BLD, key_BLD = (
            self.q_ln(query_BLD).to(query_BLD.dtype),
            self.k_ln(key_BLD).to(query_BLD.dtype),
        )
        query_BLD, key_BLD = self._apply_rotary(query_BLD, key_BLD)

        reshaper = functools.partial(
            einops.rearrange,
            pattern='b s (h d) -> b h s d',
            h=self.n_heads,
        )

        query_BHLD, key_BHLD, value_BHLD = map(
            reshaper,
            (query_BLD, key_BLD, value_BLD),
        )

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_BLL = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_BHLL = mask_BLL.unsqueeze(1)

            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD,
                key_BHLD,
                value_BHLD,
                mask_BHLL,
            )
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_BHLD = F.scaled_dot_product_attention(
                query_BHLD,
                key_BHLD,
                value_BHLD,
            )

        context_BLD = einops.rearrange(context_BHLD, 'b h s d -> b s (h d)')

        return self.out_proj(context_BLD)


class FlashMultiHeadAttention(MultiHeadAttention):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            qk_layernorm=qk_layernorm,
        )

        # Flash attention rotary.
        self.rotary = TritonRotaryEmbedding(d_model // n_heads)

    def forward(self, x, seq_id):
        assert seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(
            torch.cumsum(seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        max_seqlen = seqlens.max().item()

        qkv_ND3 = self.layernorm_qkv(x)

        query_ND, key_ND, value_ND = torch.chunk(qkv_ND3, 3, dim=-1)
        query_ND, key_ND = (
            self.q_ln(query_ND).to(query_ND.dtype),
            self.k_ln(key_ND).to(query_ND.dtype),
        )

        qkv_N3D = torch.stack([query_ND, key_ND, value_ND], dim=1)
        qkv_N3HD = einops.rearrange(
            qkv_N3D,
            pattern='n a (h d) -> n a h d',
            h=self.n_heads,
        )
        qkv_N3HD = self.rotary(qkv_N3HD, cu_seqlens, max_seqlen)

        context_NHD = flash_attn_varlen_qkvpacked_func(  # type: ignore
            qkv_N3HD,
            cu_seqlens,
            max_seqlen,
            softmax_scale=self.d_head**-0.5,
        )
        context_ND = einops.rearrange(context_NHD, 'n h d -> n (h d)')

        return self.out_proj(context_ND)


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    # set hidden dimesion to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """
    SwiGLU activation function as an nn.Module, allowing it to be used within nn.Sequential.
    This module splits the input tensor along the last dimension and applies the SiLU (Swish)
    activation function to the first half, then multiplies it by the second half.
    """

    def __init__(self):
        super(SwiGLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(d_model: int, expansion_ratio: float, bias: bool):
    return nn.Sequential(
        nn.LayerNorm(d_model),
        nn.Linear(
            d_model,
            swiglu_correction_fn(expansion_ratio, d_model) * 2,
            bias=bias,
        ),
        SwiGLU(),
        nn.Linear(
            swiglu_correction_fn(expansion_ratio, d_model),
            d_model,
            bias=bias,
        ),
    )


class UnifiedTransformerBlock(nn.Module):
    """
    A unified transformer block that can optionally incorporate geometric attention.

    This class defines a transformer block that can be configured to use geometric attention
    alongside the standard multi-head attention mechanism. It is designed to be a flexible
    component of transformer-based models, allowing for the integration of geometric reasoning.

    Parameters
    ----------
    d_model : int
        The dimensionality of the input and output features of the transformer block.
    n_heads : int
        The number of attention heads in the multi-head attention mechanism.
    n_layers : int
        The number of layers in the transformer block.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        use_plain_attn: bool = True,
        use_flash_attn: bool = False,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        qk_layernorm: bool = True,
    ):
        super().__init__()
        self.use_plain_attn = use_plain_attn
        if self.use_plain_attn:
            if use_flash_attn:
                self.attn = FlashMultiHeadAttention(
                    d_model,
                    n_heads,
                    bias,
                    qk_layernorm=qk_layernorm,
                )
            else:
                self.attn = MultiHeadAttention(
                    d_model,
                    n_heads,
                    bias,
                    qk_layernorm=qk_layernorm,
                )
        self.ffn = swiglu_ln_ffn(d_model, expansion_ratio, bias)
        self.scaling_factor = residue_scaling_factor

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the UnifiedTransformerBlock.

        Parameters
        ----------
        x : torch.Tensor[float]
            Input tensor to the transformer block, typically the output from the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch, used for attention masking.

        Returns
        -------
        torch.Tensor[float]
            The output tensor after applying the transformer block operations.
        """
        if self.use_plain_attn:
            r1 = self.attn(x, sequence_id)
            x = x + r1 / self.scaling_factor

        r3 = self.ffn(x) / self.scaling_factor
        x = x + r3

        return x


class TransformerStack(nn.Module):
    """
    A stack of transformer blocks used in the ESM-3 model. Each block is a UnifiedTransformerBlock,
    which can either be geometric attention or standard multi-head attention.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads.
        v_heads (int): The number of voting heads.
        n_layers (int): The number of transformer blocks in the stack.
        n_layers_geom (int, optional): The number of transformer blocks that use geometric attention.
        scale_residue (bool, optional): Whether to scale the residue connections in each transformer block.
        mask_and_zero_frameless (bool, optional): Whether to mask and zero frameless positions in the input.
            Only applies in the geometric attention blocks, which is conditioned on the structure
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        scale_residue: bool = True,
        bias: bool = False,
        qk_layernorm: bool = True,
        expansion_ratio: float = 8 / 3,
        use_flash_attn: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                UnifiedTransformerBlock(
                    d_model,
                    n_heads,
                    use_flash_attn=use_flash_attn,
                    residue_scaling_factor=(
                        math.sqrt(n_layers / 36) if scale_residue else 1.0
                    ),
                    expansion_ratio=expansion_ratio,
                    bias=bias,
                    qk_layernorm=qk_layernorm,
                )
                for i in range(n_layers)
            ],
        )
        self.norm = nn.LayerNorm(d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        sequence_id: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """
        Forward pass of the TransformerStack.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).
            sequence_id (torch.Tensor): The sequence ID tensor of shape (batch_size, sequence_length).

        Returns
        -------
            post_norm: The output tensor of shape (batch_size, sequence_length, d_model).
            pre_norm: The embedding of shape (batch_size, sequence_length, d_model).
            hiddens: The list of hidden states of shape (n_layers, batch_size, sequence_length, d_model).
        """
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id)
            hiddens.append(x)
        return self.norm(x), x, hiddens


def RegressionHead(
    d_model: int,
    output_dim: int,
    hidden_dim: int | None = None,
) -> nn.Module:
    """Single-hidden layer MLP for supervised output.

    Args:
        d_model: input dimension
        output_dim: dimensionality of the output.
        hidden_dim: optional dimension of hidden layer, defaults to d_model.

    Returns
    -------
        output MLP module.
    """
    hidden_dim = hidden_dim if hidden_dim is not None else d_model
    return nn.Sequential(
        nn.Linear(d_model, hidden_dim),
        nn.GELU(),
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, output_dim),
    )


@dataclass
class ESMCOutput:
    """Output of the ESMC model."""

    sequence_logits: torch.Tensor
    embeddings: torch.Tensor | None
    hidden_states: torch.Tensor | None


class ESMC(nn.Module):
    """
    ESMC model implementation.

    Args:
        d_model (int): The dimensionality of the input and output feature vectors.
        n_heads (int): The number of attention heads in the transformer layers.
        n_layers (int): The number of transformer layers.
        contrastive_temperature (float): The temperature for the contrastive loss.
        use_flash_attn (bool): Whether to use flash attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        contrastive_temperature: float,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        # 97 tokens: 64 codons + 20 amino acids + 11 special tokens / extras
        self.embed = nn.Embedding(97, d_model)

        self._use_flash_attn = is_flash_attn_available and use_flash_attn
        self.transformer = TransformerStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            use_flash_attn=self._use_flash_attn,
        )

        # NOTE: This is an artifact of the ESMC model, the data is still passed
        # through this head in the forward function, but the results are not
        # used and loss is not computed. The actual amino acid lm_head is moved
        # to the EsmForContrastiveMaskedLM model. The initial pLM ESMC weights
        # stored in sequence_head are used to initialize the weights of the
        # amino acid lm_head in the EsmForContrastiveMaskedLM model.
        self.sequence_head = RegressionHead(d_model, 64)

        # Contrastive head for the alignment of codon and amino acid embeddings
        self.contrastive_head = EsmContrastiveProjectionHead(
            d_model=d_model,
            contrastive_temperature=contrastive_temperature,
        )

        # Regression head for codon predictions
        self.codon_lm_head = RegressionHead(d_model, 69)

        # NOTE: This is unused and is an artifact of our original
        # model construction. It does not get trained and not forward passes
        # are performed through this head. We need it to load the weights
        # properly.
        self.lm_head = RegressionHead(d_model, 33)

    def forward(
        self,
        sequence_tokens: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
    ) -> ESMCOutput:
        """Perform a forward pass through the ESMC model.

        Check utils to see how to tokenize inputs from raw data.

        Args:
            sequence_tokens (torch.Tensor, optional): The amino acid tokens.
            sequence_id (torch.Tensor, optional): The sequence ID.

        Returns
        -------
            ESMCOutput: The output of the ESMC model.
        """
        if sequence_id is None:
            # For EMSC, a boolean mask is created in place of sequence_id
            # if not specified.
            sequence_id = sequence_tokens != ESMC_PAD_TOKEN_ID

        x = self.embed(sequence_tokens)

        B, L = x.shape[:2]

        # If sequence_id looks like a mask.
        if self._use_flash_attn:
            assert sequence_id.dtype == torch.bool, (
                'sequence_id must be a boolean mask if Flash Attention is used'
            )
            assert sequence_id.shape == (B, L)
            assert unpad_input is not None
            x, indices, *_ = unpad_input(  # type: ignore
                x,
                sequence_id,
            )
        else:
            indices = None

        x, _, hiddens = self.transformer(x, sequence_id=sequence_id)

        if self._use_flash_attn:
            assert indices is not None
            assert pad_input is not None
            x = pad_input(x, indices, B, L)  # Back to [B, L, D]
            hiddens = [
                # Back to [[B, L, D], ...]
                pad_input(h, indices, B, L)
                for h in hiddens
            ]

        # Stack hidden states into a [n_layers, B, L, D] matrix.
        hiddens = torch.stack(hiddens, dim=0)  # type: ignore

        sequence_logits = self.sequence_head(x)
        output = ESMCOutput(
            sequence_logits=sequence_logits,
            embeddings=x,
            hidden_states=hiddens,
        )
        return output


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float) -> None:
        """Contrastive loss for SimCLR.

        Reference: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html#SimCLR

        Parameters
        ----------
        temperature: float
            Determines how peaked the distribution. Since many similarity
            metrics are bounded, the temperature parameter allows us to
            balance the influence of many dissimilar image patches versus
            one similar patch.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # NOTE: z.shape == (batch_size, hidden_size)
        # TODO: Can we cache the pos_mask calculation with lru_cache?
        batch_size = z.shape[0]

        # If we are using distributed training, gather all the tensors
        # across all processes and concatenate them on the batch dimension.
        if size is not None and rank is not None:
            z_all = [torch.empty_like(z) for _ in range(size)]
            # gather all the tensors
            torch.distributed.all_gather(z_all, z)
            # replace local rank information so gradients propagate
            z_all[rank] = z
            # concatenate all the tensors on batch dimension
            z = torch.cat(z_all, dim=0)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        cos_sim.masked_fill_(self_mask, -65504)
        # Find positive example -> batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=batch_size // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll


class MeanPooler(nn.Module):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by averaging.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The average over sequence length gives even weighting to each sequence position
        return x.mean(dim=1)


class EsmContrastiveProjectionHead(nn.Module):
    """Contrastive projection head for multi-modal alignment."""

    def __init__(self, d_model: int, contrastive_temperature: float) -> None:
        super().__init__()
        # The projection representions z are trained to become invariant to
        # many gene/protein specific features

        # We use a different projection head for codons and amino acids
        # since, by default, the embeddings fall into different subspaces.
        self.codon_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
        )
        self.aminoacid_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model // 4),
        )

        self.loss_fn = ContrastiveLoss(temperature=contrastive_temperature)
        self.pooler = MeanPooler()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes that the codon embeddings are the first half of the tensor
        # and the aminoacid embeddings are the second half.

        # Pool the sequence embeddings to get a single embedding per sequence
        x = self.pooler(x)  # (batch_size, hidden_size)

        # Collect the codon and aminoacid embeddings separately
        # These have shape (batch_size // 2, hidden_size)
        half_batch_size = x.shape[0] // 2
        codon_embed = x[:half_batch_size]
        aminoacid_embed = x[half_batch_size:]

        # Project the embeddings into a lower dimensional space
        # These have shape (batch_size // 2, projection_size)
        z_codon = self.codon_projection(codon_embed)
        z_aminoacid = self.aminoacid_projection(aminoacid_embed)

        # Concatenate the codon and aminoacid embeddings
        # This has shape (batch_size, projection_size)
        z = torch.cat([z_codon, z_aminoacid], dim=0)

        # Compute the contrastive loss following SimCLR
        return self.loss_fn(z)


@dataclass
class EsmCForContrastiveMaskedLMOutput(ModelOutput):
    """
    Base class for ESM-C for contrastive masked language models outputs.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Masked language modeling (MLM) loss.
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    # The losses output by the model
    loss: torch.FloatTensor | None = None
    contrastive_loss: torch.FloatTensor | None = None
    mlm_loss: torch.FloatTensor | None = None
    codon_mlm_loss: torch.FloatTensor | None = None
    aminoacid_mlm_loss: torch.FloatTensor | None = None

    # The logits output by the model
    codon_logits: torch.FloatTensor | None = None
    aminoacid_logits: torch.FloatTensor | None = None

    # The hidden states output by the model
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None


class EsmCForContrastiveMaskedLM(PreTrainedModel):
    """ESMC for contrastive masked language modeling."""

    # Set the configuration class for the model to use for
    # initialization via from_pretrained()
    config_class = ContrastiveEsmCConfig

    def __init__(self, config: ContrastiveEsmCConfig) -> None:
        super().__init__(config)
        self.config = config

        self.transformer = ESMC(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            contrastive_temperature=config.contrastive_temperature,
            use_flash_attn=config.use_flash_attn,
        )

        # Regression head for amino acid predictions
        self.lm_head = RegressionHead(config.d_model, 64)

        # Loss function for masked language modeling
        self.loss_fct = nn.CrossEntropyLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self) -> None:
        """Get the output embeddings of the model."""
        # NOTE: get_output_embeddings() must return None to prevent accidental
        # weight tying. See e.g. https://github.com/huggingface/transformers/pull/39339#discussion_r2219126400
        return None

    def _compute_mlm_loss(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
    ) -> torch.FloatTensor:
        """Compute the masked language modeling loss."""
        return self.loss_fct(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
        )

    def forward(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        aminoacid_input_ids: torch.LongTensor | None = None,
        aminoacid_attention_mask: torch.Tensor | None = None,
        aminoacid_labels: torch.LongTensor | None = None,
        codon_input_ids: torch.LongTensor | None = None,
        codon_attention_mask: torch.Tensor | None = None,
        codon_labels: torch.LongTensor | None = None,
        decode_aminoacid_head: bool = False,
        decode_codon_head: bool = False,
        compute_contrastive_loss: bool = False,
    ) -> EsmCForContrastiveMaskedLMOutput:
        """Forward pass for the ESM for contrastive masked language modeling.

        Parameters
        ----------
        aminoacid_input_ids: torch.LongTensor | None
            Input ids for the amino acid sequences (batch_size, sequence_length)
        aminoacid_attention_mask: torch.Tensor | None
            Attention mask for the amino acid sequences (batch_size, sequence_length)
        aminoacid_labels: torch.LongTensor | None
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`. (batch_size, sequence_length)
        codon_input_ids: torch.LongTensor | None
            Input ids for the codon sequences (batch_size, sequence_length)
        codon_attention_mask: torch.Tensor | None
            Attention mask for the codon sequences (batch_size, sequence_length)
        codon_labels: torch.LongTensor | None
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`. (batch_size, sequence_length)
        decode_aminoacid_head (`bool`, optional, defaults to False):
            Whether to use the amino acid head for prediction, regardless of config settings.
        decode_codon_head (`bool`, optional, defaults to False):
            Whether to use the codon head for prediction, regardless of config settings.
        compute_contrastive_loss: bool
            Whether to compute the contrastive loss.
        """
        # Validate the input arguments
        # ================================================================
        if (
            aminoacid_input_ids is not None
            and aminoacid_attention_mask is None
        ):
            raise ValueError(
                'aminoacid_attention_mask must be provided if '
                'aminoacid_input_ids is not None',
            )
        if codon_input_ids is not None and codon_attention_mask is None:
            raise ValueError(
                'codon_attention_mask must be provided if '
                'codon_input_ids is not None',
            )

        # NOTE: We need to pass the codon and amino acid input_ids and
        # attention_masks separately since the distrubted sampler shuffles the
        # codon and amino acid sequences and there was no way to guarantee that
        # the codon and amino acid sequences would be indexed properly for the
        # contrastive loss.

        # Forward pass through the transformer
        # ================================================================
        # Pack the amino acid and codon input_ids and attention_mask into a
        # single tensor. This avoids two separate forward passes through the
        # model but effectively doubles the batch size.
        if aminoacid_input_ids is not None and codon_input_ids is not None:
            assert codon_attention_mask is not None
            assert aminoacid_attention_mask is not None
            input_ids = torch.cat(
                [codon_input_ids, aminoacid_input_ids],
                dim=0,
            )
            attention_mask = torch.cat(
                [codon_attention_mask, aminoacid_attention_mask],
                dim=0,
            )
        elif aminoacid_input_ids is not None:
            input_ids = aminoacid_input_ids
        elif codon_input_ids is not None:
            input_ids = codon_input_ids
        else:
            raise ValueError(
                'Must provide either codon_input_ids or aminoacid_input_ids',
            )

        # Pass the input_ids and attention_mask to the transformer
        outputs = cast(
            ESMCOutput,
            self.transformer(
                sequence_tokens=input_ids,
                sequence_id=attention_mask,
            ),
        )

        # Token embeddings from the last layer of the transformer.
        # This has shape: (batch_size, seq_length, d_model)
        sequence_output = outputs.hidden_states[-1]

        # Handle multi-modal learning for joint and contrastive pre-training
        # ================================================================
        # Initialize contrastive loss and logits to None
        contrastive_loss, codon_logits, aminoacid_logits = None, None, None

        # If both modalities are passed, then we need to split the sequence
        # output into codon and amino acid embeddings to compute logits using
        # the respective heads.
        if codon_input_ids is not None and aminoacid_input_ids is not None:
            # TODO: Should we require decode_aminoacid_head or decode_codon_head
            # to be None in this case?
            # if decode_aminoacid_head or decode_codon_head:
            #     raise ValueError(
            #         'decode_aminoacid_head and decode_codon_head must be None '
            #         'when both codon_input_ids and aminoacid_input_ids are provided',
            #     )

            # NOTE: sequence_output stores the concatenated codon and amino
            # acid embeddings here.

            # NOTE: this case is primarily used for training the model, it
            # does not require passing decode_aminoacid_head or
            # decode_codon_head.

            # Split the sequence output into codon and amino acid embeddings
            # to compute logits using the respective heads.
            # These have shape (batch_size // 2, seq_length, d_model)
            half_batch_size = sequence_output.shape[0] // 2
            codon_embed = sequence_output[:half_batch_size]
            aminoacid_embed = sequence_output[half_batch_size:]

            # Compute the logits for the codon and amino acid heads
            codon_logits = self.transformer.codon_lm_head(codon_embed)
            aminoacid_logits = self.lm_head(aminoacid_embed)

            # Compute the contrastive loss if requested to align the
            # codon and amino acid embeddings.
            if compute_contrastive_loss:
                contrastive_loss = self.transformer.contrastive_head(
                    sequence_output,
                )

        # Handle multi-modal translation tasks
        # ================================================================

        # 1. Reverse translation case (amino acid -> codon)
        # ----------------------------------------------------------------
        # If (aminoacid_input_ids is not None and decode_codon_head),
        # then sequence_output stores the amino acid embeddings here
        # and the task is to predict the codon labels given the amino acid
        # embeddings. If the codon_labels are provided, the codon_mlm_loss
        # represents the accuracy of the reverse translation task. The
        # aminoacid_mlm_loss is also computed for completeness to enable
        # checking the perplexity of the input amino acid sequences.
        # ----------------------------------------------------------------

        # 2. Forward translation case (codon -> amino acid)
        # ----------------------------------------------------------------
        # If (codon_input_ids is not None and decode_aminoacid_head),
        # then sequence_output stores the codon embeddings here and the
        # task is to predict the amino acid labels given the codon embeddings.
        # If the aminoacid_labels are provided, the aminoacid_mlm_loss
        # represents the accuracy of the forward translation task. The
        # codon_mlm_loss is also computed for completeness to enable checking
        # the model perplexity of the input codon sequences.
        # ----------------------------------------------------------------
        elif (aminoacid_input_ids is not None and decode_codon_head) or (
            codon_input_ids is not None and decode_aminoacid_head
        ):
            # Predict the codon logits given the embeddings
            codon_logits = self.transformer.codon_lm_head(sequence_output)
            # Predict the amino acid logits given the embeddings
            aminoacid_logits = self.lm_head(sequence_output)

        # 3. Standard codon language modeling task
        # ================================================================
        elif codon_input_ids is not None:
            # NOTE: sequence_output stores the codon embeddings here.
            # Compute the logits for the codon head given the embeddings
            codon_logits = self.transformer.codon_lm_head(sequence_output)

        # 4. Standard amino acid language modeling task
        # ================================================================
        elif aminoacid_input_ids is not None:
            # NOTE: sequence_output stores the amino acid embeddings here.
            # Compute the logits for the amino acid head given the embeddings
            aminoacid_logits = self.lm_head(sequence_output)

        # If no input_ids are provided, raise an error (should never happen)
        else:
            raise ValueError(
                'Must provide either codon_input_ids or aminoacid_input_ids',
            )

        # Compute masked language modeling losses using the logits and labels
        # ================================================================
        # If codon labels and logits are available, compute the MLM loss
        # for the codon head, otherwise set the loss to None.
        if codon_labels is not None and codon_logits is not None:
            codon_mlm_loss = self._compute_mlm_loss(codon_logits, codon_labels)
        else:
            codon_mlm_loss = None

        # If the amino acid labels and logits are available, compute the MLM
        # loss for the amino acid head, otherwise set the loss to None.
        if aminoacid_labels is not None and aminoacid_logits is not None:
            aminoacid_mlm_loss = self._compute_mlm_loss(
                aminoacid_logits,
                aminoacid_labels,
            )
        else:
            aminoacid_mlm_loss = None

        # If both amino acid and codon losses are available, compute the
        # total MLM loss for completeness.
        if codon_mlm_loss is not None and aminoacid_mlm_loss is not None:
            # Average the two losses to get the total MLM loss
            mlm_loss = (codon_mlm_loss + aminoacid_mlm_loss) / 2.0

        # If only codon loss is available, set the MLM loss to it
        elif codon_mlm_loss is not None:
            mlm_loss = codon_mlm_loss

        # If only amino acid loss is available, set the MLM loss to it
        elif aminoacid_mlm_loss is not None:
            mlm_loss = aminoacid_mlm_loss

        # If no losses are available, set the MLM loss to None
        else:
            mlm_loss = None

        # Compute the total multi-modal loss (MLM loss + Contrastive loss)
        # ================================================================
        # If the contrastive loss is available, add it to the MLM loss
        if contrastive_loss is not None and mlm_loss is not None:
            total_loss = mlm_loss + contrastive_loss

        # If only the contrastive loss is available, set the total loss to it
        elif contrastive_loss is not None:
            total_loss = contrastive_loss

        # If only the MLM loss is available, set the total loss to it
        elif mlm_loss is not None:
            total_loss = mlm_loss

        # If no losses are available, set the total loss to None
        else:
            total_loss = None

        # Return the outputs
        # ================================================================
        # NOTE: Since the codon and amino acid heads have different vocab
        # sizes, we need to compute and store the logits for each head
        # separately.
        return EsmCForContrastiveMaskedLMOutput(
            loss=total_loss,
            contrastive_loss=contrastive_loss,
            mlm_loss=mlm_loss,
            codon_mlm_loss=codon_mlm_loss,
            aminoacid_mlm_loss=aminoacid_mlm_loss,
            codon_logits=codon_logits,
            aminoacid_logits=aminoacid_logits,
            hidden_states=outputs.hidden_states,
            attentions=None,  # TODO: Add attentions
        )


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from transformers import EsmTokenizer

    from genslm_esm.dataset import FastaDataset
    from genslm_esm.dataset import GenSLMColatorForLanguageModeling

    model_path = '/nfs/lambda_stor_01/homes/abrace/projects/genslm/src/genslm-tutorial-05-2025/model/checkpoint-203847'

    # Load the model from the checkpoint
    model = EsmCForContrastiveMaskedLM.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    print('Reloaded model:')
    print(model)

    # Get the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    # Convert the model to bfloat16 if not on CPU
    if device.type != 'cpu':
        model = model.to(torch.bfloat16)

    print(f'Model is on device: {next(model.parameters()).device}')
    print(f'Model dtype: {next(model.parameters()).dtype}')

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    print('Tokenizer:')
    print(tokenizer)

    # Test sequences
    sequences = [
        'ATGAAGGTACTACCACAAGAAACTGTAAGAATTGGA',
        'ATGGACAAAACACATATTCGACTATCTGTTGACAATCCATTTGCAAAACTA',
    ]

    # The dataset splits the sequences into codons
    dataset = FastaDataset(
        sequences=sequences,
        return_codon=True,
        return_aminoacid=True,
    )

    # Create the collator
    collator = GenSLMColatorForLanguageModeling(
        return_codon=True,
        return_aminoacid=True,
        tokenizer=tokenizer,
    )

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # Iterate over the dataloader
    for batch in dataloader:
        batch = batch.to(device)
        print(batch)
        outputs = model(**batch)
        print(outputs.loss)
