"""Modeling code for the GenSLM-ESMC model."""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import cast

import einops
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from einops import repeat
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

try:
    from flash_attn import flash_attn_varlen_qkvpacked_func
    from flash_attn.bert_padding import pad_input
    from flash_attn.bert_padding import unpad_input
    from flash_attn.ops.triton.rotary import (
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

from genslm_esm.configuration import GenslmEsmcConfig

# ESMC pad token id
ESMC_PAD_TOKEN_ID = 1


def rotate_half(x: torch.Tensor, interleaved: bool = False) -> torch.Tensor:
    """Rotate half of the dimensions of the tensor."""
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


def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    interleaved: bool = False,
    _inplace: bool = False,
) -> torch.Tensor:
    """
    Apply rotary embeddings to the tensor.

    Parameters
    ----------
        x: (batch_size, seqlen, nheads, headdim)
        cos: (seqlen, rotary_dim / 2)
        sin: (seqlen, rotary_dim / 2)
        interleaved: Whether to use interleaved rotary embeddings.
        _inplace: Whether to use inplace operations.

    Returns
    -------
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
    """The rotary position embeddings from RoFormer_ (Su et. al).

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
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: float | None = None,
        scaling_factor: float = 1.0,
        pos_idx_in_fp32: bool = True,
        device: torch.device | None = None,
    ) -> None:
        """
        Initialize the RotaryEmbedding.

        Parameters
        ----------
        dim: int
            The dimension of the embeddings.
        base: float
            The base of the exponential function.
        interleaved: bool
            If True, rotate pairs of even and odd dimensions (GPT-J style)
            instead of 1st half and 2nd half (GPT-NeoX style).
        scale_base: float | None
            The base of the exponential function for the scale.
        scaling_factor: float
            The scaling factor for the position indices. RotaryEmbedding
            extended with linear scaling.
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1]
            are in fp32, otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when
            we construct the position indices, we use the dtype of
            self.inv_freq. In most cases this would be fp32, but if the model
            is trained in pure bf16 (not mixed precision), then self.inv_freq
            would be bf16, and the position indices are also in bf16. Because
            of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0)
            , the embeddings for some positions will coincide. To maintain
            compatibility with models previously trained in pure bf16, we add
            this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        device: torch.device | None
            The device to use for the embeddings.
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

    def reset_parameters(self) -> None:
        """Reset the parameters of the RotaryEmbedding."""
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

    def _compute_inv_freq(
        self,
        device: torch.device | None = None,
    ) -> torch.Tensor:
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

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
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
            # We want fp32 here, not self.inv_freq.dtype, since the model
            # could be loaded in bf16. And the output of arange can be quite
            # large, so bf16 would lose a lot of precision. However, for
            # compatibility reason, we add an option to use the dtype of
            # self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied
                # with t, and the output will be large. Having it in bf16 will
                # lose a lot of precision and cause the cos & sin output to
                # change significantly. We want to recompute self.inv_freq if
                # it was not loaded in fp32
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
        """Apply rotary embeddings to the query and key tensors.

        Parameters
        ----------
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in
            is only the last token in the batch.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The query and key tensors with rotary embeddings applied.

        Raises
        ------
        ValueError: If the scale is not None.
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
            )
        else:
            raise ValueError('Scale is not None')


class TritonRotaryEmbedding(RotaryEmbedding):
    """Triton rotary embedding implementation."""

    def forward(  # type: ignore[override]
        self,
        qkv: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        """Apply rotary embeddings to the query, key, and value tensors.

        Parameters
        ----------
        qkv: (n, 3, nheads, headdim)
        cu_seqlens: cumulative sequence lengths
        max_seqlen: max sequence length

        Returns
        -------
        torch.Tensor
            The query, key, and value tensors with rotary embeddings applied.
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
    """Multi-head attention implementation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ) -> None:
        """Initialize the MultiHeadAttention.

        Parameters
        ----------
        d_model: int
            The dimension of the model.
        n_heads: int
            The number of heads.
        bias: bool
            Whether to use bias.
        qk_layernorm: bool
            Whether to use layer normalization on the query and key.
        """
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

    def _apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = q.unflatten(-1, (self.n_heads, self.d_head))
        k = k.unflatten(-1, (self.n_heads, self.d_head))
        q, k = self.rotary(q, k)
        q = q.flatten(-2, -1)
        k = k.flatten(-2, -1)
        return q, k

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MultiHeadAttention.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        seq_id: torch.Tensor
            The sequence id tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        qkv_bld3 = self.layernorm_qkv(x)
        query_bld, key_bld, value_bld = torch.chunk(qkv_bld3, 3, dim=-1)
        query_bld, key_bld = (
            self.q_ln(query_bld).to(query_bld.dtype),
            self.k_ln(key_bld).to(query_bld.dtype),
        )
        query_bld, key_bld = self._apply_rotary(query_bld, key_bld)

        reshaper = functools.partial(
            einops.rearrange,
            pattern='b s (h d) -> b h s d',
            h=self.n_heads,
        )

        query_bhld, key_bhld, value_bhld = map(
            reshaper,
            (query_bld, key_bld, value_bld),
        )

        if seq_id is not None:
            # Where True, enable participation in attention.
            mask_bll = seq_id.unsqueeze(-1) == seq_id.unsqueeze(-2)
            mask_bhll = mask_bll.unsqueeze(1)

            context_bhld = F.scaled_dot_product_attention(
                query_bhld,
                key_bhld,
                value_bhld,
                mask_bhll,
            )
        else:
            # Shortcut, if we don't use attention biases then torch
            # will autoselect flashattention as the implementation
            context_bhld = F.scaled_dot_product_attention(
                query_bhld,
                key_bhld,
                value_bhld,
            )

        context_bld = einops.rearrange(context_bhld, 'b h s d -> b s (h d)')

        return self.out_proj(context_bld)


class FlashMultiHeadAttention(MultiHeadAttention):
    """Flash multi-head attention implementation."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        bias: bool = False,
        qk_layernorm: bool = True,
    ) -> None:
        """Initialize the FlashMultiHeadAttention.

        Parameters
        ----------
        d_model: int
            The dimension of the model.
        n_heads: int
            The number of heads.
        bias: bool
            Whether to use bias.
        qk_layernorm: bool
            Whether to use layer normalization on the query and key.
        """
        super().__init__(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            qk_layernorm=qk_layernorm,
        )

        # Flash attention rotary.
        self.rotary = TritonRotaryEmbedding(d_model // n_heads)

    def forward(self, x: torch.Tensor, seq_id: torch.Tensor) -> torch.Tensor:
        """Forward pass for the FlashMultiHeadAttention.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.
        seq_id: torch.Tensor
            The sequence id tensor.
        """
        assert seq_id.dtype == torch.bool

        seqlens = seq_id.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = F.pad(
            torch.cumsum(seqlens, dim=0, dtype=torch.int32),
            (1, 0),
        )
        max_seqlen = seqlens.max().item()

        qkv_nd3 = self.layernorm_qkv(x)

        query_nd, key_nd, value_nd = torch.chunk(qkv_nd3, 3, dim=-1)
        query_nd, key_nd = (
            self.q_ln(query_nd).to(query_nd.dtype),
            self.k_ln(key_nd).to(query_nd.dtype),
        )

        qkv_n3d = torch.stack([query_nd, key_nd, value_nd], dim=1)
        qkv_n3hd = einops.rearrange(
            qkv_n3d,
            pattern='n a (h d) -> n a h d',
            h=self.n_heads,
        )
        qkv_n3hd = self.rotary(qkv_n3hd, cu_seqlens, max_seqlen)

        context_nhd = flash_attn_varlen_qkvpacked_func(
            qkv_n3hd,
            cu_seqlens,
            max_seqlen,
            softmax_scale=self.d_head**-0.5,
        )
        context_nd = einops.rearrange(context_nhd, 'n h d -> n (h d)')

        return self.out_proj(context_nd)


def swiglu_correction_fn(expansion_ratio: float, d_model: int) -> int:
    """Correct the hidden dimension for the SwiGLU feed-forward network.

    Parameters
    ----------
    expansion_ratio: float
        The expansion ratio.
    d_model: int
        The dimension of the model.

    Returns
    -------
    int
        The corrected hidden dimension.
    """
    # Set hidden dimension to nearest multiple of 256 after expansion ratio
    return int(((expansion_ratio * d_model) + 255) // 256 * 256)


class SwiGLU(nn.Module):
    """SwiGLU activation function as an nn.Module.

    Allows it to be used within nn.Sequential. This module splits the input
    tensor along the last dimension and applies the SiLU (Swish) activation
    function to the first half, then multiplies it by the second half.
    """

    def __init__(self) -> None:
        """Initialize SwiGLU."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for SwiGLU.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


def swiglu_ln_ffn(
    d_model: int,
    expansion_ratio: float,
    bias: bool,
) -> nn.Sequential:
    """Create a SwiGLU feed-forward network.

    Parameters
    ----------
    d_model: int
        The dimension of the model.
    expansion_ratio: float
        The expansion ratio.
    bias: bool
        Whether to use bias.

    Returns
    -------
    nn.Sequential
        The SwiGLU feed-forward network.
    """
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
    """Unified transformer block.

    This class defines a transformer block with the standard multi-head
    attention mechanism.
    """

    def __init__(  # noqa: PLR0913
        self,
        d_model: int,
        n_heads: int,
        use_plain_attn: bool = True,
        use_flash_attn: bool = False,
        bias: bool = False,
        expansion_ratio: float = 4.0,
        residue_scaling_factor: float = 1,
        qk_layernorm: bool = True,
    ) -> None:
        """Initialize the UnifiedTransformerBlock.

        Parameters
        ----------
        d_model: int
            The dimensionality of the input and output features of the
            transformer block.
        n_heads: int
            The number of attention heads in the multi-head attention
            mechanism.
        use_plain_attn: bool
            Whether to use plain attention.
        use_flash_attn: bool
            Whether to use flash attention.
        bias: bool
            Whether to use bias.
        expansion_ratio: float
            The expansion ratio.
        residue_scaling_factor: float
            The residue scaling factor.
        qk_layernorm: bool
            Whether to use layer normalization on the query and key.
        """
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
            Input tensor to the transformer block, typically the output from
            the previous layer.
        sequence_id : torch.Tensor[int]
            Tensor containing sequence IDs for each element in the batch,
            used for attention masking.

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
    """A stack of transformer blocks.

    Each block is a UnifiedTransformerBlock with standard multi-head attention.
    """

    def __init__(  # noqa: PLR0913
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
        """Initialize the TransformerStack.

        Parameters
        ----------
        d_model: int
            The dimensionality of the input and output feature vectors.
        n_heads: int
            The number of attention heads.
        n_layers: int
            The number of transformer blocks in the stack.
        scale_residue: bool
            Whether to scale the residue connections in each transformer block.
        bias: bool
            Whether to use bias.
        qk_layernorm: bool
            Whether to use layer normalization on the query and key.
        expansion_ratio: float
            The expansion ratio.
        use_flash_attn: bool
            Whether to use flash attention.
        """
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

        Parameters
        ----------
        x: torch.Tensor
            The input tensor of shape (batch_size, sequence_length, d_model).
        sequence_id: torch.Tensor
            The sequence ID tensor of shape (batch_size, sequence_length).

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
            The output tensor of shape (batch_size, sequence_length, d_model),
            the embedding of shape (batch_size, sequence_length, d_model),
            and the list of hidden states of shape
            (n_layers, batch_size, sequence_length, d_model).
        """
        hiddens = []
        for block in self.blocks:
            x = block(x, sequence_id)
            hiddens.append(x)
        return self.norm(x), x, hiddens


def RegressionHead(  # noqa: N802
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
    The modifiedESMC model implementation.

    Parameters
    ----------
    d_model: int
        The dimensionality of the input and output feature vectors.
    n_heads: int
        The number of attention heads in the transformer layers.
    n_layers: int
        The number of transformer layers.
    contrastive_temperature: float
        The temperature for the contrastive loss.
    use_flash_attn: bool
        Whether to use flash attention.
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

        B, L = x.shape[:2]  # noqa: N806

        # If sequence_id looks like a mask.
        if self._use_flash_attn:
            assert sequence_id.dtype == torch.bool, (  # type: ignore[union-attr]
                'sequence_id must be a boolean mask if Flash Attention is used'
            )
            assert sequence_id.shape == (B, L)  # type: ignore[union-attr]
            assert unpad_input is not None
            x, indices, *_ = unpad_input(x, sequence_id)
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
        hiddens = torch.stack(hiddens, dim=0)

        sequence_logits = self.sequence_head(x)
        output = ESMCOutput(
            sequence_logits=sequence_logits,
            embeddings=x,
            hidden_states=hiddens,
        )
        return output


class ContrastiveLoss(nn.Module):
    """Contrastive SimCLR-style loss for multi-modal alignment.

    Reference: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html#SimCLR
    """

    def __init__(self, temperature: float) -> None:
        """Initialize the ContrastiveLoss.

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
        """Forward pass for the ContrastiveLoss.

        Parameters
        ----------
        z: torch.Tensor
            The embeddings to compute the contrastive loss for.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
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
        # Find positive example -> batch_size // 2 away from the original
        # example
        pos_mask = self_mask.roll(shifts=batch_size // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll


class MeanPooler(nn.Module):
    """Mean pooling layer.

    Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by averaging.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MeanPooler.

        Parameters
        ----------
        x: torch.Tensor
            The embeddings to pool.

        Returns
        -------
        torch.Tensor
            The pooled embeddings.
        """
        # The average over sequence length gives even weighting
        # to each sequence position
        return x.mean(dim=1)


class EsmContrastiveProjectionHead(nn.Module):
    """Contrastive projection head for multi-modal alignment."""

    def __init__(self, d_model: int, contrastive_temperature: float) -> None:
        """Initialize the EsmContrastiveProjectionHead.

        Parameters
        ----------
        d_model: int
            The dimensionality of the input and output features.
        contrastive_temperature: float
            The temperature for the contrastive loss.
        """
        super().__init__()
        # The projection representations z are trained to become invariant to
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
        """Forward pass for the EsmContrastiveProjectionHead.

        Parameters
        ----------
        x: torch.Tensor
            The embeddings to project.

        Returns
        -------
        torch.Tensor
            The contrastive loss.
        """
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
class GenslmEsmcModelOutput(ModelOutput):
    """
    Base class for ESM-C for contrastive masked language models outputs.

    Parameters
    ----------
    loss: torch.FloatTensor
        (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when
        `labels` is provided) Masked language modeling (MLM) loss.
    logits: torch.FloatTensor
        Prediction scores of the language modeling head (scores for each
        vocabulary token before SoftMax) of shape `(batch_size, sequence_length
        , config.vocab_size)`
    hidden_states: tuple[torch.FloatTensor, ...]
        (`tuple(torch.FloatTensor)`, *optional*, returned when
        `output_hidden_states=True` is passed or when
        `config.output_hidden_states=True`) Tuple of `torch.FloatTensor`
        (one for the output of the embeddings, if the model has an embedding
        layer, + one for the output of each layer) of shape `(batch_size,
        sequence_length, hidden_size)`.
    attentions: tuple[torch.FloatTensor, ...]
        Attentions weights after the attention softmax, used to compute the
        weighted average in the self-attention heads.
    """

    # TODO: Update this docstring

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


class GenslmEsmcModel(PreTrainedModel):
    """GenSLM-ESMC model for contrastive masked language modeling."""

    # Set the model_type to 'genslm-esmc'
    model_type = 'genslm-esmc'

    # Set the configuration class for the model to use for
    # initialization via from_pretrained()
    config_class = GenslmEsmcConfig

    def __init__(self, config: GenslmEsmcConfig) -> None:
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
    ) -> GenslmEsmcModelOutput:
        """Forward pass for the ESM for contrastive masked language modeling.

        Parameters
        ----------
        aminoacid_input_ids: torch.LongTensor | None
            Input ids for the amino acid sequences
            (batch_size, sequence_length)
        aminoacid_attention_mask: torch.Tensor | None
            Attention mask for the amino acid sequences
            (batch_size, sequence_length)
        aminoacid_labels: torch.LongTensor | None
            Labels for computing the masked language modeling loss.
            Indices should be in `[-100, 0, ..., config.vocab_size]`
            (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with
            labels in `[0, ..., config.vocab_size]`.
            (batch_size, sequence_length)
        codon_input_ids: torch.LongTensor | None
            Input ids for the codon sequences (batch_size, sequence_length)
        codon_attention_mask: torch.Tensor | None
            Attention mask for the codon sequences
            (batch_size, sequence_length)
        codon_labels: torch.LongTensor | None
            Labels for computing the masked language modeling loss. Indices
            should be in `[-100, 0, ..., config.vocab_size]`
            (see `input_ids` docstring) Tokens with indices set to `-100`
            are ignored (masked), the loss is only computed for the tokens
            with labels in `[0, ..., config.vocab_size]`.
            (batch_size, sequence_length)
        decode_aminoacid_head (`bool`, optional, defaults to False):
            Whether to use the amino acid head for prediction, regardless of
            config settings.
        decode_codon_head (`bool`, optional, defaults to False):
            Whether to use the codon head for prediction, regardless of
            config settings.
        compute_contrastive_loss: bool
            Whether to compute the contrastive loss.
        """
        # TODO: Update this docstring

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
        # attention_masks separately since the distributed sampler shuffles the
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
        sequence_output = outputs.hidden_states[-1]  # type: ignore[index]

        # Handle multi-modal learning for joint and contrastive pre-training
        # ================================================================
        # Initialize contrastive loss and logits to None
        contrastive_loss, codon_logits, aminoacid_logits = None, None, None

        # If both modalities are passed, then we need to split the sequence
        # output into codon and amino acid embeddings to compute logits using
        # the respective heads.
        if codon_input_ids is not None and aminoacid_input_ids is not None:
            # TODO: Should we require decode_aminoacid_head or
            # decode_codon_head to be None in this case?

            # if decode_aminoacid_head or decode_codon_head:
            #     raise ValueError(
            #         'decode_aminoacid_head and decode_codon_head must be None ' # noqa: E501
            #         'when both codon_input_ids and aminoacid_input_ids are provided', # noqa: E501
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
        return GenslmEsmcModelOutput(
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
