"""Adapted PyTorch ESM model."""
# braceal 08/31/23: I have modified the original ESM Huggingface model to add a contrastive loss head.
from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.esm.modeling_esm import (
    EsmForMaskedLM,
    EsmPooler,
    MaskedLMOutput,
    EsmLMHead,
)
from transformers.models.esm.configuration_esm import EsmConfig
from transformers import logging

logger = logging.get_logger(__name__)


# TODO: Currently not used
class ContrastiveEsmConfig(EsmConfig):
    """Add contrastive loss parameters to the ESM config."""

    def __init__(
        self,
        compute_contrastive_loss: bool = False,
        contrastive_temperature: float = 0.1,
        contrastive_pooler: str = "mean",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.compute_contrastive_loss = compute_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_pooler = contrastive_pooler


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
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z[:, None, :], z[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=z.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=batch_size // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()
        return nll


class MeanPooler(nn.Module):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by averaging."""

    def __init__(self, config) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The average over sequence length gives even weighting to each sequence position
        return x.mean(dim=1)


class FirstPooler(EsmPooler):
    """Reduces the sequence embeddings (batch_size, seq_length, hidden_size)
    to a single embedding (batch_size, hidden_size) by taking the first hidden state."""


POOLER_DISPATCH = {"mean": MeanPooler, "first": FirstPooler}


class EsmContrastiveProjectionHead(nn.Module):
    def __init__(self, config: ContrastiveEsmConfig) -> None:
        super().__init__()
        # The projection representions z are trained to become invariant to
        # many gene/protein specific features

        # We use a different projection head for codons and amino acids
        # since, by default, the embeddings fall into different subspaces.
        self.codon_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
        )
        self.aminoacid_projection = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size // 4),
        )

        self.loss_fn = ContrastiveLoss(temperature=config.contrastive_temperature)
        self.pooler = POOLER_DISPATCH[config.contrastive_pooler](config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Assumes that the codon embeddings are the first half of the tensor
        # and the aminoacid embeddings are the second half (although, due to
        # symmetry, the actual order doesn't matter as long as it's consistent.)

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


class EsmForContrastiveMaskedLM(EsmForMaskedLM):
    def __init__(
        self,
        config: EsmConfig,
        compute_contrastive_loss: bool = False,
        contrastive_temperature: float = 0.1,
        contrastive_pooler: str = "mean",
    ):
        super().__init__(config)
        # Inject contrastive loss parameters into the config
        config.compute_contrastive_loss = compute_contrastive_loss
        config.contrastive_temperature = contrastive_temperature
        config.contrastive_pooler = contrastive_pooler

        #if config.compute_contrastive_loss:
        self.contrastive_head = EsmContrastiveProjectionHead(config)
        
        # Initialize weights and apply final processing
        self.post_init()

    def resize_model_vocab(self, new_vocab_size: int) -> None:
        # Inject new vocabulary (modifies config)
        if new_vocab_size != self.config.vocab_size:
            logger.warning(
                "Resizing token embedding layer from {} to {}. This reinitializes the EsmLMHead and input embedding layer weights".format(
                    self.config.vocab_size, new_vocab_size
                )
            )
            self.resize_token_embeddings(new_vocab_size)
            # Make a new lm_head with uninitialized weights using the correct shape
            self.lm_head = EsmLMHead(self.config)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        kwargs (`Dict[str, any]`, optional, defaults to *{}*):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.esm(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            labels = labels.to(prediction_scores.device)
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        # Custom logic to compute a contrastive loss between Codons and Amino Acid embeddings.
        # Everything else in this function is the same as the base class implementation.
        if self.config.compute_contrastive_loss:
            # Compute the contrastive loss following SimCLR and add it to the masked language modeling loss
            masked_lm_loss += self.contrastive_head(sequence_output)

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
