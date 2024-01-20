"""Adapted PyTorch ESM model."""
# braceal 08/31/23: modified the original ESM Huggingface model to add a contrastive loss head.
# braceal 01/20/24: modified the implementation to separate lm_head's for codons and amino acids.
from typing import Optional, Tuple, Union
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging, EsmTokenizer
from transformers.models.esm.configuration_esm import EsmConfig
from transformers.models.esm.modeling_esm import (
    EsmForMaskedLM,
    EsmLMHead,
    EsmPooler,
    MaskedLMOutput,
)
from genslm_esm.dataset import translation_table

logger = logging.get_logger(__name__)


# TODO: Only used for a type hint
class ContrastiveEsmConfig(EsmConfig):
    """Add contrastive loss parameters to the ESM config."""

    def __init__(
        self,
        compute_aminoacid_loss: bool = True,
        compute_codon_loss: bool = False,
        compute_contrastive_loss: bool = False,
        contrastive_temperature: float = 0.1,
        contrastive_pooler: str = "mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.compute_aminoacid_loss = compute_aminoacid_loss
        self.compute_contrastive_loss = compute_contrastive_loss
        self.compute_codon_loss = compute_codon_loss
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


class EsmForContrastiveMaskedLM(EsmForMaskedLM):
    def __init__(
        self,
        config: EsmConfig,
        compute_aminoacid_loss: bool = True,
        compute_codon_loss: bool = False,
        compute_contrastive_loss: bool = False,
        contrastive_temperature: float = 0.1,
        contrastive_pooler: str = "mean",
    ) -> None:
        super().__init__(config)
        # Inject contrastive loss parameters into the config
        config.compute_aminoacid_loss = compute_aminoacid_loss
        config.compute_codon_loss = compute_codon_loss
        config.compute_contrastive_loss = compute_contrastive_loss
        config.contrastive_temperature = contrastive_temperature
        config.contrastive_pooler = contrastive_pooler

        # Only used if compute_contrastive_loss is True
        self.contrastive_head = EsmContrastiveProjectionHead(config)

        # Only used if compute_codon_loss is True. Make a new config with the
        # same parameters as the original config but with a different vocab size
        # (the number of codons 64 + special tokens)
        codon_config = EsmConfig(**config.to_dict())
        codon_config.vocab_size = 69
        self.codon_lm_head = EsmLMHead(codon_config)

        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def update_model_weights(self, tokenizer: EsmTokenizer) -> None:
        # If the tokenizer has the same number of tokens as the model, then
        # the model weights are already updated and we can return early.
        new_vocab_size = len(tokenizer)
        if new_vocab_size == self.config.vocab_size:
            return

        logger.warning(
            "Resizing token embedding layer from {} to {} and initializing codon_lm_head with amino acid lm_head".format(
                self.config.vocab_size, new_vocab_size
            )
        )

        # Get the original token embedding matrix (TEM)
        original_tem = deepcopy(self.esm.embeddings.word_embeddings.weight)
        # Get the original amino acid lm head
        original_lm_head = deepcopy(self.lm_head)

        # Inject new vocabulary (modifies config and TEM)
        self.resize_token_embeddings(new_vocab_size)

        # Get a reference to the new TEM matrix
        new_tem = self.esm.embeddings.word_embeddings.weight

        # Set each of the new codon representations equal to their corresponding
        # amino acid representations (note that amino acid representations are
        # unchanged by vocabulary expansion)
        # Here we are looping over the entire vocab, but we are only using the
        # vocab elements which are codons, so we name the variables accordingly.
        for codon, codon_id in tokenizer.get_vocab().items():
            # Note: the translation table maps stop codons to "" and the get
            # function maps the special tokens and amino acids to "" as well.
            aminoacid = translation_table.get(codon, "")
            if aminoacid:
                aminoacid_id = tokenizer._token_to_id[aminoacid]
                # The new TEM matrix aminoacid represenations are the same
                # as the original TEM (they are preserved during resizing)
                new_tem[codon_id] = deepcopy(new_tem[aminoacid_id])
                assert torch.equal(original_tem[aminoacid_id], new_tem[codon_id])

        # Check that the TEM matrix was updated correctly
        assert torch.equal(original_tem, new_tem[: len(original_tem)])
        assert torch.equal(self.esm.embeddings.word_embeddings.weight, new_tem)

        # Now that the TEM is updated, we also need to update the lm heads
        # for amino acids and codons.

        # Resizing vocab changes the original aminoacid lm_head, we want to
        # change it back since we have a separate lm_head for the codon vocabulary.
        self.lm_head = deepcopy(original_lm_head)
        # Check that the original amino acid lm_head was reloaded correctly
        assert torch.equal(original_lm_head.dense.weight, self.lm_head.dense.weight)
        assert torch.equal(original_lm_head.dense.bias, self.lm_head.dense.bias)
        assert torch.equal(
            original_lm_head.layer_norm.weight, self.lm_head.layer_norm.weight
        )
        assert torch.equal(
            original_lm_head.layer_norm.bias, self.lm_head.layer_norm.bias
        )
        assert torch.equal(original_lm_head.decoder.weight, self.lm_head.decoder.weight)
        assert torch.equal(original_lm_head.bias, self.lm_head.bias)

        # We initialize the codon_lm_head dense and layer_norm layers
        # with the vocab-size invariant amino acid counterparts
        self.codon_lm_head.dense = deepcopy(self.lm_head.dense)
        self.codon_lm_head.layer_norm = deepcopy(self.lm_head.layer_norm)
        # Check that the weights have been updated properly
        assert torch.equal(self.codon_lm_head.dense.weight, self.lm_head.dense.weight)
        assert torch.equal(self.codon_lm_head.dense.bias, self.lm_head.dense.bias)
        assert torch.equal(
            self.codon_lm_head.layer_norm.weight, self.lm_head.layer_norm.weight
        )
        assert torch.equal(
            self.codon_lm_head.layer_norm.bias, self.lm_head.layer_norm.bias
        )
        assert id(self.codon_lm_head.dense) != id(self.lm_head.dense)
        assert id(self.codon_lm_head.layer_norm) != id(self.lm_head.layer_norm)

        # Get the original aminoacid lm_head decoder weights
        aminoacid_decoder = original_lm_head.decoder.weight
        aminoacid_bias = original_lm_head.bias

        # Get references to the codon_lm_head components to change
        codon_decoder = self.codon_lm_head.decoder.weight
        codon_bias = self.codon_lm_head.bias

        # Since the vocabulary is combined, there is a total vocab size of 97
        # (amino acid + codons + special). However, the codon_lm_head decoder
        # and bias only has an output size of 69 (codon + special). Our goals
        # are to (1) initialize the weights relevant to special tokens to be
        # identical to the original amino acid lm_head weights, and (2) to
        # initialize the weights for each codon to the weights of the translated
        # amino acid in the original lm head. Here the enumeration is taken over
        # the 69 codon/special tokens. The first 5 tokens are the special tokens
        # which are set to the first 5 dimensions of the codon_decoder/bias.
        for codon_idx, codon in enumerate(tokenizer.added_tokens_decoder.values()):
            aminoacid = translation_table.get(str(codon), "")
            # The codon could be a special token (<cls>, <pad>, <eos>, <unk>, <mask>)
            if aminoacid or codon.special:
                # Get the id of the aminoacid/special token corresponding to the current codon
                aminoacid_id = tokenizer._token_to_id[
                    str(codon) if codon.special else aminoacid
                ]
                # Update the codon_decoder/bias with the corresponding aminoacid weights
                codon_decoder[codon_idx] = deepcopy(aminoacid_decoder[aminoacid_id])
                codon_bias[codon_idx] = deepcopy(aminoacid_bias[aminoacid_id])
                # Check that the update was successfull
                assert torch.equal(
                    codon_decoder[codon_idx], aminoacid_decoder[aminoacid_id]
                )
                assert torch.equal(codon_bias[codon_idx], aminoacid_bias[aminoacid_id])
                assert torch.equal(codon_decoder, self.codon_lm_head.decoder.weight)
                assert torch.equal(codon_bias, self.codon_lm_head.bias)

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
        sequence_output = outputs[0]  # (batch_size, seq_length, hidden_size)

        # Compute the logits / prediction scores for each head
        if self.config.compute_aminoacid_loss and self.config.compute_codon_loss:
            # Split the sequence output into codon and amino acid embeddings
            # These have shape (batch_size // 2, seq_length, hidden_size)
            half_batch_size = sequence_output.shape[0] // 2
            codon_embed = sequence_output[:half_batch_size]
            amino_embed = sequence_output[half_batch_size:]
            codon_prediction_scores = self.codon_lm_head(codon_embed)
            amino_prediction_scores = self.lm_head(amino_embed)
            # The prediction scores have different vocab sizes, so we can't concatenate them.
            # Instead, we return the aminoacid scores (during inference, set either
            # compute_aminoacid_loss or compute_codon_loss to False)
            prediction_scores = amino_prediction_scores
        elif self.config.compute_aminoacid_loss:
            prediction_scores = self.lm_head(sequence_output)
        elif self.config.compute_codon_loss:
            prediction_scores = self.codon_lm_head(sequence_output)
        else:
            raise ValueError(
                "Either compute_aminoacid_loss or compute_codon_loss must be True"
            )

        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            # Compute the masked language modeling loss for each head
            if self.config.compute_aminoacid_loss and self.config.compute_codon_loss:
                # Split the labels into codon and amino acid labels
                # These have shape (batch_size // 2, seq_length)
                half_batch_size = labels.shape[0] // 2
                codon_labels = labels[:half_batch_size]
                aminoacid_labels = labels[half_batch_size:]
                # Compute the masked language modeling loss for each head
                codon_masked_lm_loss = loss_fct(
                    codon_prediction_scores.view(-1, codon_prediction_scores.shape[-1]),
                    codon_labels.view(-1),
                )
                aminoacid_masked_lm_loss = loss_fct(
                    amino_prediction_scores.view(-1, amino_prediction_scores.shape[-1]),
                    aminoacid_labels.view(-1),
                )
                # Add the two losses together
                masked_lm_loss = (codon_masked_lm_loss + aminoacid_masked_lm_loss) / 2.0

            else:
                # # (-1, vocab_size) is the shape of the prediction scores
                masked_lm_loss = loss_fct(
                    prediction_scores.view(-1, prediction_scores.shape[-1]),
                    labels.view(-1),
                )

            # vocab_size = prediction_scores.shape[-1]
            # labels = labels.to(prediction_scores.device)
            # masked_lm_loss = loss_fct(
            #     prediction_scores.view(-1, vocab_size), labels.view(-1)
            # )

        # Custom logic to compute a contrastive loss between Codons and Amino Acid embeddings.
        # Everything else in this function is the same as the base class implementation.
        if labels is not None and self.config.compute_contrastive_loss:
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
