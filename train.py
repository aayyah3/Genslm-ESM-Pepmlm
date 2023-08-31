import re
import h5py
import torch
import torch.nn as nn
from pathlib import Path
from typing import Any, Dict, List
from dataclasses import dataclass
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    PreTrainedTokenizerFast,
    BatchEncoding,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    EsmForMaskedLM,
    EsmTokenizer,
)
from transformers.models.esm.modeling_esm import EsmLMHead

# Stop codons map to empty strings ""
translation_table = {
    "TTT": "F",
    "TTC": "F",
    "TTA": "L",
    "TTG": "L",
    "TCT": "S",
    "TCC": "S",
    "TCA": "S",
    "TCG": "S",
    "TAT": "Y",
    "TAC": "Y",
    "TGT": "C",
    "TGC": "C",
    "TGG": "W",
    "CTT": "L",
    "CTC": "L",
    "CTA": "L",
    "CTG": "L",
    "CCT": "P",
    "CCC": "P",
    "CCA": "P",
    "CCG": "P",
    "CAT": "H",
    "CAC": "H",
    "CAA": "Q",
    "CAG": "Q",
    "CGT": "R",
    "CGC": "R",
    "CGA": "R",
    "CGG": "R",
    "ATT": "I",
    "ATC": "I",
    "ATA": "I",
    "ATG": "M",
    "ACT": "T",
    "ACC": "T",
    "ACA": "T",
    "ACG": "T",
    "AAT": "N",
    "AAC": "N",
    "AAA": "K",
    "AAG": "K",
    "AGT": "S",
    "AGC": "S",
    "AGA": "R",
    "AGG": "R",
    "GTT": "V",
    "GTC": "V",
    "GTA": "V",
    "GTG": "V",
    "GCT": "A",
    "GCC": "A",
    "GCA": "A",
    "GCG": "A",
    "GAT": "D",
    "GAC": "D",
    "GAA": "E",
    "GAG": "E",
    "GGT": "G",
    "GGC": "G",
    "GGA": "G",
    "GGG": "G",
    "TAG": "",
    "TAA": "",
    "TGA": "",
}


def group_codons(seq: str) -> str:
    return " ".join(seq[i : i + 3] for i in range(0, len(seq), 3)).upper()


def codon_seq_to_amino_acid(codon_seq: str) -> str:
    return "".join(translation_table[codon] for codon in codon_seq.split())


class FastaDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        return_codon: bool = True,
        return_aminoacid: bool = False,
    ) -> None:
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid

        # Read the fasta file
        dna_sequenes = self.read_fasta_only_seq(file_path)
        # Preprocess the sequences into codons
        self.sequences = [
            group_codons(seq) for seq in dna_sequenes if len(seq) % 3 == 0
        ]

    def read_fasta_only_seq(self, fasta_file: str) -> List[str]:
        """Reads fasta file sequences without description tag."""
        text = Path(fasta_file).read_text()
        pattern = re.compile("^>", re.MULTILINE)
        non_parsed_seqs = re.split(pattern, text)[1:]
        lines = [
            line.replace("\n", "")
            for seq in non_parsed_seqs
            for line in seq.split("\n", 1)
        ]
        return lines[1::2]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # Get the idx'th codon sequence
        codon_sequence = self.sequences[idx]

        # The output data dictionary to be returned
        data = {}

        # Return the codon sequence
        if self.return_codon:
            data["codon"] = codon_sequence

        # Return the amino acid sequence
        if self.return_aminoacid:
            data["aminoacid"] = codon_seq_to_amino_acid(codon_sequence)

        return data


class HDF5Dataset(Dataset):
    """PyTorch Dataset backed by an HDF5 file which is read on-the-fly."""

    def __init__(
        self,
        file_path: str,
        return_codon: bool = True,
        return_aminoacid: bool = False,
    ) -> None:
        self.file_path = file_path
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid
        self.h5_file = None

        # Peek into the HDF5 file to determine the number of sequences
        with h5py.File(self.file_path, "r") as f:
            self.len = f["input_ids"].shape[0]

    def _init_h5(self) -> None:
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, "r")

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # Open the HDF5 file in the dataloader worker process
        self._init_h5()

        # Load the sequence from the HDF5 file
        dna_sequence: str = self.h5_file["sequences"][idx].decode("utf-8")
        codon_sequence = group_codons(dna_sequence)

        # The output data dictionary to be returned
        data = {}

        # Return the codon sequence
        if self.return_codon:
            data["codon"] = codon_sequence

        # Return the amino acid sequence
        if self.return_aminoacid:
            data["aminoacid"] = codon_seq_to_amino_acid(codon_sequence)

        return data


class GenSLMColatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Augment the underlying DataCollatorForLanguageModeling to handle
    multiple batch encoding inputs."""

    def __init__(
        self,
        return_codon: bool = True,
        return_aminoacid: bool = False,
        train_mode: bool = False,
        **kwargs,
    ):
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid
        self.train_mode = train_mode
        super().__init__(**kwargs)

    def tokenize(self, sequences: List[str]) -> BatchEncoding:
        return self.tokenizer(
            sequences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            return_special_tokens_mask=self.train_mode,
        )

    def torch_call_helper(self, batch: BatchEncoding) -> BatchEncoding:
        # We only need to mask tokens if we are training
        if not self.train_mode:
            return batch
        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_call(self, examples: List[Dict[str, BatchEncoding]]) -> Dict[str, Any]:
        if self.return_codon and self.return_aminoacid:
            # The first half of the batch is the codon sequences
            # and the second half is the amino acid sequences
            tokenized_seqs = self.tokenize(
                [e["codon"] for e in examples] + [e["aminoacid"] for e in examples]
            )
            return self.torch_call_helper(tokenized_seqs)
        elif self.return_codon:
            tokenized_seqs = self.tokenize([e["codon"] for e in examples])
            return self.torch_call_helper(tokenized_seqs)
        elif self.return_aminoacid:
            tokenized_seqs = self.tokenize([e["aminoacid"] for e in examples])
            return self.torch_call_helper(tokenized_seqs)
        assert False


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
        # NOTE: z.shape == (batch_size, embedding_size)
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


class ContrastiveProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        projection_size: int,
        temperature: float = 0.1,
    ) -> None:
        super().__init__()
        # The projection representions z are trained to become invariant to
        # many gene/protein specific features
        # TODO: Try a deeper/wider projection head
        # We use a different projection head for codons and amino acids
        # since, by default, the embeddings fall into different subspaces.
        self.codon_projection = nn.Linear(embedding_size, projection_size)
        self.aminoacid_projection = nn.Linear(embedding_size, projection_size)
        self.loss_fn = ContrastiveLoss(temperature=temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect the codon and aminoacid embeddings separately
        codon_embed = x[:, : x.shape[1] // 2]
        aminoacid_embed = x[:, x.shape[1] // 2 :]

        # Project the embeddings into a lower dimensional space
        z_codon = self.codon_projection(F.relu(codon_embed, inplace=True))
        z_aminoacid = self.codon_projection(F.relu(aminoacid_embed, inplace=True))

        # Concatenate the codon and aminoacid embeddings
        z = torch.cat([z_codon, z_aminoacid], dim=1)

        # Compute the contrastive loss following SimCLR
        return self.loss_fn(z)


# DEV NOTE: This is a hacky way to inject the contrastive loss into the Trainer
# (explicit coupling between Trainer and model.contrastive_head)
class GenSLMTrainer(Trainer):
    def __init__(self, compute_contrastive_loss: bool = False, **kwargs):
        self.compute_contrastive_loss = compute_contrastive_loss
        super().__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.compute_contrastive_loss:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
            print(outputs.keys())
            # The average over sequence length gives even weighting to each sequence position
            avg_embed = outputs["last_hidden_state"].mean(dim=1)
            # avg_embed: (batch_size, embedding_size)
            # Compute the contrastive loss following SimCLR
            loss += model.contrastive_head(avg_embed)
            return (loss, outputs) if return_outputs else loss

        return super().compute_loss(model, inputs, return_outputs=return_outputs)


@dataclass
class GenSLMTrainingConfig:
    compute_codon_loss: bool = True
    compute_aminoacid_loss: bool = True
    compute_contrastive_loss: bool = False
    temperature: float = 0.1
    max_length: int = 1024
    base_model: str = "facebook/esm2_t6_8M_UR50D"
    tokenizer_path: str = "tokenizer_esm_genslm"
    output_path: str = "mdh_natural_sequences_run_1"
    # data_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/mdh_sc23/fasta/mdh_natural_sequences.ffn"
    data_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/curriculum_datasets/curriculum_2/curriculum_2_train.h5"

    def __post_init__(self):
        if self.compute_contrastive_loss:
            self.compute_codon_loss = self.compute_aminoacid_loss = True
        if not (self.compute_codon_loss or self.compute_aminoacid_loss):
            raise ValueError(
                "At least one of return_codon or return_aminoacid must be True"
            )


def main():
    config = GenSLMTrainingConfig()

    # TODO: This would be a good option to try for more efficient packing: group_by_length
    args = TrainingArguments(
        output_dir=config.output_path,
        per_device_train_batch_size=64,
        # per_device_eval_batch_size=128,
        # evaluation_strategy="steps",
        # eval_steps=50,
        logging_steps=50,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=500,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False,  # This skips underlying logic in Trainer which modifies the data_collator
        dataloader_num_workers=4,  # Defaults to 0, may want to increase for faster data loading
    )

    # tokenizer = PreTrainedTokenizerFast.from_pretrained(config.tokenizer_path)
    tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)

    model = EsmForMaskedLM.from_pretrained(config.base_model)

    # TODO: During fine tuning or training from a checkpoint, this will restart the weights
    #       ONly do if the len(tokenizer) is not the same as the embedding layer
    # Inject new vocabulary (modifies model.config)
    model.resize_token_embeddings(len(tokenizer))
    # Make a new lm_head with uninitialized weights using the correct shape
    model.lm_head = EsmLMHead(model.config)

    # Inject a contrastive projection head if needed
    if config.compute_contrastive_loss:
        model.contrastive_head = ContrastiveProjectionHead(
            embedding_size=model.config.hidden_size,
            projection_size=model.config.hidden_size // 4,
        )

    # Select the dataset type based on the file extension
    dset_class = HDF5Dataset if config.data_path.endswith(".h5") else FastaDataset
    train_dataset = dset_class(
        file_path=config.data_path,
        return_codon=config.compute_codon_loss,
        return_aminoacid=config.compute_aminoacid_loss,
    )

    data_collator = GenSLMColatorForLanguageModeling(
        return_codon=config.compute_codon_loss,
        return_aminoacid=config.compute_aminoacid_loss,
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    trainer = GenSLMTrainer(
        compute_contrastive_loss=config.compute_contrastive_loss,
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()


if __name__ == "__main__":
    main()
