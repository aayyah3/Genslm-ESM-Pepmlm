import h5py
import torch
import torch.nn as nn
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
    return "".join(translation_table[codon] for codon in codon_seq)


class HDF5Dataset(Dataset):
    """PyTorch Dataset backed by an HDF5 file which is read on-the-fly."""

    def __init__(
        self,
        file_path: str,
        tokenizer: PreTrainedTokenizerFast,
        max_length: int = 1024,
        return_codon: bool = True,
        return_aminoacid: bool = False,
    ) -> None:
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid
        self.h5_file = None

        if not (return_codon or return_aminoacid):
            raise ValueError(
                "At least one of return_codon or return_aminoacid must be True"
            )

        # Peek into the HDF5 file to determine the number of sequences
        with h5py.File(self.file_path, "r") as f:
            self.len = f["input_ids"].shape[0]

    def _init_h5(self) -> None:
        """Open the HDF5 file in the dataloader worker process."""
        if self.h5_file is None:
            self.h5_file = h5py.File(self.file_path, "r")

    def tokenize(self, sequence: str) -> BatchEncoding:
        return self.tokenizer(
            sequence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )

    def __len__(self) -> int:
        return self.len

    def __getitem__(self, idx: int) -> Dict[str, BatchEncoding]:
        self._init_h5()

        # Load the sequence from the HDF5 file
        dna_sequence: str = self.h5_file["sequences"][idx].decode("utf-8")
        codon_sequence = group_codons(dna_sequence)

        # The output data dictionary to be returned
        data = {}

        # Tokenize the codon sequence
        if self.return_codon:
            data["codon"] = self.tokenize(codon_sequence)

        # Tokenize the amino acid sequence
        if self.return_aminoacid:
            amino_acid_sequence = codon_seq_to_amino_acid(codon_sequence)
            data["aminoacid"] = self.tokenize(amino_acid_sequence)

        return data


class GenSLMColatorForLanguageModeling(DataCollatorForLanguageModeling):
    """Augment the underlying DataCollatorForLanguageModeling to handle
    multiple batch encoding inputs."""

    def __init__(
        self, return_codon: bool = True, return_aminoacid: bool = False, *args, **kwargs
    ):
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid
        super().__init__(*args, **kwargs)

    def torch_call(self, examples: List[Dict[str, BatchEncoding]]) -> Dict[str, Any]:
        batch = {}
        if self.return_codon:
            batch["codon"] = super().torch_call([e["codon"] for e in examples])
        if self.return_aminoacid:
            batch["aminoacid"] = super().torch_call([e["aminoacid"] for e in examples])
        return batch


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def forward(self, z_codon, z_aminoacid):
        # NOTE: z_codon.shape == (batch_size, embedding_size)
        # NOTE: z_aminoacid.shape == (batch_size, embedding_size)

        # TODO: This implementation could have bugs

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(z_codon, z_aminoacid, dim=-1)
        # InfoNCE loss
        cos_sim = cos_sim / self.temperature
        nll = -torch.logsumexp(cos_sim, dim=-1)
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
        self.projection = nn.Linear(embedding_size, projection_size)
        self.loss_fn = ContrastiveLoss(temperature=temperature)

    def compute_loss(self, z_codon, z_aminoacid):
        # Project the embeddings into a lower dimensional space
        codon_proj = self(z_codon)
        aminoacid_proj = self(z_aminoacid)

        # Compute the contrastive loss following SimCLR
        return self.loss_fn(codon_proj, aminoacid_proj)

    def forward(self, x):
        return self.projection(F.relu(x, inplace=True))


# DEV NOTE: This is a hacky way to inject the contrastive loss into the Trainer
# (explicit coupling between Trainer and model.contrastive_head)
class GenSLMTrainer(Trainer):
    def __init__(
        self,
        compute_codon_loss: bool = True,
        compute_aminoacid_loss: bool = False,
        compute_contrastive_loss: bool = False,
        **kwargs
    ):
        self.compute_codon_loss = compute_codon_loss
        self.compute_aminoacid_loss = compute_aminoacid_loss
        self.compute_contrastive_loss = compute_contrastive_loss

        if self.compute_contrastive_loss:
            if not (self.compute_codon_loss and self.compute_aminoacid_loss):
                raise ValueError(
                    "Contrastive loss requires both codon and aminoacid loss"
                )

        super().__init__(**kwargs)

    def compute_llm_loss(self, model, inputs):
        if self.compute_contrastive_loss:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        else:
            loss = super().compute_loss(model, inputs, return_outputs=False)
            outputs = None

        return loss, outputs

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss = 0.0
        if self.compute_codon_loss:
            codon_loss, codon_outputs = self.compute_llm_loss(model, inputs["codon"])
            loss += codon_loss
        if self.compute_aminoacid_loss:
            aminoacid_loss, aminoacid_outputs = self.compute_llm_loss(
                model, inputs["aminoacid"]
            )
            loss += aminoacid_loss
        if self.compute_contrastive_loss:
            # The average over sequence length gives even weighting to each sequence position
            codon_avg_embed = codon_outputs["last_hidden_state"].mean(dim=1)
            aminoacid_avg_embed = aminoacid_outputs["last_hidden_state"].mean(dim=1)

            # Compute the contrastive loss following SimCLR
            contrastive_loss = model.contrastive_head.compute_loss(
                codon_avg_embed, aminoacid_avg_embed
            )
            loss += contrastive_loss
        return loss


@dataclass
class GenSLMTrainingArguments:
    compute_codon_loss: bool = True
    compute_aminoacid_loss: bool = False
    compute_contrastive_loss: bool = False
    temperature: float = 0.1
    esm_base_model: str = "facebook/esm2_t6_8M_UR50D"
    tokenizer_path: str = "tokenizer_path"


# TODO: Make script to output tokenizer files
# TODO: In the collator, just pack the codon and aminacid sequences
#       into the same batch. This is more efficient than the current
#       implementation which creates a batch for each sequence type.
#       This will require some changes to the Trainer class in how
#       the loss is computed. This will also allow us to use the
#       more standed SimCLR loss which assumes the positive and negative
#       pairs are in the same batch. Note, the effective batch size
#       will be twice as large.


def main():
    args = TrainingArguments(
        output_dir="output_path",
        remove_unused_columns=False,  # This skips underlying logic in Trainer which modifies the data_collator
        dataloader_num_workers=0,  # Defaults to 0, may want to increase for faster data loading
    )

    tokenizer = PreTrainedTokenizerFast.from_pretrained("tokenizer_path")

    model = EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")

    # Inject new vocabulary (modifies model.config)
    model.resize_token_embeddings(len(tokenizer))
    # Make a new lm_head with uninitialized weights using the correct shape
    model.lm_head = EsmLMHead(model.config)

    # Inject a contrastive projection head if needed
    if compute_contrastive_loss:
        model.contrastive_head = ContrastiveProjectionHead(
            embedding_size=model.config.hidden_size,
            projection_size=model.config.hidden_size // 4,
        )

    train_dataset = HDF5Dataset(
        "data_path.h5", tokenizer, return_codon=True, return_aminoacid=False
    )

    data_collator = GenSLMColatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=0.15
    )

    trainer = Trainer(
        model=model, args=args, data_collator=data_collator, train_dataset=train_dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
