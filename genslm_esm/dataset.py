import re
import h5py
from pathlib import Path
from typing import Any, Dict, List

from torch.utils.data import Dataset
from transformers import BatchEncoding, DataCollatorForLanguageModeling

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
    return " ".join(
        translation_table.get(codon, "<unk>") for codon in codon_seq.split()
    )


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
        dna_sequence: str = self.h5_file["sequence"][idx].decode("utf-8")
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
            return_special_tokens_mask=self.train_mode and self.mlm,
        )

    def torch_call_helper(self, batch: BatchEncoding) -> BatchEncoding:
        # First, tokenize the batch
        batch = self.tokenize(batch)

        # We only need to mask tokens if we are training
        if not self.train_mode:
            return batch

        if self.mlm:
            # If special token mask has been preprocessed, pop it from the dict.
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"],
                special_tokens_mask=batch.pop("special_tokens_mask", None),
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
            return self.torch_call_helper(
                [e["codon"] for e in examples] + [e["aminoacid"] for e in examples]
            )
        elif self.return_codon:
            return self.torch_call_helper([e["codon"] for e in examples])
        elif self.return_aminoacid:
            return self.torch_call_helper([e["aminoacid"] for e in examples])
        assert False
