import h5py
from typing import Dict
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast, BatchEncoding

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
