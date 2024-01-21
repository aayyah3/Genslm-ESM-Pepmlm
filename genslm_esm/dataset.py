import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Any, Optional, Tuple
import torch
import h5py
from torch.utils.data import Dataset
from transformers import BatchEncoding, DataCollatorForLanguageModeling

PathLike = Union[str, Path]

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


@dataclass
class Sequence:
    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""

    def translate(self) -> "Sequence":
        amino_acid_seq = codon_seq_to_amino_acid(group_codons(self.sequence))
        return Sequence(sequence=amino_acid_seq.replace(" ", ""), tag=self.tag)


def read_fasta(fasta_file: PathLike) -> List[Sequence]:
    """Reads fasta file sequences and description tags into dataclass."""
    text = Path(fasta_file).read_text()
    pattern = re.compile("^>", re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace("\n", "") for seq in non_parsed_seqs for line in seq.split("\n", 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag) for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Union[Sequence, List[Sequence]], fasta_file: PathLike, mode: str = "w"
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f">{seq.tag}\n{seq.sequence}\n")


def random_split_fasta(
    input_fasta: PathLike, output_dir: PathLike, split: float = 0.8, seed: int = 0
) -> None:
    """Randomly split a fasta file into train and validation fasta file."""
    # Read the input file
    sequences = read_fasta(input_fasta)

    # Shuffle the sequences
    random.seed(seed)
    random.shuffle(sequences)

    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Write the train and validation fasta files
    split_idx = int(len(sequences) * split)
    write_fasta(sequences[:split_idx], output_dir / "train.fasta")
    write_fasta(sequences[split_idx:], output_dir / "valid.fasta")

    # Copy the original fasta file to the output directory for reference
    shutil.copy(input_fasta, output_dir)

    # Log JSON metadata on the split
    metadata = {
        "input_fasta": str(Path(input_fasta).resolve()),
        "output_dir": str(output_dir.resolve()),
        "split": split,
        "seed": seed,
        "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


class FastaDataset(Dataset):
    def __init__(
        self,
        file_path: PathLike,
        return_codon: bool = True,
        return_aminoacid: bool = False,
    ) -> None:
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid

        # Read the fasta file
        dna_sequenes = self.read_fasta_only_seq(file_path)
        # Preprocess the sequences into codons
        # TODO: We could also use an <unk> token (this would be better)
        self.sequences = [
            group_codons(seq) for seq in dna_sequenes if len(seq) % 3 == 0
        ]

    def read_fasta_only_seq(self, fasta_file: PathLike) -> List[str]:
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


class FastaAminoAcidDataset(FastaDataset):
    """Assumes the fasta file contains amino acid sequences."""

    def __init__(self, file_path: PathLike) -> None:
        self.sequences = self.read_fasta_only_seq(file_path)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"aminoacid": self.sequences[idx]}


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

    def torch_call_helper(
        self, sequences: List[str], low: int = 0, high: Optional[int] = None
    ) -> BatchEncoding:
        # First, tokenize the batch
        batch = self.tokenize(sequences)

        # We only need to mask tokens if we are training
        if not self.train_mode:
            return batch

        if self.mlm:
            # If special token mask has been preprocessed, pop it from the dict.
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"],
                special_tokens_mask=batch.pop("special_tokens_mask", None),
                low=low,
                high=high,
            )
        else:
            # TODO: This region of the code is not used for our BERT models
            # please test this if you want to use it.
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch

    def torch_call(self, examples: List[Dict[str, str]]) -> BatchEncoding:
        # if self.return_codon and self.return_aminoacid:
        #     # The first half of the batch is the codon sequences
        #     # and the second half is the amino acid sequences
        #     return self.torch_call_helper(
        #         [e["codon"] for e in examples] + [e["aminoacid"] for e in examples]
        #     )
        if self.return_codon:
            # Set the low parameter to 33 to sample random noise from the
            # codon vocabulary and not the amino acid vocabulary
            codon_batch = self.torch_call_helper([e["codon"] for e in examples], low=33)
            # We first need to realign the codon labels onto the output label range [0, 69)
            # We do this by subtracting 28 from the codon labels since the codon labels start with
            # the mask token at '<mask>': 32, and we also need to account for the special tokens
            # '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, which are included in the codon vocabulary
            mask = codon_batch["labels"] > 32
            codon_batch["labels"][mask] -= 28

        if self.return_aminoacid:
            # Set the high parameter to 25 to sample random noise from the
            # amino acid vocabulary and not the codon vocabulary (there are a
            # tokens in the amino acid vocabulary that we want to avoid sampling from)
            # 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, '<null_1>': 31, '<mask>': 32,
            # Note: we also avoid sampling the special tokens '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3,
            amino_batch = self.torch_call_helper(
                [e["aminoacid"] for e in examples], low=4, high=25
            )

        if self.return_codon and self.return_aminoacid:
            # Then we need to add an extra pad token to the amino acid input ids
            # and labels to account for the stop codon
            amino_batch["input_ids"] = torch.cat(
                [amino_batch["input_ids"], torch.ones_like(amino_batch["input_ids"])],
                dim=1,
            )
            amino_batch["labels"] = torch.cat(
                [amino_batch["labels"], torch.ones_like(amino_batch["labels"])],
                dim=1,
            )

            # Now we need stack the codon and amino acid batches
            input_ids = torch.cat(
                [codon_batch["input_ids"], amino_batch["input_ids"]], dim=0
            )

            # We also need to stack the attention masks
            attention_mask = torch.cat(
                [codon_batch["attention_mask"], amino_batch["attention_mask"]], dim=0
            )

            # We also need to stack the labels
            labels = torch.cat([codon_batch["labels"], amino_batch["labels"]], dim=0)

            # Finally, we need to stack the token type ids
            token_type_ids = torch.cat(
                [codon_batch["token_type_ids"], amino_batch["token_type_ids"]], dim=0
            )

            # Return the stacked batch as a BatchEncoding
            return BatchEncoding(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                    "token_type_ids": token_type_ids,
                }
            )

            pass
        elif self.return_codon:
            return codon_batch
        elif self.return_aminoacid:
            return amino_batch

        assert False

    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
        high: Optional[int] = None,  # Custom parameter
        low: int = 0,  # Custom parameter
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        import torch

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        # Custom edit: By default, the random words are sampled from the entire
        # vocabulary. We want to sample from the same vocabulary as the input
        # sequences (i.e. the codon sequences or aminoacid sequences, but not both).
        # This is important because the amino acid and codon sequences are not in
        # the same vocabulary. This is done by passing the vocab_size argument to torch.randint
        random_words = torch.randint(
            low=low,
            high=len(self.tokenizer) if high is None else high,
            size=labels.shape,
            dtype=torch.long,
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels
