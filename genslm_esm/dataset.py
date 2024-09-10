from __future__ import annotations
import json
import random
import re
import shutil
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Union, Any, Optional, Tuple
import torch
import h5py
from abc import ABC, abstractmethod
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
        file_path: PathLike | None = None,
        sequences: List[Sequence] | None = None,
        return_codon: bool = True,
        return_aminoacid: bool = False,
    ) -> None:
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid

        if file_path is None and sequences is None:
            raise ValueError("Either file_path or sequences must be provided.")

        # Read the fasta file
        if sequences is None:
            dna_sequenes = self.read_fasta_only_seq(file_path)
        else:
            dna_sequenes = sequences

        # Preprocess the sequences into codons
        # TODO: We could also use an <unk> token (this would be better)
        self.sequences = [
            group_codons(seq) for seq in dna_sequenes
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

    def __init__(self, file_path: PathLike | None = None, sequences: List[Sequence] | None = None) -> None:
        if file_path is None and sequences is None:
            raise ValueError("Either file_path or sequences must be provided.")

        if sequences is None:
            self.sequences = self.read_fasta_only_seq(file_path)
        else:
            self.sequences = sequences

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {"aminoacid": self.sequences[idx]}


class CurriculumHDF5DatasetBuilder:
    """Encapsulate logic for building a sequence homology curriculum dataset."""

    def make_clusters_mmseqs(self) -> None:
        info_string = "mmseqs easy-cluster [input_fasta_file] [output_prefix] [temp_folder] --min-seq-id 0.5 --alignment-mode 3 --max-seqs 200 -s 7 -c 0.8 --cov-mode 0"
        print(
            f"Run command:\n`{info_string}`\nTo generate the cluster files, see mmseqs2 documentation for more info"
        )

    def _read_cluster_file(self, fp: Path) -> Dict[str, List[str]]:
        """
        Reads a *_cluster.tsv file from mmseqs and reads it into a dictionary of {rep_seq: [member_seq, member_seq,...,member_seq]}
        """
        clusters = defaultdict(list)

        with open(fp, "r") as f:
            text = f.readlines()
            for line in text:
                if len(line) == 0:
                    continue
                rep, member = line.strip().split("\t")
                clusters[rep].append(member)

        return clusters

    def make_id_to_index(
        self, curriculum_h5_fp: Path, cluster_file: Path, outfile: Path
    ) -> None:
        # Read in the output *_cluster.tsv from mmseqs `cluster_rep\tcluster_member\n`, takes ~2min
        clusters = self._read_cluster_file(cluster_file)
        # Load the h5 file for the seq md5's
        curriculum = h5py.File(curriculum_h5_fp, "r")

        # Takes a very long time (~2 hr)
        ids = curriculum["id"]
        id_to_index = {
            seq_id.decode("utf-8"): idx
            for idx, seq_id in tqdm(enumerate(ids), total=len(ids))
        }

        # Takes ~2 min
        md5_idx_to_h5_idx = [
            np.array([id_to_index[member] for member in cluster_members])
            for cluster_members in clusters.values()
        ]

        np.save(outfile, md5_idx_to_h5_idx)

        curriculum.close()


class HDF5SequenceSampler(ABC):
    """Abstract base class for sampling sequences from a dataset."""

    @abstractmethod
    def num_sequences(self, split: str) -> int:
        """Return the number of sequences in the split."""
        pass

    @abstractmethod
    def sample(self, idx: int, split: str) -> int:
        """Sample a sequence index of the split."""
        pass


class StandardSampler(HDF5SequenceSampler):
    """Standard sampler that samples sequences uniformly at random."""

    def __init__(
        self,
        file_path: PathLike,
        train_ratio: float = 0.9,
        seed: int = 42,
    ) -> None:
        """The StandardSampler samples sequences from a dataset uniformly at random.

        Parameters
        ----------
        file_path : PathLike
            The path to the HDF5 file containing the sequences (contains the key: 'sequence').
        train_ratio : float, optional
            The proportion of data to use for training, by default 0.9
        seed : int, optional
            The random seed to reproduce the split, by default 42

        Raises
        ------
        ValueError
            If the `train_ratio` is not between 0 and 1.
        """
        # Check that the split ratios are valid
        if not (0 < train_ratio < 1):
            raise ValueError("Invalid split ratios, must sum to 1.0")

        # Peek into the HDF5 file to determine the number of sequences
        with h5py.File(file_path, "r") as f:
            self.len = f["sequence"].shape[0]

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Generate train and eval indices
        inds = np.random.permutation(self.len)
        train_end = int(self.len * train_ratio)
        self.train_inds = inds[:train_end]
        self.eval_inds = inds[train_end:]

    def num_sequences(self, split: str) -> int:
        """Return the number of sequences in the split."""
        return len(self.train_inds if split == "train" else self.eval_inds)

    def sample(self, idx: int, split: str) -> int:
        """Sample a random sequence index."""
        inds = self.train_inds if split == "train" else self.eval_inds
        return inds[idx]


class SequenceHomologySampler(HDF5SequenceSampler):
    """
    This dataset requires mmseqs easycluster files to be made.

    For a sequence identity threshold of 0.5, this command was used:
    ```
    mmseqs easy-cluster [input_fasta_file] [output_prefix] [temp_folder] \
        --min-seq-id 0.5 \
        --alignment-mode 3 \
        --max-seqs 200 \
        -s 7 \
        -c 0.8 \
        --cov-mode 0
    ```
    """

    def __init__(
        self,
        file_path: PathLike,
        train_ratio: float = 0.9,
        seed: int = 42,
        num_eval_samples: int | None = None,
    ) -> None:
        """The SequenceHomologySampler samples sequences from a dataset such that
        each cluster is represented in both the training and evaluation sets.

        Parameters
        ----------
        file_path : PathLike
            The path to the mmseqs cluster mapping file.
        train_ratio : float, optional
            The proportion of data to use for training, by default 0.9
        seed : int, optional
            The random seed to reproduce the split, by default 42
        num_eval_samples : int, optional
            The number of evaluation samples to use each epoch. Since the evaluation
            set can be very large, if this option is not None, then we cycle through
            the evaluation set, evaluating `num_eval_samples` per epoch, by default None.

        Raises
        ------
        ValueError
            If the `train_ratio` is not between 0 and 1.
        """
        # Check that the split ratios are valid
        if not (0 < train_ratio < 1):
            raise ValueError("Invalid split ratios, must sum to 1.0")

        # Load cluster mapping file (generated by CurriculumHDF5DatasetBuilder)
        # Stores a mapping from md5 cluster index to the HDF5
        # indices of the sequences within that cluster (md5_idx_to_h5_idx)
        clusters: np.ndarray = np.load(file_path, allow_pickle=True)

        # Remove any clusters with only 2 sequences
        clusters = clusters[[len(cluster) > 2 for cluster in clusters]]

        # Set the random seed for reproducibility
        np.random.seed(seed)

        # Store a random sample of the sequence indices within each cluster for training and eval
        # This effectively splits md5_idx_to_h5_idx into train and eval sets such that
        # sequences from each cluster are represented in the both training and evaluation sets
        self.train_inds: dict[int, np.ndarray] = {}
        self.eval_inds: dict[int, np.ndarray] = {}

        # From each cluster, randomly sample train_ratio of the sequences for training
        # and eval_ratio of the sequences for evaluation
        for cluster_idx, cluster in enumerate(clusters):
            # Generate a random permutation of the cluster indices
            sample_inds = np.random.permutation(len(cluster))
            # Split the indices into train and eval sets
            train_end = int(len(cluster) * train_ratio)
            # Store the train and eval indices for this cluster
            self.train_inds[cluster_idx] = cluster[sample_inds[:train_end]]
            self.eval_inds[cluster_idx] = cluster[sample_inds[train_end:]]

        # Store the number of evaluation samples to use each epoch
        # as well as the current eval cluster to sample
        self.current_eval_idx = 0
        self.num_eval_samples = num_eval_samples

    def num_sequences(self, split: str) -> int:
        """Return the number of clusters."""
        # The number of clusters is the same for train and eval
        if split == "train" or self.num_eval_samples is None:
            return len(self.train_inds)
        return self.num_eval_samples

    def sample(self, idx: int, split: str) -> int:
        """Sample a random sequence index from the `idx` cluster."""
        if split == "train":
            return np.random.choice(self.train_inds[idx])

        # Sample from the eval set according to the number of eval samples
        if self.num_eval_samples is not None:
            # If we have already sampled all of the evaluation clusters,
            # start over and sample again
            self.current_eval_idx %= len(self.eval_inds)
            idx = self.current_eval_idx
            self.current_eval_idx += 1

        # Sample from the eval set at the selected index
        return np.random.choice(self.eval_inds[idx])


class HDF5Dataset(Dataset):
    def __init__(
        self,
        file_path: PathLike,
        hdf5_sampler: HDF5SequenceSampler,
        split: str = "train",
        return_codon: bool = True,
        return_aminoacid: bool = False,
    ) -> None:
        """A PyTorch Dataset for loading sequences from an HDF5 file according to any sampling strategy.

        Parameters
        ----------
        file_path : Path
            The path to the HDF5 file containing the sequences (contains the key: 'sequence').
        hdf5_sampler : HDF5SequenceSampler
            The sampler to use for sampling sequences from the HDF5 file.
        split : str, optional
            The split to sample from, by default "train".
        return_codon : bool, optional
            Whether to return the codon sequence, by default True.
        return_aminoacid : bool, optional
            Whether to return the amino acid sequence, by default False.
        """
        self.file_path = file_path
        self.hdf5_sampler = hdf5_sampler
        self.split = split
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid

    @property
    def h5_data(self) -> h5py.File:
        """Lazy load the h5 file in the dataloader worker process."""
        if not hasattr(self, "_h5_data"):
            self._h5_data = h5py.File(self.file_path, "r")
        return self._h5_data

    def __len__(self) -> int:
        return self.hdf5_sampler.num_sequences(self.split)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        # Randomly sample one of the sequences within the `idx` sequence cluster
        sample_idx = self.hdf5_sampler.sample(idx, self.split)

        # Load the sequence from the HDF5 file
        dna_sequence: str = self.h5_data["sequence"][sample_idx].decode("utf-8")
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
        if self.return_codon:
            # Set the low parameter to 33 to sample random noise from the
            # codon vocabulary and not the amino acid vocabulary
            codon_batch = self.torch_call_helper([e["codon"] for e in examples], low=33)
            # We first need to realign the codon labels onto the output label range [0, 69)
            # We do this by subtracting 28 from the codon labels since the codon labels start with
            # the mask token at '<mask>': 32, and we also need to account for the special tokens
            # '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, which are included in the codon vocabulary
            # Note: labels are only present during training, not inference.
            if "labels" in codon_batch:
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
            # Then we need to add an extra pad token to the amino acid input_ids,
            # labels, and attention_mask to account for the stop codon
            batch_size, seq_len = codon_batch["input_ids"].shape
            pad_size = seq_len - amino_batch["input_ids"].shape[1]
            pad = torch.ones((batch_size, pad_size), dtype=torch.long)
            amino_batch["input_ids"] = torch.cat([amino_batch["input_ids"], pad], dim=1)
            amino_batch["labels"] = torch.cat(
                [amino_batch["labels"], pad * -100], dim=1
            )
            amino_batch["attention_mask"] = torch.cat(
                [amino_batch["attention_mask"], pad * 0], dim=1
            )

            # We have to put the amino acid and codon sequences into separate
            # fields in the BatchEncoding object because, otherwise the order
            # gets shuffled in the hugging face distributed sampler.
            return BatchEncoding(
                {
                    # The amino acids are passed through the standard variables
                    "input_ids": amino_batch["input_ids"],
                    "attention_mask": amino_batch["attention_mask"],
                    "labels": amino_batch["labels"],
                    "codon_input_ids": codon_batch["input_ids"],
                    "codon_attention_mask": codon_batch["attention_mask"],
                    "codon_labels": codon_batch["labels"],
                }
            )

        elif self.return_codon:
            return codon_batch
        elif self.return_aminoacid:
            return amino_batch

        assert False

    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Optional[Any] = None,
        low: int = 0,  # Custom parameter
        high: Optional[int] = None,  # Custom parameter
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
