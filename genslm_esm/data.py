"""Data utilities for GenSLM-ESMC."""

from __future__ import annotations

import json
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers.data.data_collator import DataCollatorForLanguageModeling
from transformers.tokenization_utils_base import BatchEncoding

# Stop codons map to empty strings ""
translation_table = {
    'TTT': 'F',
    'TTC': 'F',
    'TTA': 'L',
    'TTG': 'L',
    'TCT': 'S',
    'TCC': 'S',
    'TCA': 'S',
    'TCG': 'S',
    'TAT': 'Y',
    'TAC': 'Y',
    'TGT': 'C',
    'TGC': 'C',
    'TGG': 'W',
    'CTT': 'L',
    'CTC': 'L',
    'CTA': 'L',
    'CTG': 'L',
    'CCT': 'P',
    'CCC': 'P',
    'CCA': 'P',
    'CCG': 'P',
    'CAT': 'H',
    'CAC': 'H',
    'CAA': 'Q',
    'CAG': 'Q',
    'CGT': 'R',
    'CGC': 'R',
    'CGA': 'R',
    'CGG': 'R',
    'ATT': 'I',
    'ATC': 'I',
    'ATA': 'I',
    'ATG': 'M',
    'ACT': 'T',
    'ACC': 'T',
    'ACA': 'T',
    'ACG': 'T',
    'AAT': 'N',
    'AAC': 'N',
    'AAA': 'K',
    'AAG': 'K',
    'AGT': 'S',
    'AGC': 'S',
    'AGA': 'R',
    'AGG': 'R',
    'GTT': 'V',
    'GTC': 'V',
    'GTA': 'V',
    'GTG': 'V',
    'GCT': 'A',
    'GCC': 'A',
    'GCA': 'A',
    'GCG': 'A',
    'GAT': 'D',
    'GAC': 'D',
    'GAA': 'E',
    'GAG': 'E',
    'GGT': 'G',
    'GGC': 'G',
    'GGA': 'G',
    'GGG': 'G',
    'TAG': '',
    'TAA': '',
    'TGA': '',
}

# The valid codons are the keys of the translation table
valid_codons = set(translation_table.keys())


def group_codons(seq: str) -> str:
    """Group codons into three-mers and replace invalid codons with '<unk>'."""
    seq = seq.upper()
    three_mers = (seq[i : i + 3] for i in range(0, len(seq), 3))
    return ' '.join(x if x in valid_codons else '<unk>' for x in three_mers)


def codon_seq_to_amino_acid(codon_seq: str) -> str:
    """Translate codons to amino acids.

    Replace invalid codons with '<unk>'.
    """
    return ' '.join(
        translation_table.get(codon, '<unk>') for codon in codon_seq.split()
    )


@dataclass
class Sequence:
    """A biological sequence and its description tag."""

    sequence: str
    """Biological sequence (Nucleotide sequence)."""
    tag: str
    """Sequence description tag."""

    def translate(self) -> Sequence:
        """Translate the sequence to amino acids."""
        amino_acid_seq = codon_seq_to_amino_acid(group_codons(self.sequence))
        return Sequence(sequence=amino_acid_seq.replace(' ', ''), tag=self.tag)


def read_fasta(fasta_file: str | Path) -> list[Sequence]:
    """Read a fasta file sequences and description tags."""
    text = Path(fasta_file).read_text()
    pattern = re.compile('^>', re.MULTILINE)
    non_parsed_seqs = re.split(pattern, text)[1:]
    lines = [
        line.replace('\n', '')
        for seq in non_parsed_seqs
        for line in seq.split('\n', 1)
    ]

    return [
        Sequence(sequence=seq, tag=tag)
        for seq, tag in zip(lines[1::2], lines[::2])
    ]


def write_fasta(
    sequences: Sequence | list[Sequence],
    fasta_file: str | Path,
    mode: str = 'w',
) -> None:
    """Write or append sequences to a fasta file."""
    seqs = [sequences] if isinstance(sequences, Sequence) else sequences
    with open(fasta_file, mode) as f:
        for seq in seqs:
            f.write(f'>{seq.tag}\n{seq.sequence}\n')


def random_split_fasta(
    input_fasta: str | Path,
    output_dir: str | Path,
    split: float = 0.8,
    seed: int = 0,
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
    write_fasta(sequences[:split_idx], output_dir / 'train.fasta')
    write_fasta(sequences[split_idx:], output_dir / 'valid.fasta')

    # Copy the original fasta file to the output directory for reference
    shutil.copy(input_fasta, output_dir)

    # Log JSON metadata on the split
    metadata = {
        'input_fasta': str(Path(input_fasta).resolve()),
        'output_dir': str(output_dir.resolve()),
        'split': split,
        'seed': seed,
        'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


class FastaDataset(Dataset):
    """A dataset of biological sequences from a fasta file."""

    def __init__(
        self,
        file_path: str | Path | None = None,
        sequences: list[str] | None = None,
        return_codon: bool = True,
        return_aminoacid: bool = False,
        contains_nucleotide: bool = True,
    ) -> None:
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid
        self.contains_nucleotide = contains_nucleotide

        # Check that either return_codon or return_aminoacid is True
        if not return_codon and not return_aminoacid:
            raise ValueError(
                'Either return_codon or return_aminoacid must be True.',
            )

        # If the sequences do not contain nucleotides, then we cannot
        # return codons
        if not contains_nucleotide and return_codon:
            raise ValueError(
                'Cannot return codons if the sequences do not contain '
                'nucleotides.',
            )

        # Check that either file_path or sequences is provided
        if file_path is None and sequences is None:
            raise ValueError('Either file_path or sequences must be provided.')

        # Read the fasta file
        if sequences is None:
            assert file_path is not None
            sequences = [seq.sequence for seq in read_fasta(file_path)]

        # If the input is nucleotide sequences, then preprocess them
        # into codons
        if contains_nucleotide:
            self.sequences = [group_codons(seq) for seq in sequences]
        else:
            self.sequences = sequences

    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Get the idx'th sequence from the dataset."""
        # Get the idx'th sequence
        sequence = self.sequences[idx]

        # The output data dictionary to be returned
        data = {}

        # Handle the case where the sequences contain nucleotides
        if self.contains_nucleotide:
            if self.return_codon:
                data['codon'] = sequence

            if self.return_aminoacid:
                data['aminoacid'] = codon_seq_to_amino_acid(sequence)

        # Handle the case where the sequences do not contain nucleotides
        # Return the amino acid sequence if requested
        elif self.return_aminoacid:
            data['aminoacid'] = sequence

        # Return the data dictionary
        return data


class GenslmEsmcDataCollator(DataCollatorForLanguageModeling):
    """Collate sequences for language modeling.

    Augment the underlying DataCollatorForLanguageModeling to handle
    multiple batch encoding inputs.
    """

    def __init__(
        self,
        return_codon: bool = True,
        return_aminoacid: bool = False,
        train_mode: bool = False,
        max_length: int = 2048,
        padding: str = 'longest',
        **kwargs: Any,
    ) -> None:
        """Collate sequences for language modeling.

        Augment the underlying DataCollatorForLanguageModeling to handle
        multiple batch encoding inputs.

        Parameters
        ----------
        return_codon : bool, optional
            Whether to return codon sequences, by default True.
        return_aminoacid : bool, optional
            Whether to return amino acid sequences, by default False.
        train_mode : bool, optional
            Whether we are in training mode (i.e. whether to mask tokens),
            by default False.
        max_length : int, optional
            Maximum sequence length, by default 2048.
        padding : str, optional
            Padding strategy ('longest' or 'max_length'). If 'longest', pad to
            the longest sequence in the batch. If 'max_length', pad to the
            maximum length specified by max_length. By default 'longest'.
        """
        self.return_codon = return_codon
        self.return_aminoacid = return_aminoacid
        self.train_mode = train_mode
        self.max_length = max_length
        self.padding = padding
        super().__init__(**kwargs)

    def tokenize(self, sequences: list[str]) -> BatchEncoding:
        """Tokenize a list of sequences."""
        return self.tokenizer(
            sequences,
            return_tensors='pt',
            truncation=True,
            padding=self.padding,
            max_length=self.max_length,
            return_special_tokens_mask=self.train_mode and self.mlm,
        )

    def torch_call_helper(
        self,
        sequences: list[str],
        low: int = 0,
        high: int | None = None,
    ) -> BatchEncoding:
        """Tokenize a list of sequences and mask tokens if we are training."""
        # First, tokenize the batch
        batch = self.tokenize(sequences)

        # We only need to mask tokens if we are training
        if not self.train_mode:
            # Need to manually set the labels so that torch_call can adjust
            # the label range to [0, 69) for the codon vocabulary, and so that
            # losses are computed for the codon and amino acid heads.
            batch['labels'] = batch['input_ids']
            return batch

        if self.mlm:
            # If special token mask has been preprocessed, pop it from the
            # dict.
            batch['input_ids'], batch['labels'] = self.torch_mask_tokens(
                batch['input_ids'],
                special_tokens_mask=batch.pop('special_tokens_mask', None),
                low=low,
                high=high,
            )
        else:
            # TODO: This region of the code is not used for our BERT models
            # please test this if you want to use it.
            labels = batch['input_ids'].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch['labels'] = labels
        return batch

    def torch_call(self, examples: list[dict[str, str]]) -> BatchEncoding:
        """Collate a list of sequences for language modeling."""
        if self.return_codon:
            # Set the low parameter to 33 to sample random noise from the
            # codon vocabulary and not the amino acid vocabulary
            codon_batch = self.torch_call_helper(
                [e['codon'] for e in examples],
                low=33,
            )
            # We first need to realign the codon labels onto the output label
            # range [0, 69). We do this by subtracting 28 from the codon
            # labels since the codon labels start with the mask token at
            # '<mask>': 32, and we also need to account for the special tokens
            # '<cls>': 0, '<pad>': 1, '<eos>': 2, '<unk>': 3, which are
            # included in the codon vocabulary.
            # Note: labels are only present during training, not inference.
            if 'labels' in codon_batch:
                mask = codon_batch['labels'] > 32  # noqa: PLR2004
                codon_batch['labels'][mask] -= 28

        if self.return_aminoacid:
            # Set the high parameter to 25 to sample random noise from the
            # amino acid vocabulary and not the codon vocabulary (there are a
            # tokens in the amino acid vocabulary that we want to avoid
            # sampling from) 'B': 25, 'U': 26, 'Z': 27, 'O': 28, '.': 29,
            # '-': 30, '<null_1>': 31, '<mask>': 32,
            # Note: we also avoid sampling the special tokens '<cls>': 0,
            # '<pad>': 1, '<eos>': 2, '<unk>': 3,
            amino_batch = self.torch_call_helper(
                [e['aminoacid'] for e in examples],
                low=4,
                high=24,
            )

        if self.return_codon and self.return_aminoacid:
            # Then we need to add an extra pad token to the amino acid
            # input_ids, labels, and attention_mask to account for the stop
            # codon
            batch_size, seq_len = codon_batch['input_ids'].shape
            pad_size = seq_len - amino_batch['input_ids'].shape[1]
            pad = torch.ones((batch_size, pad_size), dtype=torch.long)
            amino_batch['input_ids'] = torch.cat(
                [amino_batch['input_ids'], pad],
                dim=1,
            )
            amino_batch['labels'] = torch.cat(
                [amino_batch['labels'], pad * -100],
                dim=1,
            )
            amino_batch['attention_mask'] = torch.cat(
                [amino_batch['attention_mask'], pad * 0],
                dim=1,
            )

            # We have to put the amino acid and codon sequences into separate
            # fields in the BatchEncoding object because, otherwise the order
            # gets shuffled in the hugging face distributed sampler.
            return BatchEncoding(
                {
                    'aminoacid_input_ids': amino_batch['input_ids'],
                    'aminoacid_attention_mask': amino_batch['attention_mask'],
                    'aminoacid_labels': amino_batch['labels'],
                    'codon_input_ids': codon_batch['input_ids'],
                    'codon_attention_mask': codon_batch['attention_mask'],
                    'codon_labels': codon_batch['labels'],
                },
            )

        elif self.return_codon:
            return BatchEncoding(
                {
                    'codon_input_ids': codon_batch['input_ids'],
                    'codon_attention_mask': codon_batch['attention_mask'],
                    'codon_labels': codon_batch['labels'],
                },
            )
        elif self.return_aminoacid:
            return BatchEncoding(
                {
                    'aminoacid_input_ids': amino_batch['input_ids'],
                    'aminoacid_attention_mask': amino_batch['attention_mask'],
                    'aminoacid_labels': amino_batch['labels'],
                },
            )

        raise ValueError(
            'Either return_codon or return_aminoacid must be True.',
        )

    def torch_mask_tokens(
        self,
        inputs: Any,
        special_tokens_mask: Any | None = None,
        low: int = 0,  # Custom parameter
        high: int | None = None,  # Custom parameter
    ) -> tuple[Any, Any]:
        """Prepare masked tokens inputs/labels for masked language modeling.

        Prepare masked tokens inputs/labels for masked language modeling:
        80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training
        # (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val,
                    already_has_special_tokens=True,
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(
                special_tokens_mask,
                dtype=torch.bool,
            )
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with
        # tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool()
            & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token,
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        # Custom edit: By default, the random words are sampled from the entire
        # vocabulary. We want to sample from the same vocabulary as the input
        # sequences (i.e. the codon sequences or aminoacid sequences, but not
        # both). This is important because the amino acid and codon sequences
        # are not in the same vocabulary. This is done by passing the
        # vocab_size argument to torch.randint
        random_words = torch.randint(
            low=low,
            high=len(self.tokenizer) if high is None else high,
            size=labels.shape,
            dtype=torch.long,
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input
        # tokens unchanged
        return inputs, labels
