from pathlib import Path

import typer

app = typer.Typer()


@app.command()
def random_split_fasta(
    fasta_path: Path = typer.Option(
        ...,
        "--fasta_path",
        "-f",
        help="The fasta file containing nucleotide gene sequences.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output_path",
        "-o",
        help="The directory to write the split fasta files to (train.fasta and valid.fasta).",
    ),
    split: float = typer.Option(
        0.8,  # default
        "--split",
        help="The proportion of sequences to put in the training set. "
        "The remaining sequences will be put in the validation set.",
    ),
    seed: int = typer.Option(
        0,  # default
        "--split",
        help="The random seed to use when splitting the fasta file.",
    ),
) -> None:
    """Utility to convert a codon fasta file to amino acid fasta file."""
    from genslm_esm.dataset import random_split_fasta

    random_split_fasta(fasta_path, output_path, split, seed)


@app.command()
def write_aminoacid_fasta(
    fasta_path: Path = typer.Option(
        ...,
        "--fasta_path",
        "-f",
        help="The fasta file containing nucleotide gene sequences.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output_path",
        "-o",
        help="The fasta file containing the translated amino acid sequences.",
    ),
) -> None:
    """Utility to convert a codon fasta file to amino acid fasta file."""
    from genslm_esm.dataset import read_fasta, write_fasta

    sequences = read_fasta(fasta_path)
    amino_acid_sequences = [seq.translate() for seq in sequences]
    write_fasta(amino_acid_sequences, output_path)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
