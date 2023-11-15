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


@app.command()
def gather_fastas(
    fasta_dir: Path = typer.Option(
        ...,
        "--fasta_dir",
        "-f",
        help="The directory containing fasta files to gather.",
    ),
    glob_pattern: str = typer.Option(
        "*.fasta",
        "--glob",
        "-g",
        help="A glob pattern specifiying several fasta files.",
    ),
    output_path: Path = typer.Option(
        ...,
        "--output_path",
        "-o",
        help="The fasta file containing the gathered sequences.",
    ),
) -> None:
    """Utility to gather many fasta files into a single large one."""
    from tqdm import tqdm

    from genslm_esm.dataset import read_fasta, write_fasta

    sequences = []
    for fasta_file in tqdm(fasta_dir.glob(glob_pattern)):
        sequences.extend(read_fasta(fasta_file))
    write_fasta(sequences, output_path)


@app.command()
def generate_embeddings(
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
        help="The numpy (.npy) file containing the sequence embeddings."
        "The shape of the array is (num_sequences, embedding_size).",
    ),
    tokenizer_path: Path = typer.Option(
        ...,
        "--tokenizer_path",
        "-t",
        help="The path to the tokenizer to use (or ESM Huggingface model name).",
    ),
    model_path: Path = typer.Option(
        ...,
        "--model_path",
        "-m",
        help="The path to the model to use (or ESM Huggingface model name).",
    ),
    return_aminoacid: bool = typer.Option(
        False,
        "--return_aminoacid",
        "-a",
        help="Whether to return the amino acid embeddings.",
    ),
    return_codon: bool = typer.Option(
        False,
        "--return_codon",
        "-c",
        help="Whether to return the codon embeddings.",
    ),
    fasta_contains_aminoacid: bool = typer.Option(
        False,
        "--fasta_contains_aminoacid",
        help="Whether the fasta file contains amino acid sequences.",
    ),
    batch_size: int = typer.Option(
        512,
        "--batch_size",
        "-b",
        help="The batch size to use when computing embeddings.",
    ),
) -> None:
    """Utility to generate sequence embeddings given an input fasta file."""
    import numpy as np

    from genslm_esm.embedding import embedding_inference

    if not return_codon and not return_aminoacid:
        raise ValueError(
            "At least one of return_codon and return_aminoacid must be True."
        )
    if return_codon and return_aminoacid:
        raise ValueError("Only one of return_codon and return_aminoacid can be True.")

    print(f"Returning {'codon' if return_codon else 'amino acid'} embeddings.")

    embeddings = embedding_inference(
        tokenizer_path=tokenizer_path,
        model_path=model_path,
        fasta_path=fasta_path,
        return_codon=return_codon,
        return_aminoacid=return_aminoacid,
        batch_size=batch_size,
        fasta_contains_aminoacid=fasta_contains_aminoacid,
    )

    # Save the embeddings to disk
    np.save(output_path, embeddings)


@app.command()
def best_checkpoint(
    train_output_dir: Path = typer.Option(
        ...,
        "--train_output_dir",
        "-o",
        help="The directory containing the training run checkpoints.",
    ),
) -> None:
    """Utility to report the best checkpoint from a training run (measured by smallest eval_loss)."""
    from genslm_esm.utils import best_checkpoint

    # Get the best checkpoint
    best_loss, best_ckpt = best_checkpoint(train_output_dir)

    # Print the best checkpoint
    print(f"Best checkpoint at {best_loss} eval loss: {best_ckpt}")


@app.command()
def collate_checkpoints(
    run_dir: Path = typer.Option(
        ...,
        "--run_dir",
        "-r",
        help="The directory containing many training output directories.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output_dir",
        "-o",
        help="The directory to write the gathered checkpoints to.",
    ),
) -> None:
    """Utility to collect the best checkpoints from several training output
    directories into a single directory (measured by smallest eval_loss)."""
    import shutil
    from genslm_esm.utils import best_checkpoint

    # Get the best checkpoint from each training run
    best_ckpts = []
    for train_output_dir in filter(lambda x: x.is_dir(), run_dir.glob("*")):
        best_loss, best_ckpt = best_checkpoint(train_output_dir)
        best_ckpts.append(best_ckpt)

        # Print the best checkpoint
        print(f"Best checkpoint at {best_loss} eval loss: {best_ckpt}")

    # Collate the runs and there best checkpoints into a single directory
    output_dir.mkdir(exist_ok=True)
    for best_ckpt in best_ckpts:
        # Make an output directory for each best checkpoint
        original_train_output_dir = best_ckpt.parent
        new_train_output_dir = output_dir / original_train_output_dir.name

        # Copy any extra files or folders from the original training output directory
        # shutil.copytree(
        #     original_train_output_dir,
        #     new_train_output_dir,
        #     ignore=shutil.ignore_patterns("checkpoint-*"),
        # )

        # Copy the best checkpoint
        # shutil.copytree(best_ckpt, new_train_output_dir / best_ckpt.name)


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
