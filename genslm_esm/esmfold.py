"""
This code was adapted from the EsmFold tutorial notebook, which can be found here:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/protein_folding.ipynb#scrollTo=1c9de19e
"""

import torch
from pathlib import Path
from tqdm import tqdm
from typing import List
from argparse import ArgumentParser
from torch.utils.data import DataLoader, Dataset
from transformers import EsmForProteinFolding
from genslm_esm.dataset import read_fasta


class SequenceDataset(Dataset):
    def __init__(self, sequences: List[str]) -> None:
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> str:
        return self.sequences[idx]


def main(fasta_file: str, output_dir: str, batch_size: int) -> None:
    # Load the model
    model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", low_cpu_mem_usage=True
    )

    # Performance optimization
    model = model.cuda()
    model.esm = model.esm.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    # Uncomment this line if your GPU memory is 16GB or less, or if you're folding longer (over 600 or so) sequences
    # model.trunk.set_chunk_size(64)

    dataset = SequenceDataset([seq.sequence for seq in read_fasta(fasta_file)])

    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Setup output directory
    output_dir = Path(output_dir)
    Path(output_dir).mkdir(exist_ok=True)

    sequence_idx = 0

    for batch in tqdm(dataloader):
        # Run inference
        print(batch)
        pdbs = model.infer_pdbs(batch)

        # Write the PDBs to disk
        for pdb in pdbs:
            with open(output_dir / f"{sequence_idx}.pdb", "w") as f:
                f.write(pdb)
            sequence_idx += 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-f", "--fasta_file", required=True, help="Amino acid sequences in fasta format"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Folding output directory"
    )
    parser.add_argument(
        "-b", "--batch_size", default=1, help="How many sequences to fold at once"
    )
    args = parser.parse_args()
    main(args.fasta_file, args.output_dir, args.batch_size)
