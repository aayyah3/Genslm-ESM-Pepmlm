"""Generate embeddings for a given model and dataloader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EsmTokenizer

from genslm_esm.dataset import FastaAminoAcidDataset
from genslm_esm.dataset import FastaDataset
from genslm_esm.dataset import GenSLMColatorForLanguageModeling
from genslm_esm.modeling_esm_v3 import EsmForContrastiveMaskedLM


def generate_embeddings(
    model: torch.nn.Module,
    dataloader: DataLoader,
) -> npt.ArrayLike:
    """Generate embeddings for a given model and dataloader."""
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(model.device)  # noqa: PLW2901
            outputs = model(**batch, output_hidden_states=True)
            last_hidden_states = outputs.hidden_states[-1]
            seq_lengths = batch.attention_mask.sum(axis=1)
            for seq_len, elem in zip(seq_lengths, last_hidden_states):
                # Compute averaged embedding
                embedding = elem[1 : seq_len - 1, :].mean(dim=0).cpu().numpy()
                embeddings.append(embedding)

    return np.array(embeddings)


def embedding_inference(
    tokenizer_path: Path,
    model_path: Path,
    fasta_path: Path,
    return_codon: bool,
    return_aminoacid: bool,
    batch_size: int,
    fasta_contains_aminoacid: bool = False,
) -> npt.ArrayLike:
    """Run embedding inference."""
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)
    # import pdb; pdb.set_trace()
    model = EsmForContrastiveMaskedLM.from_pretrained(model_path)
    # model = EsmForMaskedLM.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device).eval()

    if fasta_contains_aminoacid:
        dataset = FastaAminoAcidDataset(file_path=fasta_path)
    else:
        dataset = FastaDataset(
            file_path=fasta_path,
            return_codon=return_codon,
            return_aminoacid=return_aminoacid,
        )

    data_collator = GenSLMColatorForLanguageModeling(
        return_codon=return_codon,
        return_aminoacid=return_aminoacid,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )

    embeddings = generate_embeddings(model, dataloader)

    return embeddings
