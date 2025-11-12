"""Run the inference on the original model."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer

from genslm_esm.dataset import FastaDataset
from genslm_esm.dataset import GenSLMColatorForLanguageModeling
from genslm_esm.modeling_esmc import EsmCForContrastiveMaskedLM


def run_inference() -> None:
    """Run the inference using the original model used for training."""
    model_path = '/nfs/lambda_stor_01/homes/abrace/projects/genslm/src/genslm-tutorial-05-2025/model/checkpoint-203847'

    # Load the model from the checkpoint
    model = EsmCForContrastiveMaskedLM.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    print('Original model:')
    print(model)

    # Get the device to run the model on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Move the model to the device
    model = model.to(device)

    # Convert the model to bfloat16 if not on CPU
    if device.type != 'cpu':
        model = model.to(torch.bfloat16)

    print(f'Model is on device: {next(model.parameters()).device}')
    print(f'Model dtype: {next(model.parameters()).dtype}')

    # Test sequences
    sequences = [
        'ATGAAGGTACTACCACAAGAAACTGTAAGAATTGGA',
        'ATGGACAAAACACATATTCGACTATCTGTTGACAATCCATTTGCAAAACTA',
    ]

    tokenizer = EsmTokenizer.from_pretrained(model_path)

    # The dataset splits the sequences into codons
    dataset = FastaDataset(
        sequences=sequences,
        return_codon=True,
        return_aminoacid=True,
    )

    # Create the collator
    collator = GenSLMColatorForLanguageModeling(
        return_codon=True,
        return_aminoacid=True,
        tokenizer=tokenizer,
    )

    # Create the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # Iterate over the dataloader
    for batch in dataloader:
        batch = batch.to(device)
        print(batch)
        outputs = model(**batch)
        print(outputs.loss)
