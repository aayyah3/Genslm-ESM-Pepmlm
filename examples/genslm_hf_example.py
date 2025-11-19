"""A quick example of using the GensLM-ESMC model for inference."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer

from genslm_esm.data import FastaDataset
from genslm_esm.data import GenslmEsmcDataCollator


def main() -> None:
    """Run the example."""
    # Define the model id to use
    model_id = 'genslm-test/genslm-esmc-600M-contrastive'

    # Load the model and tokenizer from Hugging Face
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Set the model to evaluation mode
    model.eval()

    # Print the model architecture
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

    # The dataset splits the sequences into codons
    dataset = FastaDataset(
        sequences=sequences,
        return_codon=True,
        return_aminoacid=True,
    )

    # Create the collator
    collator = GenslmEsmcDataCollator(
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
        items = batch.to(device)
        print(items)
        outputs = model(**items)
        print(outputs.loss)


if __name__ == '__main__':
    main()
