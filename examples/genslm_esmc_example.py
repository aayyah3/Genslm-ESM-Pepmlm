"""A quick example of using the GensLM-ESMC model for inference."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from transformers import EsmTokenizer

from genslm_esm.data import FastaDataset
from genslm_esm.data import GenslmEsmcDataCollator
from genslm_esm.data import group_codons
from genslm_esm.modeling import EsmCForContrastiveMaskedLM


def main() -> None:
    """Run the example."""
    model_path = '/nfs/lambda_stor_01/homes/abrace/projects/genslm/src/genslm-tutorial-05-2025/model/checkpoint-203847'  # noqa: E501

    # Load the model from the checkpoint
    model = EsmCForContrastiveMaskedLM.from_pretrained(model_path)

    # Set the model to evaluation mode
    model.eval()

    print('Reloaded model:')
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

    tokenizer = EsmTokenizer.from_pretrained(model_path)
    print('Tokenizer:')
    print(tokenizer)

    # Test sequences
    sequences = [
        'ATGAAGGTACTACCACAAGAAACTGTAAGAATTGGA',
        'ATGGACAAAACACATATTCGACTATCTGTTGACAATCCATTTGCAAAACTA',
    ]

    split_seqs = [group_codons(x) for x in sequences]

    # Tokenize the sequences to test the tokenizer
    tokenized = tokenizer(
        split_seqs,
        return_tensors='pt',
        truncation=True,
        padding='max_length',
        max_length=2048,
        return_special_tokens_mask=False,
    )
    print(tokenized)

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
