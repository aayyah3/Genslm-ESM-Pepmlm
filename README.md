# genslm-esm

This repository contains the code for the GenSLM-ESM model.

GenSLM-ESM is a multi-modal representation learning model that learns to represent both codon and amino acid sequences in a shared embedding space using the genetic code as an inductive bias. It supports the following tasks:
- Multi-modal representation learning task
- Reverse translation task
- Forward translation task
- Standard codon language modeling task
- Standard amino acid language modeling task

The model is based on the ESM-300M and ESMC-600M models.

## Installation

```bash
pip install git+ssh://git@github.com/ramanathanlab/genslm-esm.git@main
```

The models are hosted on the Hugging Face hub. To install the models, you can use the `from_pretrained` method. For example:
```python
from transformers import AutoModel

model = AutoModel.from_pretrained("genslm-test/genslm-esmc-300M-contrastive")
```

### Supported model IDs

Models using the ESM-300M as a base:
- `genslm-test/genslm-esmc-300M-aminoacid`
- `genslm-test/genslm-esmc-300M-codon`
- `genslm-test/genslm-esmc-300M-joint`
- `genslm-test/genslm-esmc-300M-contrastive`

Models using ESMC-600M as a base:
- `genslm-test/genslm-esmc-600M-aminoacid`
- `genslm-test/genslm-esmc-600M-codon`
- `genslm-test/genslm-esmc-600M-joint`
- `genslm-test/genslm-esmc-600M-contrastive`

## Usage

To use the model, you can either use the `GenslmEsmcModel` class or the `AutoModel` class. For example:
```python
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer

from genslm_esm.data import FastaDataset
from genslm_esm.data import GenslmEsmcDataCollator

# Define which model id to use
model_id = "genslm-test/genslm-esmc-300M-contrastive"

model = AutoModel.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Set the model to evaluation mode
model.eval()

print('Model details:')
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
```
### GenSLM-ESM inputs

As GenSLM-ESM is a multi-modal model, it accepts both codon and amino acid sequences as input, we've
provided the `FastaDataset` class and `GenslmEsmcDataCollator` class to help you prepare the data for the model, as shown in the example above.

The `GenslmEsmcModel.forward()` method accepts the following input parameters:

#### Required Inputs

- **`codon_input_ids`** or **`aminoacid_input_ids`** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *at least one required*):
  - Tokenized input sequences. At least one of `codon_input_ids` or `aminoacid_input_ids` must be provided.
  - `codon_input_ids`: Tokenized codon sequences
  - `aminoacid_input_ids`: Tokenized amino acid sequences

- **`codon_attention_mask`** or **`aminoacid_attention_mask`** (`torch.Tensor` of shape `(batch_size, sequence_length)`, *required if corresponding input_ids provided*):
  - Attention masks indicating which tokens should be attended to (1) and which should be ignored (0).
  - Must be provided if the corresponding `input_ids` is provided.

#### Optional Inputs

- **`codon_labels`** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
  - Labels for computing the masked language modeling loss for codons.
  - Indices should be in `[-100, 0, ..., 69]`. Tokens with indices set to `-100` are ignored (masked), and the loss is only computed for tokens with labels in `[0, ..., 69]`.
  - The vocabulary size is 69 (64 codon tokens + 5 special tokens).

- **`aminoacid_labels`** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
  - Labels for computing the masked language modeling loss for amino acids.
  - Indices should be in `[-100, 0, ..., 64]`. Tokens with indices set to `-100` are ignored (masked), and the loss is only computed for tokens with labels in `[0, ..., 64]`.

- **`decode_aminoacid_head`** (`bool`, *optional*, defaults to `False`):
  - Whether to use the amino acid head for prediction, regardless of config settings.
  - Used for forward translation tasks (predicting amino acids from codons).

- **`decode_codon_head`** (`bool`, *optional*, defaults to `False`):
  - Whether to use the codon head for prediction, regardless of config settings.
  - Used for reverse translation tasks (predicting codons from amino acids).

- **`compute_contrastive_loss`** (`bool`, *optional*, defaults to `False`):
  - Whether to compute the SimCLR-style contrastive loss.
  - Requires both `codon_input_ids` and `aminoacid_input_ids` to be provided.
  - Used for multi-modal representation learning tasks.

#### Supported Tasks

The GenSLM-ESM model supports five different tasks based on the combination of inputs provided:

1. **Multi-modal Representation Learning Task**
   - **Inputs**: Both `codon_input_ids` and `aminoacid_input_ids` with `compute_contrastive_loss=True`
   - **Purpose**: Learns to represent both codon and amino acid sequences in a shared embedding space
   - **Outputs**: Computes contrastive loss in addition to MLM losses. The total `loss` is the sum of averaged MLM losses and contrastive loss.

2. **Reverse Translation Task**
   - **Inputs**: `aminoacid_input_ids` with `decode_codon_head=True`
   - **Purpose**: Predicts codon sequences from amino acid sequences
   - **Outputs**: `codon_mlm_loss` represents the accuracy of reverse translation. `aminoacid_mlm_loss` is also computed for completeness.

3. **Forward Translation Task**
   - **Inputs**: `codon_input_ids` with `decode_aminoacid_head=True`
   - **Purpose**: Predicts amino acid sequences from codon sequences
   - **Outputs**: `aminoacid_mlm_loss` represents the accuracy of forward translation. `codon_mlm_loss` is also computed for completeness.

4. **Standard Codon Language Modeling Task**
   - **Inputs**: Only `codon_input_ids` and `codon_attention_mask`
   - **Purpose**: Standard masked language modeling for codon sequences
   - **Outputs**: `codon_mlm_loss` represents the accuracy of codon language modeling. Both `loss` and `mlm_loss` are set to the codon MLM loss.

5. **Standard Amino Acid Language Modeling Task**
   - **Inputs**: Only `aminoacid_input_ids` and `aminoacid_attention_mask`
   - **Purpose**: Standard masked language modeling for amino acid sequences
   - **Outputs**: `aminoacid_mlm_loss` represents the accuracy of amino acid language modeling. Both `loss` and `mlm_loss` are set to the amino acid MLM loss.

#### GenSLM-ESM outputs
The output of the model is a `GenslmEsmcModelOutput` dataclass with the following attributes:

#### Loss Attributes

- **`loss`** (`torch.FloatTensor` of shape `(1,)`, *optional*):
  The total loss for the model. This is computed based on the provided inputs:
  - If only `codon_labels` is provided: returns the masked language modeling (MLM) loss for the codon head
  - If only `aminoacid_labels` is provided: returns the MLM loss for the amino acid head
  - If both `codon_labels` and `aminoacid_labels` are provided: returns the average of the codon and amino acid MLM losses
  - If `compute_contrastive_loss=True`: returns the sum of the averaged MLM losses and the contrastive loss
  - Returns `None` if no labels are provided. All losses are averaged over the batch dimension.

- **`contrastive_loss`** (`torch.FloatTensor` of shape `(1,)`, *optional*):
  The SimCLR-style contrastive loss, returned when both `codon_labels` and `aminoacid_labels` are provided and `compute_contrastive_loss=True`. Averaged over the batch dimension.

- **`mlm_loss`** (`torch.FloatTensor` of shape `(1,)`, *optional*):
  The masked language modeling (MLM) loss:
  - If both `codon_labels` and `aminoacid_labels` are provided: returns the average of codon and amino acid MLM losses
  - If only `codon_labels` is provided: returns the codon MLM loss
  - If only `aminoacid_labels` is provided: returns the amino acid MLM loss
  - Returns `None` if no labels are provided. Averaged over the batch dimension.

- **`codon_mlm_loss`** (`torch.FloatTensor` of shape `(1,)`, *optional*):
  The masked language modeling loss specifically for the codon head. Returned when `codon_labels` is provided. Averaged over the batch dimension.

- **`aminoacid_mlm_loss`** (`torch.FloatTensor` of shape `(1,)`, *optional*):
  The masked language modeling loss specifically for the amino acid head. Returned when `aminoacid_labels` is provided. Averaged over the batch dimension.

#### Prediction Logits

- **`codon_logits`** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 69)`, *optional*):
  Prediction scores for the codon head (scores for each vocabulary token before SoftMax). The vocabulary size is 69, which accounts for 64 codon tokens and 5 special tokens. Returned when `codon_labels` is provided.

- **`aminoacid_logits`** (`torch.FloatTensor` of shape `(batch_size, sequence_length, 64)`, *optional*):
  Prediction scores for the amino acid head (scores for each vocabulary token before SoftMax). The vocabulary size is 64, where a subset is used for the 20 standard amino acids and special tokens. The extra vocabulary space is an artifact of the base ESMC model architecture. Returned when `aminoacid_labels` is provided.

#### Hidden States

- **`hidden_states`** (`torch.FloatTensor` of shape `(num_layers, batch_size, sequence_length, hidden_size)`, *optional*):
  The hidden states of the model transformer at each layer. The final hidden state is stored in the last entry, i.e., `hidden_states[-1]` has shape `(batch_size, sequence_length, hidden_size)`.

- **`attentions`** (`tuple[torch.FloatTensor, ...]`, *optional*):
  Attention weights after the attention softmax (not currently supported).


## Contributing

For development, it is recommended to use a virtual environment. The following
commands will create a virtual environment, install the package in editable
mode, and install the pre-commit hooks.
```bash
python -m venv venv
source venv/bin/activate
pip install -U pip setuptools wheel
pip install -e '.[dev,docs]'
pre-commit install
```
To test the code, run the following command:
```bash
pre-commit run --all-files
tox -e py310
```

## Questions
If you have any questions, please open an issue on the GitHub repository and we'll get back to you as soon as possible.
