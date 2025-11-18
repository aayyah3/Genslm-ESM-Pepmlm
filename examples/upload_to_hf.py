"""Upload the model to the Hugging Face hub."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import create_repo
from huggingface_hub import HfApi
from transformers import EsmTokenizer

import genslm_esm
from genslm_esm.configuration import GenslmEsmcConfig
from genslm_esm.modeling import GenslmEsmcModel


def main() -> None:
    """Upload the model to the Hugging Face hub."""
    # Parse the command line arguments
    parser = argparse.ArgumentParser(
        description='Upload the model to the Hugging Face hub.',
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='The path to the model checkpoint.',
    )
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        help='The model id to use for the Hugging Face hub.',
    )
    parser.add_argument(
        '--save-dir',
        type=str,
        required=True,
        help='The directory to save the model to.',
    )
    args = parser.parse_args()

    # Load the config from the model path
    loaded_config = GenslmEsmcConfig.from_pretrained(args.model_path)

    # Update the relevant parameters only
    config = GenslmEsmcConfig(
        d_model=loaded_config.d_model,
        n_heads=loaded_config.n_heads,
        n_layers=loaded_config.n_layers,
        contrastive_temperature=loaded_config.contrastive_temperature,
        use_flash_attn=loaded_config.use_flash_attn,
    )

    # Set the Hugging Face metadata for the model
    config.set_hf_metadata(args.model_id)

    # Load the model and tokenizer from the checkpoint
    model = GenslmEsmcModel.from_pretrained(args.model_path, config=config)
    tokenizer = EsmTokenizer.from_pretrained(args.model_path)

    # Save locally to the save directory
    model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)

    # Copy custom code into repo directory
    genslm_esm_dir = Path(genslm_esm.__file__).relative_to(Path.cwd()).parent
    for file in ['configuration.py', 'modeling.py']:
        shutil.copy(genslm_esm_dir / file, Path(args.save_dir) / file)

    # # Push everything to the Hugging Face hub
    # model.push_to_hub(args.model_id, private=True)
    # tokenizer.push_to_hub(args.model_id, private=True)

    # Ensure the repo exists (idempotent; will not error if it already exists)
    create_repo(
        repo_id=args.model_id,
        private=True,
        exist_ok=True,
        repo_type='model',
    )

    # Upload the model folder to the Hugging Face hub
    api = HfApi()
    api.upload_folder(
        folder_path=args.save_dir,
        repo_id=args.model_id,
        repo_type='model',
    )


if __name__ == '__main__':
    main()
