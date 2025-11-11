"""Script to upload EsmCForContrastiveMaskedLM model to Hugging Face Hub.

python scripts/upload_to_hf.py \
    --model_path /path/to/model \
    --repo_id username/model-name \
    [--tokenizer_path /path/to/tokenizer] \
    [--token HF_TOKEN] \
    [--private] \
    [--commit_message "Custom message"] \
    [--create_pr]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub import login
from transformers import EsmTokenizer

from genslm_esm.modeling_esmc import EsmCForContrastiveMaskedLM


def upload_model_to_hf(
    model_path: str,
    repo_id: str,
    tokenizer_path: str | None = None,
    token: str | None = None,
    private: bool = False,
    commit_message: str = 'Upload model',
    create_pr: bool = False,
) -> None:
    """Upload EsmCForContrastiveMaskedLM model to Hugging Face Hub.

    Parameters
    ----------
    model_path : str
        Path to the local model directory (should contain config.json,
        model.pt or pytorch_model.bin, and tokenizer files).
    repo_id : str
        The repository ID on Hugging Face Hub (e.g., "username/model-name").
    tokenizer_path : str, optional
        Path to the tokenizer directory. If not provided, will use
        config.tokenizer_name_or_path from the loaded model config.
    token : str, optional
        Hugging Face API token. If not provided, will try to use cached token
        or prompt for login.
    private : bool, default=False
        Whether to create a private repository.
    commit_message : str, default="Upload model"
        Commit message for the upload.
    create_pr : bool, default=False
        Whether to create a pull request instead of pushing directly.
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise ValueError(f'Model path does not exist: {model_path_obj}')

    # Authenticate with Hugging Face
    if token:
        login(token=token)
    else:
        # Try to use cached token, or prompt for login
        try:
            login()
        except Exception as e:
            print(
                f'Failed to authenticate. Please set HF_TOKEN environment '
                f"variable or run 'huggingface-cli login'. Error: {e}",
            )
            raise

    # Load the model
    print(f'Loading model from {model_path_obj}...')
    model = EsmCForContrastiveMaskedLM.from_pretrained(str(model_path_obj))
    print('Model loaded successfully.')

    # Load and set the tokenizer from the specified path or config
    if tokenizer_path:
        tokenizer_path_to_use = tokenizer_path
        print(
            f'Loading tokenizer from specified path: {tokenizer_path_to_use}',
        )
    elif (
        hasattr(model.config, 'tokenizer_name_or_path')
        and model.config.tokenizer_name_or_path
    ):
        tokenizer_path_to_use = model.config.tokenizer_name_or_path
        print(
            f'Loading tokenizer from config.tokenizer_name_or_path: '
            f'{tokenizer_path_to_use}',
        )
    else:
        # Fall back to model directory
        tokenizer_path_to_use = str(model_path_obj)
        print(
            f'Loading tokenizer from model directory: {tokenizer_path_to_use}',
        )

    tokenizer = EsmTokenizer.from_pretrained(tokenizer_path_to_use)
    model.transformer.tokenizer = tokenizer
    print('Tokenizer loaded and set on model.')

    # Create repository on Hugging Face Hub if it doesn't exist
    api = HfApi()
    try:
        api.repo_info(repo_id=repo_id, repo_type='model')
        print(f'Repository {repo_id} already exists.')
    except Exception:
        print(f'Creating repository {repo_id}...')
        api.create_repo(
            repo_id=repo_id,
            repo_type='model',
            private=private,
            exist_ok=True,
        )
        print(f'Repository {repo_id} created.')

    # Upload the model using push_to_hub
    # This will also upload the tokenizer since save_pretrained saves it
    print(f'Uploading model to {repo_id}...')
    model.push_to_hub(  # type: ignore[attr-defined]
        repo_id=repo_id,
        commit_message=commit_message,
        create_pr=create_pr,
        private=private,
    )

    # Explicitly upload the tokenizer to ensure it's included
    print(f'Uploading tokenizer to {repo_id}...')
    tokenizer.push_to_hub(  # type: ignore[attr-defined]
        repo_id=repo_id,
        commit_message=commit_message,
        create_pr=create_pr,
    )
    print(f'Model and tokenizer uploaded successfully to {repo_id}!')


def main() -> None:
    """Entry point for uploading the model to Hugging Face Hub."""
    parser = argparse.ArgumentParser(
        description='Upload the model to Hugging Face Hub',
    )
    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to the local model directory',
    )
    parser.add_argument(
        '--repo_id',
        type=str,
        required=True,
        help="Hugging Face repository ID (e.g., 'username/model-name')",
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help=(
            'Path to the tokenizer directory. If not provided, will use '
            'config.tokenizer_name_or_path from the loaded model config.'
        ),
    )
    parser.add_argument(
        '--token',
        type=str,
        default=None,
        help='Hugging Face API token (or set HF_TOKEN env var)',
    )
    parser.add_argument(
        '--private',
        action='store_true',
        help='Create a private repository',
    )
    parser.add_argument(
        '--commit_message',
        type=str,
        default='Upload EsmCForContrastiveMaskedLM model',
        help='Commit message for the upload',
    )
    parser.add_argument(
        '--create_pr',
        action='store_true',
        help='Create a pull request instead of pushing directly',
    )

    args = parser.parse_args()

    # Use token from environment if not provided
    token = args.token or os.getenv('HF_TOKEN')

    upload_model_to_hf(
        model_path=args.model_path,
        repo_id=args.repo_id,
        tokenizer_path=args.tokenizer_path,
        token=token,
        private=args.private,
        commit_message=args.commit_message,
        create_pr=args.create_pr,
    )


if __name__ == '__main__':
    main()
