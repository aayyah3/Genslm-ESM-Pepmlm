from __future__ import annotations

import os
from argparse import ArgumentParser
from dataclasses import asdict, dataclass, field
from pathlib import Path
import wandb
import yaml
import json
import transformers
from transformers import EsmTokenizer, Trainer, EsmConfig
from transformers.trainer_utils import get_last_checkpoint

from genslm_esm.dataset import (
    FastaDataset,
    HDF5Dataset,
    SequenceHomologySampler,
    GenSLMColatorForLanguageModeling,
)
from genslm_esm.modeling_esm_v3 import EsmForContrastiveMaskedLM


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """TrainingArguments for configuring the Hugging Face Trainer.
    Here we provide some sensible defaults for the arguments for our use case.
    """

    output_dir: str = field(
        default="test_run",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written."
        },
    )
    per_device_train_batch_size: int = field(
        default=64,
        metadata={"help": "Batch size per GPU/TPU core/CPU for training."},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."},
    )
    num_train_epochs: float = field(
        default=20,
        metadata={"help": "Total number of training epochs to perform."},
    )
    learning_rate: float = field(
        default=4e-4,
        metadata={"help": "The initial learning rate for Adam."},
    )
    warmup_steps: int = field(
        default=1_000,
        metadata={"help": "Linear warmup over `warmup_steps`."},
    )
    lr_scheduler_type: str = field(
        default="cosine",
        metadata={"help": "The scheduler type to use."},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "The weight decay to apply."},
    )
    eval_steps: int = field(
        default=500,
        metadata={
            "help": "Number of steps between evaluations. If `eval_steps` "
            "is modified, update `logging_steps` and `save_steps` to the same value."
        },
    )
    save_total_limit: int = field(
        default=1,
        metadata={"help": "Total number of checkpoints to save."},
    )
    save_strategy: str = field(
        default="steps",
        metadata={"help": "Strategy for saving checkpoints."},
    )
    evaluation_strategy: str = field(
        default="steps",
        metadata={"help": "Strategy for evaluating."},
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": "Whether to load the best model at the end of training. "
            "When `save_total_limit` is set to 1, will save the best model as "
            "well as the last model if the last model is worse (eval_loss) than the best model."
        },
    )
    fp16: bool = field(
        default=True,
        metadata={"help": "Whether to use 16-bit (mixed) precision training."},
    )
    dataloader_num_workers: int = field(
        default=4,
        metadata={"help": "Number of subprocesses to use for data loading."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "This skips underlying logic in Trainer which modifies the data_collator (do not change)."
        },
    )


@dataclass
class TrainingConfig:
    training_args: TrainingArguments = field(
        default_factory=TrainingArguments,
        metadata={
            "help": "Hugging face arguments for training the model (see transformers.TrainingArguments)."
        },
    )
    train_path: str = field(
        default="data/mdh/train.fasta",
        metadata={"help": "Path to training data."},
    )
    eval_path: str = field(
        default="data/mdh/valid.fasta",
        metadata={"help": "Path to validation data."},
    )
    base_model: str = field(
        default="facebook/esm2_t6_8M_UR50D",
        metadata={
            "help": "Base model to use for training. Or path to model config json file."
        },
    )
    tokenizer_path: str = field(
        default="tokenizer_esm_genslm",
        metadata={"help": "Path to tokenizer to use for training."},
    )
    sequence_homology_path: str = field(
        default="",
        metadata={"help": "Path to sequence homology cluster map (.npy)."},
    )
    num_eval_samples_per_epoch: int | None = field(
        default=None,
        metadata={
            "help": "Number of samples to use for evaluation per epoch. "
            "If None, use all samples."
        },
    )
    compute_codon_loss: bool = field(
        default=True, metadata={"help": "Whether to compute codon loss."}
    )
    compute_aminoacid_loss: bool = field(
        default=True, metadata={"help": "Whether to compute aminoacid loss."}
    )
    compute_contrastive_loss: bool = field(
        default=True, metadata={"help": "Whether to compute contrastive loss."}
    )
    contrastive_temperature: float = field(
        default=0.1, metadata={"help": "Temperature for contrastive loss."}
    )
    contrastive_pooler: str = field(
        default="mean", metadata={"help": "Pooling strategy for contrastive loss."}
    )
    wandb_project: str = field(
        default="",
        metadata={
            "help": "Wandb project name (By default, set to empty string to turn off wandb)."
        },
    )

    def __post_init__(self):
        self.training_args = TrainingArguments(**self.training_args)
        if self.compute_contrastive_loss:
            self.compute_codon_loss = self.compute_aminoacid_loss = True
        if not (self.compute_codon_loss or self.compute_aminoacid_loss):
            raise ValueError(
                "At least one of return_codon or return_aminoacid must be True"
            )
        # print(self)
        output_dir = Path(self.training_args.output_dir)

        # Create the output directory if it doesn't exist
        output_dir.mkdir(exist_ok=True, parents=True)

        # wandb needs to be initialized once on all node ranks
        if self.wandb_project and self.training_args.local_process_index == 0:
            os.environ["WANDB_PROJECT"] = self.wandb_project
            # Assign the same group name as the output directory
            # so that multi-node runs are grouped together
            wandb.init(dir=output_dir, group=output_dir.name)
            wandb.config.update({"train_config": asdict(self)}, allow_val_change=True)

        self.training_args.report_to = ["wandb" if self.wandb_project else ""]

        # Log the config to a yaml file
        with open(output_dir / "train_config.yaml", "w") as fp:
            yaml.dump(asdict(self), fp)


class ClearEvalMemoryTrainer(Trainer):
    """Trainer that clears the cuda cache before each evaluation (reduces OOMs for some models)."""

    def clear_cuda_cache(self) -> None:
        import gc
        import torch

        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()

    def evaluate(self, *args, **kwargs) -> dict[str, float]:
        self.clear_cuda_cache()
        return super().evaluate(*args, **kwargs)


def main():
    # Parse a yaml file to get the training config
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config) as fp:
        config = TrainingConfig(**yaml.safe_load(fp))

    # Load the tokenizer
    tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)

    # If we receive a hugging face json file instead of a model
    # name, load the config from that
    if config.base_model.endswith(".json"):
        # Load the json config
        with open(config.base_model) as fp:
            model_config = EsmConfig(**json.load(fp))

        # Load the model from the config
        model = EsmForContrastiveMaskedLM(
            config=model_config,
            compute_aminoacid_loss=config.compute_aminoacid_loss,
            compute_codon_loss=config.compute_codon_loss,
            compute_contrastive_loss=config.compute_contrastive_loss,
            contrastive_temperature=config.contrastive_temperature,
            contrastive_pooler=config.contrastive_pooler,
        )
    else:
        # Load the model from the model name
        model = EsmForContrastiveMaskedLM.from_pretrained(
            config.base_model,
            compute_aminoacid_loss=config.compute_aminoacid_loss,
            compute_codon_loss=config.compute_codon_loss,
            compute_contrastive_loss=config.compute_contrastive_loss,
            contrastive_temperature=config.contrastive_temperature,
            contrastive_pooler=config.contrastive_pooler,
        )

    # If the number of tokens in the tokenizer is different from the number of tokens
    # in the model resize the input embedding layer and the MLM prediction head
    model.update_model_weights(tokenizer)

    # Construct the train and validation datasets
    if config.train_path.endswith(".h5"):
        # If provided and HDF5 file, use the SequenceHomologySampler
        # which samples sequences from diverse clusters with high sequence homology
        hdf5_sampler = SequenceHomologySampler(
            file_path=config.sequence_homology_path,
            num_eval_samples=config.num_eval_samples_per_epoch,
        )
        train_dataset = HDF5Dataset(
            split="train",
            file_path=config.train_path,
            hdf5_sampler=hdf5_sampler,
            return_codon=config.compute_codon_loss,
            return_aminoacid=config.compute_aminoacid_loss,
        )
        # Use the same file for HDF5 eval (with different split)
        eval_dataset = HDF5Dataset(
            split="eval",
            file_path=config.train_path,
            hdf5_sampler=hdf5_sampler,
            return_codon=config.compute_codon_loss,
            return_aminoacid=config.compute_aminoacid_loss,
        )
    else:  # Fall back to fasta dataset
        train_dataset = FastaDataset(
            file_path=config.train_path,
            return_codon=config.compute_codon_loss,
            return_aminoacid=config.compute_aminoacid_loss,
        )
        eval_dataset = FastaDataset(
            file_path=config.eval_path,
            return_codon=config.compute_codon_loss,
            return_aminoacid=config.compute_aminoacid_loss,
        )

    data_collator = GenSLMColatorForLanguageModeling(
        return_codon=config.compute_codon_loss,
        return_aminoacid=config.compute_aminoacid_loss,
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
        pad_to_multiple_of=8 if config.training_args.fp16 else None,
    )

    trainer = ClearEvalMemoryTrainer(
        model=model,
        args=config.training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Attempt to load a checkpoint
    checkpoint = get_last_checkpoint(config.training_args.output_dir)
    if checkpoint is not None:
        print("Training from checkpoint:", checkpoint)

    # Train the model
    train_result = trainer.train(resume_from_checkpoint=checkpoint)

    # Saves the tokenizer too for easy upload
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    main()
