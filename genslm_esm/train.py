import os
import wandb
from dataclasses import dataclass, asdict
from typing import Union

from transformers import EsmTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

from genslm_esm.dataset import (
    FastaDataset,
    GenSLMColatorForLanguageModeling,
    HDF5Dataset,
)
from genslm_esm.modeling_esm import EsmForContrastiveMaskedLM


@dataclass
class GenSLMTrainingConfig:
    compute_codon_loss: bool = True
    compute_aminoacid_loss: bool = True
    compute_contrastive_loss: bool = True
    contrastive_temperature: float = 0.1
    contrastive_pooler: str = "mean"
    base_model: str = "facebook/esm2_t6_8M_UR50D"
    tokenizer_path: str = "tokenizer_esm_genslm"
    output_path: str = "dev_test_reinit_refactor_v1"
    train_path: str = "data/mdh/train.fasta"
    validation_path: str = "data/mdh/valid.fasta"
    # train_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/curriculum_datasets/curriculum_2/curriculum_2_train.h5"
    wandb_project: str = "genslm-esm"  # Set to empty string to turn off wandb

    def __post_init__(self):
        if self.compute_contrastive_loss:
            self.compute_codon_loss = self.compute_aminoacid_loss = True
        if not (self.compute_codon_loss or self.compute_aminoacid_loss):
            raise ValueError(
                "At least one of return_codon or return_aminoacid must be True"
            )

        # Setting this environment variable enables wandb logging
        if self.wandb_project:
            os.environ["WANDB_PROJECT"] = self.wandb_project
            wandb.init()
            wandb.config.update(asdict(self))

    def construct_dataset(self, file_path: str) -> Union[FastaDataset, HDF5Dataset]:
        dset_class = HDF5Dataset if file_path.endswith(".h5") else FastaDataset
        return dset_class(
            file_path=file_path,
            return_codon=self.compute_codon_loss,
            return_aminoacid=self.compute_aminoacid_loss,
        )


def main():
    config = GenSLMTrainingConfig()

    # TODO: This would be a good option to try for more efficient packing: group_by_length
    args = TrainingArguments(
        output_dir=config.output_path,
        per_device_train_batch_size=64,
        per_device_eval_batch_size=128,
        evaluation_strategy="steps",
        eval_steps=50,
        logging_steps=50,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=50,
        fp16=True,
        push_to_hub=False,
        report_to="wandb" if config.wandb_project else None,
        remove_unused_columns=False,  # This skips underlying logic in Trainer which modifies the data_collator
        dataloader_num_workers=4,  # Defaults to 0, may want to increase for faster data loading
    )

    tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)
    model = EsmForContrastiveMaskedLM.from_pretrained(
        config.base_model,
        compute_contrastive_loss=config.compute_contrastive_loss,
        contrastive_temperature=config.contrastive_temperature,
        contrastive_pooler=config.contrastive_pooler,
    )

    # If the number of tokens in the tokenizer is different from the number of tokens
    # in the model resize the input embedding layer and the MLM prediction head
    model.resize_model_vocab(len(tokenizer))

    # Construct the train and validation datasets
    train_dataset = config.construct_dataset(config.train_path)
    eval_dataset = config.construct_dataset(config.validation_path)

    data_collator = GenSLMColatorForLanguageModeling(
        return_codon=config.compute_codon_loss,
        return_aminoacid=config.compute_aminoacid_loss,
        train_mode=True,
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Attempt to load a checkpoint
    checkpoint = get_last_checkpoint(args.output_dir)
    if checkpoint is not None:
        print("Training from checkpoint:", checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
