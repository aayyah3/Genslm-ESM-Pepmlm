from dataclasses import dataclass
from transformers import Trainer, TrainingArguments, EsmTokenizer
from transformers.models.esm.modeling_esm import EsmLMHead

from genslm_esm.dataset import (
    FastaDataset,
    HDF5Dataset,
    GenSLMColatorForLanguageModeling,
)

from genslm_esm.modeling_esm import EsmForContrastiveMaskedLM, ContrastiveEsmConfig


@dataclass
class GenSLMTrainingConfig:
    compute_codon_loss: bool = True
    compute_aminoacid_loss: bool = True
    compute_contrastive_loss: bool = False
    temperature: float = 0.1
    max_length: int = 1024
    base_model: str = "facebook/esm2_t6_8M_UR50D"
    tokenizer_path: str = "tokenizer_esm_genslm"
    output_path: str = "mdh_natural_sequences_run_1"
    # data_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/mdh_sc23/fasta/mdh_natural_sequences.ffn"
    data_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/curriculum_datasets/curriculum_2/curriculum_2_train.h5"

    def __post_init__(self):
        if self.compute_contrastive_loss:
            self.compute_codon_loss = self.compute_aminoacid_loss = True
        if not (self.compute_codon_loss or self.compute_aminoacid_loss):
            raise ValueError(
                "At least one of return_codon or return_aminoacid must be True"
            )


def main():
    config = GenSLMTrainingConfig()

    # TODO: This would be a good option to try for more efficient packing: group_by_length
    args = TrainingArguments(
        output_dir=config.output_path,
        per_device_train_batch_size=64,
        # per_device_eval_batch_size=128,
        # evaluation_strategy="steps",
        # eval_steps=50,
        logging_steps=50,
        gradient_accumulation_steps=2,
        num_train_epochs=20,
        weight_decay=0.1,
        warmup_steps=1_000,
        lr_scheduler_type="cosine",
        learning_rate=5e-4,
        save_steps=500,
        fp16=True,
        push_to_hub=False,
        remove_unused_columns=False,  # This skips underlying logic in Trainer which modifies the data_collator
        dataloader_num_workers=4,  # Defaults to 0, may want to increase for faster data loading
    )

    tokenizer = EsmTokenizer.from_pretrained(config.tokenizer_path)
    model_config = ContrastiveEsmConfig(
        compute_contrastive_loss=config.compute_contrastive_loss
    )
    model = EsmForContrastiveMaskedLM.from_pretrained(
        config.base_model, config=model_config
    )

    # TODO: During fine tuning or training from a checkpoint, this will restart the weights
    #       ONly do if the len(tokenizer) is not the same as the embedding layer
    # TODO: We should move this logic inside the model init, to insure proper weight initialization
    # Inject new vocabulary (modifies model.config)
    model.resize_token_embeddings(len(tokenizer))
    # Make a new lm_head with uninitialized weights using the correct shape
    model.lm_head = EsmLMHead(model.config)

    # Select the dataset type based on the file extension
    dset_class = HDF5Dataset if config.data_path.endswith(".h5") else FastaDataset
    train_dataset = dset_class(
        file_path=config.data_path,
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
    )

    trainer = Trainer(
        model=model, args=args, data_collator=data_collator, train_dataset=train_dataset
    )

    trainer.train()


if __name__ == "__main__":
    main()
    # TODO: Setup checkpoint resume logic
    # TODO: Add validation dataset
    # TODO: Consider adding a loss weight to balance contrastive and MLM losses
