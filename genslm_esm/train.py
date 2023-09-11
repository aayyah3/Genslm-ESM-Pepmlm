from dataclasses import dataclass

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
    # base_model: str = "dev_test_reinit_refactor_v1/checkpoint-100"
    tokenizer_path: str = "tokenizer_esm_genslm"
    output_path: str = "dev_test_reinit_refactor_v1"
    data_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/mdh_sc23/fasta/mdh_natural_sequences.ffn"
    # data_path: str = "/lambda_stor/homes/khippe/genslm_foundation/genome_data/curriculum_datasets/curriculum_2/curriculum_2_train.h5"

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
        save_steps=50,
        fp16=True,
        push_to_hub=False,
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

    print("init success")

    # TODO: During fine tuning or training from a checkpoint, this will restart the weights
    #       ONly do if the len(tokenizer) is not the same as the embedding layer
    # TODO: We should move this logic inside the model init, to insure proper weight initialization
    # Inject new vocabulary (modifies model.config)
    # if len(tokenizer) != model.config.vocab_size:
    #     model.resize_token_embeddings(len(tokenizer))
    #     # Make a new lm_head with uninitialized weights using the correct shape
    #     model.lm_head = EsmLMHead(model.config)

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

    # Attempt to load a checkpoint
    # checkpoint = None
    # if Path(args.output_dir).exists():
    checkpoint = get_last_checkpoint(args.output_dir)
    # model_id = config.base_model if checkpoint is None else checkpoint

    print("Training from checkpoint:", checkpoint)

    trainer.train(resume_from_checkpoint=checkpoint)


if __name__ == "__main__":
    main()
    # TODO: Setup checkpoint resume logic
    # TODO: Add validation dataset
