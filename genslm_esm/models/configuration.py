"""Configuration for the GensLM-ESMC model."""

from __future__ import annotations

from transformers import PretrainedConfig


class ContrastiveEsmCConfig(PretrainedConfig):
    """Add contrastive loss parameters to the ESM config."""

    def __init__(  # noqa: PLR0913
        self,
        model_name: str = 'esmc_300m',
        base_model_path: str | None = None,
        d_model: int = 960,
        n_heads: int = 15,
        n_layers: int = 30,
        compute_aminoacid_loss: bool = True,
        compute_codon_loss: bool = False,
        compute_contrastive_loss: bool = False,
        contrastive_temperature: float = 0.1,
        contrastive_pooler: str = 'mean',
        use_flash_attn: bool = False,
        **kwargs,
    ):
        """Initialize the configuration."""
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.d_model = d_model if '300M' in model_name else 1152
        self.n_heads = n_heads if '300M' in model_name else 18
        self.n_layers = n_layers if '300M' in model_name else 36
        self.compute_aminoacid_loss = compute_aminoacid_loss
        self.compute_codon_loss = compute_codon_loss
        self.compute_contrastive_loss = compute_contrastive_loss
        self.contrastive_temperature = contrastive_temperature
        self.contrastive_pooler = contrastive_pooler
        self.use_flash_attn = use_flash_attn
        super().__init__(**kwargs)


if __name__ == '__main__':
    contrastive_esmc_config = ContrastiveEsmCConfig(
        model_name='esmc_300m',
        base_model_path='esm_esmc_300m',
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer_name_or_path='esm_esmc_300m',
        compute_aminoacid_loss=True,
        compute_codon_loss=True,
        compute_contrastive_loss=True,
        contrastive_temperature=0.1,
        contrastive_pooler='mean',
    )
    contrastive_esmc_config.save_pretrained('custom-esmc')
