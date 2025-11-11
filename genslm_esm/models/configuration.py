"""Configuration for the GensLM-ESMC model."""

from __future__ import annotations

from transformers import PretrainedConfig


class ContrastiveEsmCConfig(PretrainedConfig):
    """Add contrastive loss parameters to the ESM config."""

    def __init__(
        self,
        model_name: str = 'esmc_300m',
        base_model_path: str | None = None,
        d_model: int = 960,
        n_heads: int = 15,
        n_layers: int = 30,
        contrastive_temperature: float = 0.1,
        use_flash_attn: bool = False,
        **kwargs,
    ):
        """Initialize the configuration."""
        self.model_name = model_name
        self.base_model_path = base_model_path
        self.d_model = d_model if '300M' in model_name.upper() else 1152
        self.n_heads = n_heads if '300M' in model_name.upper() else 18
        self.n_layers = n_layers if '300M' in model_name.upper() else 36
        self.contrastive_temperature = contrastive_temperature
        self.use_flash_attn = use_flash_attn
        super().__init__(**kwargs)


if __name__ == '__main__':
    contrastive_esmc_config = ContrastiveEsmCConfig(
        model_name='esmc_300m',
        base_model_path='esm_esmc_300m',
    )
    contrastive_esmc_config.save_pretrained('custom-esmc')
