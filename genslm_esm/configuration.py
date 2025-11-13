"""Configuration for the GensLM-ESMC model."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class GenslmEsmcConfig(PretrainedConfig):
    """Configuration for the GenSLM-ESMC model."""

    # Set the model type to 'contrastive-esmc'
    # This is used to identify the model type in the config file
    model_type = 'genslm-esmc'
    # Set the model architecture to 'EsmCForContrastiveMaskedLM'
    # This is used to identify the model architecture in the config file
    architecture = 'EsmCForContrastiveMaskedLM'

    def __init__(
        self,
        d_model: int = 960,
        n_heads: int = 15,
        n_layers: int = 30,
        contrastive_temperature: float = 0.1,
        use_flash_attn: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the configuration."""
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.contrastive_temperature = contrastive_temperature
        self.use_flash_attn = use_flash_attn
        super().__init__(**kwargs)


if __name__ == '__main__':
    config = GenslmEsmcConfig()
    # Set the architectures and auto_map to use EsmCForContrastiveMaskedLM
    config.architectures = ['GenslmEsmcModel']
    config.auto_map = {
        'AutoModel': 'genslm_esm.modeling.GenslmEsmcModel',
        'AutoTokenizer': 'transformers.models.esm.tokenization_esm.EsmTokenizer',  # noqa: E501
    }
    config.library_name = 'genslm_esm'
    config._name_or_path = (
        'genslm-test/genslm-test-v1.5'  #'genslm/genslm-esmc-300m-contrastive'
    )
    config.save_pretrained('genslm-esmc')
