"""Configuration for the GensLM-ESMC model."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class GenslmEsmcConfig(PretrainedConfig):
    """Configuration for the GenSLM-ESMC model."""

    # This is used to identify the model type in the config file
    model_type = 'genslm-esmc'

    # Set the model architecture to 'GenslmEsmcModel'
    # This is used to identify the model architecture in the config file
    architecture = 'GenslmEsmcModel'

    def __init__(
        self,
        d_model: int = 960,
        n_heads: int = 15,
        n_layers: int = 30,
        contrastive_temperature: float = 0.1,
        use_flash_attn: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the configuration.

        Parameters
        ----------
        d_model: int
            The dimension of the model.
        n_heads: int
            The number of heads.
        n_layers: int
            The number of layers.
        contrastive_temperature: float
            The temperature for the contrastive loss.
        use_flash_attn: bool
            Whether to use flash attention.
        kwargs: Any
            Additional keyword arguments.
        """
        # Set the model parameters
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.contrastive_temperature = contrastive_temperature
        self.use_flash_attn = use_flash_attn
        super().__init__(**kwargs)

    def set_hf_metadata(self, model_name: str) -> GenslmEsmcConfig:
        """Set the Hugging Face metadata for the model.

        Parameters
        ----------
        model_name: str
            The path to the model.

        Returns
        -------
        GenslmEsmcConfig
            The configuration with the Hugging Face metadata set.
        """
        self.architectures = ['GenslmEsmcModel']
        self.auto_map = {
            'AutoConfig': 'configuration.GenslmEsmcConfig',
            'AutoModel': 'modeling.GenslmEsmcModel',
            'AutoTokenizer': 'transformers.models.esm.tokenization_esm.EsmTokenizer',  # noqa: E501
        }
        self.library_name = 'genslm_esm'
        self._name_or_path = model_name
        return self


if __name__ == '__main__':
    #'genslm/genslm-esmc-300m-contrastive'
    _name_or_path = 'genslm-test/genslm-test-v1.5'
    config = GenslmEsmcConfig()
    # Set the Hugging Face metadata for the model
    config.set_hf_metadata(_name_or_path)
    # Save the configuration to the local directory
    config.save_pretrained('genslm-esmc')
