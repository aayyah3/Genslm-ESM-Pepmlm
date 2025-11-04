"""

    This script will take a model specified by path and convert it from 
    the OG esmc model to the genslm-esm model. Changes are:
        - Add codon embedding layer with weights reflecting the AA embedding
        - Add codon decoding head with weights reflecting the AA decoder
        - Add contrastive loss head
"""

from typing import Optional, Tuple, Union
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import logging, EsmTokenizer
from esm.models.esmc import ESMC
from esm.layers.regression_head import RegressionHead

from genslm_esm.dataset import translation_table
from esm.pretrained import register_local_model, ESMC_300M_202412, ESMC_600M_202412
from genslm_esm.modeling_emsc import ContrastiveEsmConfig
logging.set_verbosity_info()
model_rel = {
    "ESMC_300M": {
        "d_model"=960, "n_heads"=15, "n_layers"=30
    },
    "ESMC_600M": {
        "d_model"=1152, "n_heads"=18, "n_layers"=36
    }
}
def convert_model(model_name_or_path: str, output_dir: str):
    # Load the model
    model_codons = True
    model_aa = True
    contrastive_loss = True
    model_path = "/lus/eagle/projects/FoundEpidem/azton/esmc_models/esmc_300m"
    model_spec = 'ESMC_300M'
    register_local_model(model_path, ESMC_300M_202412)
    config = ContrastiveEsmConfig(model_name=model_path, compute_contrastive_loss=True)
    model = ESMC.from_pretrained(model_path)

    tokenizer_path = "/lus/eagle/projects/CVD-Mol-AI/braceal/src/genslm-esm/tokenizer_esm_genslm"
    tokenizer = EsmTokenizer.from_pretrained(tokenizer_path)

    model.tokenizer = tokenizer
    # build a config so that we can more easily use a 
    # .from_pretrained() method
    model.config = config
    model.config.d_model = model_rel[model_spec]["d_model"]
    model.config.n_heads = model_rel[model_spec]["n_heads"]
    model.config.n_layers = model_rel[model_spec]["n_layers"]

    # add relevant heads for modeling
    
    

    

    