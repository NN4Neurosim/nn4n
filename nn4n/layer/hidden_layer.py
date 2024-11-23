import torch
import torch.nn as nn

import numpy as np
import nn4n.utils as utils
from .base_layer import BaseLayer


class HiddenLayer(BaseLayer):
    """
    Hidden layer of the RNN. The layer is initialized by passing specs in layer_struct.

    Required keywords in layer_struct:
        - input_dim: input dimension
        - output_dim: output dimension
        - weight: weight matrix init method/init weight matrix
        - bias: bias vector init method/init bias vector
        - sparsity_mask: mask for sparse connectivity
        - ei_mask: mask for Dale's law
        - plasticity_mask: mask for plasticity
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        weight: str = "uniform",
        bias: str = "uniform",
        ei_mask: torch.Tensor = None,
        sparsity_mask: torch.Tensor = None,
        plasticity_mask: torch.Tensor = None,
    ):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            weight=weight,
            bias=bias,
            ei_mask=ei_mask,
            sparsity_mask=sparsity_mask,
            plasticity_mask=plasticity_mask,
        )

    # INITIALIZATION
    # ======================================================================================
    @staticmethod
    def _check_keys(layer_struct):
        BaseLayer._check_keys(layer_struct)
    # ======================================================================================
