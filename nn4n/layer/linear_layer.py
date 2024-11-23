import torch
import torch.nn as nn

import numpy as np
import nn4n.utils as utils
from .base_layer import BaseLayer


class LinearLayer(BaseLayer):
    """
    Linear Layer with optional sparsity, excitatory/inhibitory, and plasticity constraints.
    The layer is initialized by passing specs in layer_struct.
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

    def auto_rescale(self, param_type):
        """
        Rescale weight or bias. This is useful when the layer is sparse
        and insufficent/over-sufficient in driving the next layer dynamics
        """
        if param_type == "weight":
            mat = self.weight.detach().clone()
        elif param_type == "bias":
            mat = self.bias.detach().clone()
        else:
            raise NotImplementedError(
                f"Parameter type '{param_type}' is not implemented"
            )

        if self.sparsity_mask is not None:
            scale = self.sparsity_mask.sum(axis=1).max() / self.input_dim
        else:
            scale = 1
        mat /= scale

        if param_type == "weight":
            self.weight.data.copy_(mat)
        elif param_type == "bias":
            self.bias.data.copy_(mat)
    # ======================================================================================
