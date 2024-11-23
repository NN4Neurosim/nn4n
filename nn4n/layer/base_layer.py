import torch
import torch.nn as nn

import numpy as np
import nn4n.utils as utils


class BaseLayer(nn.Module):
    """
    Linear Layer with optional sparsity, excitatory/inhibitory, and plasticity constraints.
    The layer is initialized by passing specs in layer_struct.

    Required keywords in layer_struct:
        - input_dim: dimension of input
        - output_dim: dimension of output
        - weight: weight matrix init method/init weight matrix, default: 'uniform'
        - bias: bias vector init method/init bias vector, default: 'uniform'
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_dist = weight
        self.bias_dist = bias
        self.weight = self._generate_weight(self.weight_dist)
        self.bias = self._generate_bias(self.bias_dist)
        self.ei_mask = ei_mask.T if ei_mask is not None else None
        self.sparsity_mask = sparsity_mask.T if sparsity_mask is not None else None
        self.plasticity_mask = (
            plasticity_mask.T if plasticity_mask is not None else None
        )
        # All unique plasticity values in the plasticity mask
        self.plasticity_scales = (
            torch.unique(self.plasticity_mask)
            if self.plasticity_mask is not None
            else None
        )

        self._init_trainable()
        self._check_layer()

    # INITIALIZATION
    # ======================================================================================
    @staticmethod
    def _check_keys(layer_struct):
        required_keys = ["input_dim", "output_dim"]
        for key in required_keys:
            if key not in layer_struct:
                raise ValueError(f"Key '{key}' is missing in layer_struct")
        
        valid_keys = ["input_dim", "output_dim", "weight", "bias", "ei_mask", "sparsity_mask", "plasticity_mask"]
        for key in layer_struct.keys():
            if key not in valid_keys:
                raise ValueError(f"Key '{key}' is not a valid key in layer_struct")

    @classmethod
    def from_dict(cls, layer_struct):
        """
        Alternative constructor to initialize LinearLayer from a dictionary.
        """
        # Create an instance using the dictionary values
        cls._check_keys(layer_struct)
        instance = cls(
            input_dim=layer_struct["input_dim"],
            output_dim=layer_struct["output_dim"],
            weight=layer_struct.get("weight", "uniform"),
            bias=layer_struct.get("bias", "uniform"),
            ei_mask=layer_struct.get("ei_mask"),
            sparsity_mask=layer_struct.get("sparsity_mask"),
            plasticity_mask=layer_struct.get("plasticity_mask"),
        )
        # Initialize the trainable parameters then check the layer
        instance._init_trainable()
        instance._check_layer()

        return instance

    def _init_trainable(self):
        # enfore constraints
        self._init_constraints()
        # convert weight and bias to torch tensor
        self.weight = nn.Parameter(
            self.weight, requires_grad=self.weight_dist is not None
        )
        self.bias = nn.Parameter(self.bias, requires_grad=self.bias_dist is not None)

    def _check_layer(self):
        pass
