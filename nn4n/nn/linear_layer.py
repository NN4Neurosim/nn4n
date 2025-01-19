import nn4n
import torch
import numpy as np
import nn4n.utils as utils
from .module import Module


class LinearLayer(Module):
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
        **kwargs
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_dist = weight
        self.bias_dist = bias
        self.weight = self._generate_weight(self.weight_dist)
        self.bias = self._generate_bias(self.bias_dist)

        # Call super init after initializing weight and bias
        super().__init__(**kwargs)

    # INITIALIZATION
    # ======================================================================================
    def _generate_bias(self, bias_init):
        """Generate random bias"""
        if bias_init == "uniform":
            # If uniform, let b be uniform in [-sqrt(k), sqrt(k)]
            sqrt_k = torch.sqrt(torch.tensor(1 / self.input_dim))
            b = torch.rand(self.output_dim) * sqrt_k
            b = b * 2 - sqrt_k
        elif bias_init == "normal":
            b = torch.randn(self.output_dim) / torch.sqrt(torch.tensor(self.input_dim))
        elif bias_init == "zero" or bias_init == None:
            b = torch.zeros(self.output_dim)
        elif type(bias_init) == np.ndarray:
            b = torch.from_numpy(bias_init)
        else:
            raise NotImplementedError
        return b.float()

    def _generate_weight(self, weight_init):
        """Generate random weight"""
        if weight_init == "uniform":
            # If uniform, let w be uniform in [-sqrt(k), sqrt(k)]
            sqrt_k = torch.sqrt(torch.tensor(1 / self.input_dim))
            w = torch.rand(self.output_dim, self.input_dim) * sqrt_k
            w = w * 2 - sqrt_k
        elif weight_init == "normal":
            w = torch.randn(self.output_dim, self.input_dim) / torch.sqrt(
                torch.tensor(self.input_dim)
            )
        elif weight_init == "zero":
            w = torch.zeros((self.output_dim, self.input_dim))
        elif type(weight_init) == np.ndarray:
            w = torch.from_numpy(weight_init)
        else:
            raise NotImplementedError
        return w.float()

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

    # TRAINING
    # ======================================================================================
    def forward(self, x):
        """
        Forwardly update network

        Inputs:
            - x: input, shape: (batch_size, input_dim)

        Returns:
            - state: shape: (batch_size, hidden_size)
        """
        return x.float() @ self.weight.T + self.bias

    def freeze(self):
        """Freeze the layer"""
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def unfreeze(self):
        """Unfreeze the layer"""
        self.weight.requires_grad = True
        self.bias.requires_grad = self.bias_dist is not None

    # HELPER FUNCTIONS
    # ======================================================================================
    def get_specs(self):
        """Print the specs of each layer"""
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "weight_learnable": self.weight.requires_grad,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "bias_learnable": self.bias.requires_grad,
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": (
                self.sparsity_mask.sum() / self.sparsity_mask.numel()
                if self.sparsity_mask is not None
                else 1
            )
        }

    def print_layer(self):
        """
        Print the specs of the layer
        """
        utils.print_dict(f"{self.__class__.__name__} layer", self.get_specs())