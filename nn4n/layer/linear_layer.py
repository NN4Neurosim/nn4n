import torch
import torch.nn as nn

import numpy as np
import nn4n.utils as utils


class LinearLayer(nn.Module):
    """
    Linear Layer with optional sparsity, excitatory/inhibitory, and plasticity constraints.
    The layer is initialized by passing specs in layer_struct.
    
    Required keywords in layer_struct:
        - input_dim: input dimension
        - output_dim: output dimension
        - weights: weight matrix init method/init weight matrix
        - biases: bias vector init method/init bias vector
        - sparsity_mask: mask for sparse connectivity
        - ei_mask: mask for Dale's law
        - plasticity_mask: mask for plasticity
    """
    def __init__(self, layer_struct):
        super().__init__()
        self.input_dim = layer_struct['input_dim']
        self.output_dim = layer_struct['output_dim']
        self.ei_mask = layer_struct['ei_mask'].T if layer_struct['ei_mask'] is not None else None
        self.sparsity_mask = layer_struct['sparsity_mask'].T if layer_struct['sparsity_mask'] is not None else None
        self.plasticity_mask = layer_struct['plasticity_mask'].T if layer_struct['plasticity_mask'] is not None else None
        self.plasticity_scales = torch.unique(self.plasticity_mask) # all unique plasticity values in the plasticity mask

        # generate weights
        self.weight = self._generate_weight(layer_struct['weights'])
        self.bias = self._generate_bias(layer_struct['biases'])

        # enfore constraints
        self._init_constraints()

        # convert weight and bias to torch tensor
        self.weight = nn.Parameter(self.weight, requires_grad=layer_struct['weights'] is not None)
        self.bias = nn.Parameter(self.bias, requires_grad=layer_struct['biases'] is not None)

    # INITIALIZATION
    # ======================================================================================
    def _init_constraints(self):
        """
        Initialize constraints
        It will also balance excitatory and inhibitory neurons
        """
        if self.sparsity_mask is not None:
            self.weight *= self.sparsity_mask
        if self.ei_mask is not None:
            self.weight[self.ei_mask == 1] = torch.clamp(self.weight[self.ei_mask == 1], min=0)
            self.weight[self.ei_mask == -1] = torch.clamp(self.weight[self.ei_mask == -1], max=0)
            self._balance_excitatory_inhibitory()

    def _balance_excitatory_inhibitory(self):
        """ Balance excitatory and inhibitory weights """
        scale_mat = torch.ones_like(self.weight)
        ext_sum = self.weight[self.sparsity_mask == 1].sum()
        inh_sum = self.weight[self.sparsity_mask == -1].sum()
        if ext_sum == 0 or inh_sum == 0:
            # automatically stop balancing if one of the sums is 0
            # devide by 10 to avoid recurrent explosion/decay
            self.weight /= 10
        else:
            if ext_sum > abs(inh_sum):
                _scale = abs(inh_sum).item() / ext_sum.item()
                scale_mat[self.sparsity_mask == 1] = _scale
            elif ext_sum < abs(inh_sum):
                _scale = ext_sum.item() / abs(inh_sum).item()
                scale_mat[self.sparsity_mask == -1] = _scale
            # apply scaling
            self.weight *= scale_mat

    def _generate_weight(self, weight_init):
        """ Generate random weight """
        if weight_init == 'uniform':
            # if uniform, let w be uniform in [-sqrt(k), sqrt(k)]
            sqrt_k = torch.sqrt(torch.tensor(1/self.input_dim))
            w = torch.rand(self.output_dim, self.input_dim) * sqrt_k
            w = w * 2 - sqrt_k
        elif weight_init == 'normal':
            w = torch.randn(self.output_dim, self.input_dim) / torch.sqrt(torch.tensor(self.input_dim))
        elif weight_init == 'zero':
            w = torch.zeros((self.output_dim, self.input_dim))
        elif type(weight_init) == np.ndarray:
            w = torch.from_numpy(weight_init)
        else:
            raise NotImplementedError
        return w.float()

    def _generate_bias(self, bias_init):
        """ Generate random bias """
        if bias_init == 'uniform':
            # if uniform, let b be uniform in [-sqrt(k), sqrt(k)]
            sqrt_k = torch.sqrt(torch.tensor(1/self.input_dim))
            b = torch.rand(self.output_dim) * sqrt_k
            b = b * 2 - sqrt_k
        elif bias_init == 'normal':
            b = torch.randn(self.output_dim) / torch.sqrt(torch.tensor(self.input_dim))
        elif bias_init == 'zero' or bias_init == None:
            b = torch.zeros(self.output_dim)
        elif type(bias_init) == np.ndarray:
            b = torch.from_numpy(bias_init)
        else:
            raise NotImplementedError
        return b.float()

    def auto_rescale(self, param_type):
        """ 
        Rescale weight or bias. This is useful when the layer is sparse 
        and insufficent/over-sufficient in driving the next layer dynamics
        """
        if param_type == 'weight':
            mat = self.weight.detach().clone()
        elif param_type == 'bias':
            mat = self.bias.detach().clone()
        else:
            raise NotImplementedError(f"Parameter type '{param_type}' is not implemented")

        if self.sparsity_mask is not None:
            scale = self.sparsity_mask.sum(axis=1).max() / self.input_dim
        else:
            scale = 1
        mat /= scale

        if param_type == 'weight':
            self.weight.data.copy_(mat)
        elif param_type == 'bias':
            self.bias.data.copy_(mat)
    # ======================================================================================

    # FORWARD
    # ======================================================================================
    def to(self, device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        if self.sparsity_mask is not None:
            self.sparsity_mask = self.sparsity_mask.to(device)
        if self.ei_mask is not None:
            self.ei_mask = self.ei_mask.to(device)
        if self.bias.requires_grad:
            self.bias = self.bias.to(device)

    def forward(self, x):
        """ 
        Forwardly update network

        Inputs:
            - x: input, shape: (batch_size, input_dim)

        Returns:
            - state: shape: (batch_size, hidden_size)
        """
        return x.float() @ self.weight.T + self.bias

    def apply_plasticity(self):
        """
        Apply plasticity mask to the weight gradient
        """
        with torch.no_grad():
            # assume the plasticity mask are all valid and being checked in ctrnn class
            for scale in self.plasticity_scales:
                if self.weight.grad is not None:
                    self.weight.grad[self.plasticity_mask == scale] *= scale
                else:
                    raise RuntimeError("Weight gradient is None, possibly because the forward loop is non-differentiable")

    def enforce_constraints(self):
        """ 
        Enforce sparsity and excitatory/inhibitory constraints if applicable.
        This is by default automatically called after each forward pass,
        but can be called manually if needed
        """
        if self.sparsity_mask is not None:
            self._enforce_sparsity()
        if self.ei_mask is not None:
            self._enforce_ei()

    def _enforce_sparsity(self):
        """ Enforce sparsity """
        w = self.weight.detach().clone() * self.sparsity_mask
        self.weight.data.copy_(torch.nn.Parameter(w))

    def _enforce_ei(self):
        """ Enforce Dale's law """
        w = self.weight.detach().clone()
        w[self.ei_mask == 1] = torch.clamp(w[self.ei_mask == 1], min=0)
        w[self.ei_mask == -1] = torch.clamp(w[self.ei_mask == -1], max=0)
        self.weight.data.copy_(torch.nn.Parameter(w))
    # ======================================================================================

    # HELPER FUNCTIONS
    # ======================================================================================
    def set_weight(self, weight):
        """ Set the value of the weight """
        assert weight.shape == self.weight.shape, f"Weight shape mismatch, expected {self.weight.shape}, got {weight.shape}"
        with torch.no_grad():
            self.weight.copy_(weight)

    def plot_layers(self):
        """ Plot the weights matrix and distribution of each layer """
        weight = self.weight.cpu() if self.weight.device != torch.device('cpu') else self.weight
        if weight.size(0) < weight.size(1):
            utils.plot_connectivity_matrix_dist(weight.detach().numpy(), "Weight Matrix (Transposed)", False, self.sparsity_mask is not None)
        else:
            utils.plot_connectivity_matrix_dist(weight.detach().numpy().T, "Weight Matrix", False, self.sparsity_mask is not None)
        
    def print_layers(self):
        """ Print the specs of each layer """
        param_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "weight_learnable": self.weight.requires_grad,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "bias_learnable": self.bias.requires_grad,
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.weight.nonzero().size(0) / self.weight.numel(),
        }
        utils.print_dict("Linear Layer", param_dict)
    # ======================================================================================
