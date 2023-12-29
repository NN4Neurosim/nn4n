import torch
import torch.nn as nn

import numpy as np
import nn4n.utils as utils


class HiddenLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            dist,
            use_bias,
            scaling,
            mask,
            positivity_constraints,
            sparsity_constraints,
            self_connections,
            learnable=True,
            ) -> None:
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden units
            @param dist: distribution of hidden weights
            @param use_bias: use bias or not
            @param sparsity_constraints: use sparsity_constraints or not
            @param scaling: scaling of hidden weights
            @param mask: mask for hidden weights, used to enforce sparsity_constraints and/or dale's law
            @param positivity_constraints: whether to enforce positivity constraint
            @param self_connections: allow self connections or not
        """
        super().__init__()
        # some params are for verbose printing
        self.hidden_size = hidden_size
        self.dist = dist
        self.use_bias = use_bias
        self.scaling = scaling
        self.positivity_constraints = positivity_constraints
        self.sparsity_constraints = sparsity_constraints
        self.self_connections = self_connections
        # generate weights and bias
        self.weight = self._generate_weight()
        self.bias = self._generate_bias()
        # init sparsity_constraints, positivity_constraints and scaling
        self.sparsity_mask, self.positivity_mask = None, None
        self._init_constraints(mask)
        self._enforce_scaling()
        # parameterize the weights and bias
        self.weight = nn.Parameter(self.weight, requires_grad=learnable)
        self.bias = nn.Parameter(self.bias, requires_grad=self.use_bias and learnable)

    # INITIALIZATION
    # ======================================================================================
    def _generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            sqrt_k = torch.sqrt(torch.tensor(1/self.hidden_size))
            w = torch.rand(self.hidden_size, self.hidden_size) * sqrt_k
            w = w * 2 - sqrt_k
        elif self.dist == 'normal':
            w = torch.randn(self.hidden_size, self.hidden_size) / torch.sqrt(torch.tensor(self.hidden_size))
        elif self.dist == 'zero':
            w = torch.zeros((self.output_dim, self.input_dim))
        else:
            raise NotImplementedError

        return w.float()

    def _generate_bias(self):
        return torch.zeros(self.hidden_size).float()

    def _init_constraints(self, mask):
        """
        Initialize constraints
        It will also balance excitatory and inhibitory neurons
        """
        # Initialize dale's law and sparsity_constraints masks
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).int()  # convert to torch.Tensor
            if self.positivity_constraints:
                self._init_positivity_mask(mask)  # initialize dale's mask
            if self.sparsity_constraints:
                self.sparsity_mask = (mask != 0).int()  # convert to a binary mask
        
        # Whether to delete self connections
        if not self.self_connections:
            if self.sparsity_mask is None:
                # if mask is not provided, create a mask
                self.sparsity_mask = torch.ones((self.hidden_size, self.hidden_size))
            self.sparsity_mask = torch.where(torch.eye(self.hidden_size) == 1, 0, self.sparsity_mask)
        
        # Apply dale's law and sparsity_constraints
        if self.sparsity_constraints or not self.self_connections:
            self.weight *= self.sparsity_mask
        if self.positivity_constraints:
            self.weight[self.positivity_mask == 1] = torch.clamp(self.weight[self.positivity_mask == 1], min=0)
            self.weight[self.positivity_mask == -1] = torch.clamp(self.weight[self.positivity_mask == -1], max=0)
            self._balance_excitatory_inhibitory()

    def _init_positivity_mask(self, mask):
        """ initialize settings required for sparsity_constraints """
        self.positivity_mask = torch.zeros(mask.shape, dtype=int)
        self.positivity_mask[mask < 0] = -1
        self.positivity_mask[mask > 0] = 1

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
        # raise NotImplementedError
    # ======================================================================================

    # FORWARD
    # ======================================================================================
    def forward(self, x):
        """ Forward """
        return x.float() @ self.weight.T + self.bias

    def _enforce_spec_rad(self):
        """ Enforce spectral radius """
        # Calculate scale
        print('WARNING: spectral radius not applied, the feature is deprecated, use scaling instead')

    def _enforce_scaling(self):
        """ Enforce scaling """
        # Calculate scale
        self.weight.data *= self.scaling

    def enforce_constraints(self):
        """ Enforce constraints """
        if self.sparsity_mask is not None:
            self._enforce_sparsity()
        if self.positivity_mask is not None:
            self._enforce_positivity()

    def _enforce_sparsity(self):
        """ Enforce sparsity """
        w = self.weight.detach().clone() * self.sparsity_mask
        self.weight.data.copy_(torch.nn.Parameter(w))

    def _enforce_positivity(self):
        """ Enforce Dale's law """
        w = self.weight.detach().clone()
        w[self.positivity_mask == 1] = torch.clamp(w[self.positivity_mask == 1], min=0)
        w[self.positivity_mask == -1] = torch.clamp(w[self.positivity_mask == -1], max=0)
        self.weight.data.copy_(torch.nn.Parameter(w))
    # ======================================================================================

    # HELPER FUNCTIONS
    # ======================================================================================
    def to(self, device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        if self.sparsity_mask is not None:
            self.sparsity_mask = self.sparsity_mask.to(device)
        if self.positivity_mask is not None:
            self.positivity_mask = self.positivity_mask.to(device)
        if not self.use_bias:
            self.bias = self.bias.to(device)

    def plot_layers(self):
        """ Plot weight """
        weight = self.weight.cpu() if self.weight.device != torch.device('cpu') else self.weight
        utils.plot_connectivity_matrix_dist(weight.detach().numpy(), "Hidden Layer", False, self.sparsity_constraints)
    
    def print_layers(self):
        # plot weight matrix
        param_dict = {
            "self_connections": self.self_connections,
            "input/output_dim": self.hidden_size,
            "distribution": self.dist,
            "use_bias": self.use_bias,
            "positivity_constraints": self.positivity_constraints,
            "shape": self.weight.shape,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "weight_mean": self.weight.mean().item(),
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.sparsity_mask.sum() / self.sparsity_mask.numel() if self.sparsity_mask is not None else 1,
            "scaling": self.scaling,
            "sparsity_constraints": self.sparsity_constraints,
            # "spectral_radius": torch.abs(torch.linalg.eig(self.weight)[0]).max().item(),
        }
        utils.print_dict("Hidden Layer", param_dict)
        # utils.plot_eigenvalues(weight.detach().numpy(), "Hidden Layer")
    # ======================================================================================
