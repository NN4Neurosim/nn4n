import torch
import torch.nn as nn

import numpy as np
import nn4n.utils as utils


class LinearLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            dist,
            use_bias,
            mask,
            positivity_constraints,
            sparsity_constraints,
            learnable=True,
            ) -> None:
        """
        Sparse Linear Layer
        TODO: add range, add warning when using normal dist as negative values may exist
        Parameters:
            @param use_bias: whether to use bias
            @param mask: mask for sparse connectivity
            @param dist: distribution of weights
            @param input_dim: input dimension
            @param output_dim: output dimension
            @param positivity_constraints: whether to use Dale's law
            @param sparsity_constraints: whether to use sparsity_constraints
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dist = dist
        self.use_bias = use_bias
        self.positivity_constraints = positivity_constraints
        self.sparsity_constraints = sparsity_constraints

        # generate weights
        self.weight = self._generate_weight()
        self.bias = self._generate_bias()

        # enfore constraints
        self.positivity_mask, self.sparsity_mask = None, None
        self._init_constraints(mask)

        # convert weight and bias to torch tensor
        self.weight = nn.Parameter(self.weight, requires_grad=learnable)
        self.bias = nn.Parameter(self.bias, requires_grad=self.use_bias and learnable)

    # INITIALIZATION
    # ======================================================================================
    def _init_constraints(self, mask):
        """
        Initialize constraints (because _enforce_positivity will clip the weights, but we don't want that during initialization)
        It will also balance excitatory and inhibitory neurons
        """
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).int()  # convert to torch.Tensor
            if self.positivity_constraints:
                self._init_positivity_mask(mask)
            if self.sparsity_constraints:
                self.sparsity_mask = (mask != 0).float()

        if self.positivity_constraints:
            self.weight *= self.positivity_mask
            # self._balance_excitatory_inhibitory()
        if self.sparsity_constraints:
            self.weight *= self.sparsity_mask
            self.weight = self._rescale_weight_bias(self.weight)
            # self.bias = self._rescale_weight_bias(self.bias)

    def _balance_excitatory_inhibitory(self):
        """ Balance excitatory and inhibitory weights """
        scale_mat = torch.ones_like(self.weight)
        ext_sum = self.weight[self.positivity_mask == 1].sum()
        inh_sum = self.weight[self.positivity_mask == -1].sum()
        if ext_sum > abs(inh_sum):
            scale_mat[self.positivity_mask == 1] = abs(inh_sum) / ext_sum
        elif ext_sum < abs(inh_sum):
            scale_mat[self.positivity_mask == -1] = ext_sum / abs(inh_sum)
        self.weight *= scale_mat

    def _init_positivity_mask(self, mask):
        """ initialize settings required for Dale's law """
        # # Dale's law only applies to output edges
        # # create a ei_list to store whether a neuron's output edges are all positive or all negative
        # ei_list = torch.zeros(mask.shape[1])
        # all_neg = torch.all(mask <= 0, axis=0)
        # all_pos = torch.all(mask >= 0, axis=0)
        # all_zero = torch.all(mask == 0, axis=0)  # readout layer may be sparse

        # # check if all neurons are either all negative or all positive
        # for i in range(mask.shape[1]):
        #     # if this neuron has all negative or all positive output edges, set it to -1 or 1
        #     if all_neg[i]:
        #         ei_list[i] = -1
        #     elif all_pos[i]:
        #         ei_list[i] = 1
        #     else:
        #         assert False, "a neuron's output edges must be either all positive or all negative"

        #     # if this neuron has no output edges, set it to 0 too
        #     if all_zero[i]:
        #         ei_list[i] = 0
        # self.ei_list = ei_list

        # # create a mask for Dale's law
        # self.positivity_mask = torch.ones(mask.shape, dtype=int)
        # self.positivity_mask[:, self.ei_list == -1] = -1
        """ initialize settings required for sparsity_constraints """
        self.positivity_mask = torch.ones(mask.shape, dtype=int)
        self.positivity_mask[mask < 0] = -1
        self.positivity_mask[mask > 0] = 1

    def _generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            # if uniform, let w be uniform in [-sqrt(k), sqrt(k)]
            sqrt_k = torch.sqrt(torch.tensor(1/self.input_dim))
            w = torch.rand(self.output_dim, self.input_dim) * sqrt_k
            w = w * 2 - sqrt_k
        elif self.dist == 'normal':
            w = torch.randn(self.output_dim, self.input_dim) / torch.sqrt(torch.tensor(self.input_dim))
        elif self.dist == 'zero':
            w = torch.zeros((self.output_dim, self.input_dim))
        else:
            raise NotImplementedError

        return w.float()

    def _generate_bias(self):
        """ Generate random bias """
        b = torch.zeros(self.output_dim)
        return b.float()

    def _rescale_weight_bias(self, mat):
        """ Rescale weight or bias due to sparsity """
        # if the layer is sparse in input dimension, scale the weight
        if self.sparsity_mask is not None:
            scale = self.sparsity_mask.sum(axis=1).max() / self.input_dim
        else:
            scale = 1
        return mat / scale
    # ======================================================================================

    # FORWARD
    # ======================================================================================
    def enforce_constraints(self):
        """ Enforce mask """
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

    def forward(self, x):
        """ Forward Pass """
        return x.float() @ self.weight.T + self.bias
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
        # plot weight matrix
        weight = self.weight.cpu() if self.weight.device != torch.device('cpu') else self.weight
        if weight.size(0) < weight.size(1):
            utils.plot_connectivity_matrix_dist(weight.detach().numpy(), "Weight Matrix (Transposed)", False, self.sparsity_constraints)
        else:
            utils.plot_connectivity_matrix_dist(weight.detach().numpy().T, "Weight Matrix", False, self.sparsity_constraints)
        
    def print_layers(self):
        param_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dist": self.dist,
            "shape": self.weight.shape,
            "sparsity_constraints": self.sparsity_constraints,
            "learnable": self.weight.requires_grad,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "use_bias": self.bias.requires_grad,
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.weight.nonzero().size(0) / self.weight.numel(),
        }
        utils.print_dict("Linear Layer", param_dict)
    # ======================================================================================
