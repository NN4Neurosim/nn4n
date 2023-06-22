import torch
import numpy as np
import torch.nn as nn
import nn4n.utils as utils
import matplotlib.pyplot as plt

class LinearLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            dist,
            use_bias,
            mask,
            use_dale,
            new_synapse,
            allow_negative,
            ei_balance,
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
            @param use_dale: whether to use Dale's law
            @param new_synapse: whether to use new_synapse
            @param allow_negative: whether to allow negative weights, a boolean value
            @param ei_balance: method to balance E/I neurons, default: "neuron"
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dist = dist
        self.use_bias = use_bias
        self.mask = mask
        self.use_dale = use_dale
        self.new_synapse = new_synapse
        self.allow_negative = allow_negative
        self.ei_balance = ei_balance


        # initialize constraints
        self.sparse_mask, self.dale_mask = None, None
        if self.mask is None:
            assert not self.use_dale, "mask must be provided if use_dale is True"
            assert self.new_synapse, "mask must be provided if synapses are not plastic"
        else:
            if self.use_dale:
                self.init_dale_mask(mask)
            if not self.new_synapse:
                self.sparse_mask = np.where(mask == 0, 0, 1)

        # generate weights
        self.weight = self.generate_weight()
        self.bias = self.generate_bias()
        
        # enfore constraints
        self.init_constraints()

        # convert weight and bias to torch tensor
        self.weight = nn.Parameter(torch.from_numpy(self.weight).float())
        self.bias = nn.Parameter(torch.from_numpy(self.bias).float(), requires_grad=self.use_bias)


    # INITIALIZATION
    # ======================================================================================
    def init_constraints(self):
        """
        Initialize constraints (because enforce_dale will clip the weights, but we don't want that during initialization)
        It will also balance excitatory and inhibitory neurons
        """
        if self.dale_mask is not None:
            self.weight *= self.dale_mask
            self.balance_excitatory_inhibitory()
        if self.sparse_mask is not None:
            self.weight *= self.sparse_mask


    def balance_excitatory_inhibitory(self):
        """ Balance excitatory and inhibitory weights """
        scale_mat = np.ones_like(self.weight)
        if self.ei_balance == 'neuron':
            exc_pct = np.count_nonzero(self.ei_list == 1.0) / self.hidden_size
            if exc_pct == 0 or exc_pct == 1: return # avoid division by zero
            scale_mat[:, self.ei_list == 1] = 1 / exc_pct
            scale_mat[:, self.ei_list == -1] = 1 / (1 - exc_pct)
        elif self.ei_balance == 'synapse':
            exc_syn = np.count_nonzero(self.weight > 0)
            inh_syn = np.count_nonzero(self.weight < 0)
            exc_pct = exc_syn / (exc_syn + inh_syn)
            if exc_pct == 0 or exc_pct == 1: return # avoid division by zero
            scale_mat[self.weight > 0] = 1 / exc_pct
            scale_mat[self.weight < 0] = 1 / (1 - exc_pct)
        else:
            assert False, "ei_balance must be either 'neuron' or 'synapse'"

        self.weight *= scale_mat


    def init_dale_mask(self, mask):
        """ initialize settings required for Dale's law """
        # Dale's law only applies to output edges
        # create a ei_list to store whether a neuron's output edges are all positive or all negative
        ei_list = np.zeros(mask.shape[1])
        all_neg = np.all(mask <= 0, axis=0)
        all_pos = np.all(mask >= 0, axis=0)
        all_zero = np.all(mask == 0, axis=0) # readout layer may be sparse

        # check if all neurons are either all negative or all positive
        for i in range(mask.shape[1]):
            if all_neg[i]: ei_list[i] = -1
            elif all_pos[i]: ei_list[i] = 1
            else: assert False, "a neuron's output edges must be either all positive or all negative"

            if all_zero[i]: ei_list[i] = 0
        self.ei_list = ei_list
        
        # generate mask for Dale's law
        dale_mask = np.ones(mask.shape)
        dale_mask[:, self.ei_list == -1] = -1
        self.dale_mask = dale_mask


    def generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            k = 1/self.input_dim
            w = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.output_dim, self.input_dim))
            if not self.allow_negative: w = np.abs(w)
        elif self.dist == 'normal':
            w = np.random.normal(0, 1/3, (self.output_dim, self.input_dim))

        if self.sparse_mask is not None:
            w = self.rescale_weight_bias(w)

        return w


    def generate_bias(self):
        """ Generate random bias """
        if self.use_bias:
            if self.dist == 'uniform':
                k = 1/self.input_dim
                b = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.output_dim))
            elif self.dist == 'normal':
                b = np.random.normal(0, 1/3, (self.output_dim))
            
            if self.sparse_mask is not None:
                b = self.rescale_weight_bias(b)
        else:
            b = np.zeros(self.output_dim)

        return b
    

    def rescale_weight_bias(self, w):
        """ Rescale weight or bias """
        # check if input size is fully connected except for all zeros
        no_need_to_scale = np.any([self.sparse_mask.sum(axis=1) == self.input_dim, self.sparse_mask.sum(axis=1) == 0])
        if no_need_to_scale: scale = 1
        else:
            n_entries = self.sparse_mask.sum()
            n_total = self.output_dim * self.input_dim
            scale = (n_total / n_entries)
        return w * scale
    # ======================================================================================


    # FORWARD
    # ======================================================================================
    def enforce_constraints(self):
        """ Enforce mask """
        if self.sparse_mask is not None:
            self.enforce_sparsity()
        if self.use_dale:
            self.enforce_dale()


    def enforce_sparsity(self):
        """ Enforce sparsity """
        w = self.weight.detach().numpy()
        w *= self.sparse_mask
        w = torch.nn.Parameter(torch.from_numpy(w))
        self.weight.data.copy_(w)


    def enforce_dale(self):
        """ Enforce Dale's law """
        w = self.weight.detach().numpy()
        w[self.dale_mask == 1] = w[self.dale_mask == 1].clip(min=0)
        w[self.dale_mask == -1] = w[self.dale_mask == -1].clip(max=0)
        w = torch.nn.Parameter(torch.from_numpy(w))
        self.weight.data.copy_(w)


    def forward(self, x):
        """ Forward Pass """
        result = x.float() @ self.weight.T + self.bias
        return result
    # ======================================================================================
    

    # HELPER FUNCTIONS
    # ======================================================================================
    def print_layer(self):
        param_dict = {
            "in_size": self.input_dim,
            "out_size": self.output_dim,
            "dist": self.dist,
            "bias": self.use_bias,
            "shape": self.weight.shape,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.sparse_mask.sum() / self.sparse_mask.size if self.sparse_mask is not None else "None"
        }
        utils.print_dict("Linear Layer", param_dict)
        if self.weight.size(0) < self.weight.size(1):
            utils.plot_connectivity_matrix_dist(self.weight.detach().numpy(), "Weight Matrix", False, not self.new_synapse)
        else:
            utils.plot_connectivity_matrix_dist(self.weight.detach().numpy().T, "Weight Matrix", False, not self.new_synapse)
    

    # def get_weight(self):
    #     """ Get weight """
    #     return self.weight.detach().numpy()
    # ======================================================================================