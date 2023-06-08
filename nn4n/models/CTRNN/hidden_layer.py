import torch
import numpy as np
import torch.nn as nn
import nn4n.utils as utils

class HiddenLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            dist,
            use_bias,
            spec_rad,
            mask,
            use_dale,
            new_synapse,
            self_connections,
            allow_neg,
            ei_balance,
            ) -> None:
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden units
            @param dist: distribution of hidden weights
            @param use_bias: use bias or not
            @param new_synapse: use new_synapse or not
            @param spec_rad: spectral radius of hidden weights
            @param mask: mask for hidden weights, used to enforce new_synapse and/or dale's law
            @param use_dale: use dale's law or not. If use_dale is True, mask must be provided
            @param self_connections: allow self connections or not
            @param allow_neg: allow negative weights or not, a boolean value
            @param ei_balance: method to balance e/i connections, based on number of neurons or number of synapses
        """
        super().__init__()
        # some params are for verbose printing
        self.hidden_size = hidden_size
        self.dist = dist
        self.use_bias = use_bias
        self.spec_rad = spec_rad
        self.use_dale = use_dale
        self.new_synapse = new_synapse
        self.self_connections = self_connections
        self.allow_neg = allow_neg
        self.ei_balance = ei_balance

        # initialize constraints
        self.dale_mask, self.sparse_mask = None, None
        if mask is None:
            assert not self.use_dale, "mask must be provided if use_dale is True"
            assert self.new_synapse, "mask must be provided if synapses are not plastic"
        else:
            if self.use_dale:
                self._init_ei_neurons(mask)
            if not self.new_synapse:
                self.sparse_mask = np.where(mask == 0, 0, 1)
        # whether to delete self connections
        if not self.self_connections: 
            if self.sparse_mask is None: 
                # if mask is not provided, create a mask
                self.sparse_mask = np.ones((self.hidden_size, self.hidden_size))
            self.sparse_mask = np.where(np.eye(self.hidden_size) == 1, 0, self.sparse_mask)
            
        # generate weights and bias
        self.weight = self.generate_weight()
        

        # NOTE: weight are np.ndarray before that, but now they are torch.Tensor
        # init new_synapse, dale's law, and spectral radius
        self.init_constraints()
        self.enforce_spec_rad()

        # parameterize the weights and bias
        self.weight = nn.Parameter(self.weight.float())
        if self.use_bias: self.bias = torch.nn.Parameter(self.generate_bias())
        else: self.bias = torch.nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)



    def _init_ei_neurons(self, mask):
        """ initialize settings required for Dale's law """
        # Dale's law only applies to output edges
        # create a ei_list to store whether a neuron's output edges are all positive or all negative
        ei_list = np.zeros(mask.shape[1])
        all_neg = np.all(mask <= 0, axis=0) # whether all output edges are negative
        all_pos = np.all(mask >= 0, axis=0) # whether all output edges are positive

        # check whether a neuron's output edges are all positive or all negative
        for i in range(mask.shape[1]):
            if all_neg[i]: ei_list[i] = -1
            elif all_pos[i]: ei_list[i] = 1
            else: assert False, "a neuron's output edges must be either all positive or all negative"
        self.ei_list = ei_list

        # create a mask for Dale's law
        dale_mask = np.ones(mask.shape)
        dale_mask[:, self.ei_list == -1] = -1
        self.dale_mask = dale_mask.astype(int)


    def generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            k = 1/self.hidden_size
            w = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.hidden_size, self.hidden_size))
            if not self.allow_neg: w = np.abs(w)
        elif self.dist == 'normal':
            if self.allow_neg: 
                w = np.random.normal(0, 1/3, (self.hidden_size, self.hidden_size))
            else:
                w = np.random.normal(0, 1/3, (self.hidden_size, self.hidden_size)) / 2 + 0.5

        return torch.from_numpy(w).float()
    

    def generate_bias(self):
        if self.dist == 'uniform':
            k = 1/self.hidden_size
            b = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.hidden_size))
        elif self.dist == 'normal':
            b = np.random.normal(0, 1, (self.hidden_size))

        return torch.from_numpy(b).float()


    def forward(self, x):
        """ Forward """
        self.enforce_constraints()
        return x.float() @ self.weight.T + self.bias


    def balance_excitatory_inhibitory(self, w):
        """ Balance excitatory and inhibitory weights """
        scale_mat = np.ones((self.hidden_size, self.hidden_size))
        if self.ei_balance == 'neuron':
            exc_pct = np.count_nonzero(self.ei_list == 1.0) / self.hidden_size
            scale_mat[:, self.ei_list == 1] = 1 / exc_pct
            scale_mat[:, self.ei_list == -1] = 1 / (1 - exc_pct)
        elif self.ei_balance == 'synapse':
            exc_syn = np.count_nonzero(w > 0)
            inh_syn = np.count_nonzero(w < 0)
            exc_pct = exc_syn / (exc_syn + inh_syn)
            scale_mat[w > 0] = 1 / exc_pct
            scale_mat[w < 0] = 1 / (1 - exc_pct)
        else:
            assert False, "ei_balance must be either 'neuron' or 'synapse'"

        return w * scale_mat


    def enforce_spec_rad(self):
        """ Enforce spectral radius """
        scale = self.spec_rad / np.max(np.abs(np.linalg.eigvals(self.weight)))
        self.weight *= scale


    def init_constraints(self):
        """
        Initialize constraints (because enforce_dale will clip the weights, but we don't want that during initialization)
        It will also balance excitatory and inhibitory neurons
        """
        if self.dale_mask is not None:
            self.weight *= self.dale_mask
            self.weight = self.balance_excitatory_inhibitory(self.weight)
        if self.sparse_mask is not None:
            self.weight *= self.sparse_mask


    def enforce_constraints(self):
        """ Enforce constraints """
        if self.sparse_mask is not None:
            self.enforce_sparsity()
        if self.dale_mask is not None:
            self.enforce_dale()


    def enforce_dale(self):
        """ Enforce dale """
        w = self.weight.detach().numpy()
        w[self.dale_mask == 1] = w[self.dale_mask == 1].clip(min=0)
        w[self.dale_mask == -1] = w[self.dale_mask == -1].clip(max=0)
        w = torch.nn.Parameter(torch.from_numpy(w))
        self.weight.data.copy_(w)


    def enforce_sparsity(self):
        """ Enforce sparsity """
        w = self.weight.detach().numpy()
        w *= self.sparse_mask
        w = torch.nn.Parameter(torch.from_numpy(w))
        self.weight.data.copy_(w)


    def print_layer(self):
        param_dict = {
            "self_connections": self.self_connections,
            "spec_rad": self.spec_rad,
            "in_size": self.hidden_size,
            "out_size": self.hidden_size,
            "distribution": self.dist,
            "bias": self.use_bias,
            "dale": self.use_dale,
            "shape": self.weight.shape,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "weight_mean": self.weight.mean().item(),
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.sparse_mask.sum() / self.sparse_mask.size if self.sparse_mask is not None else "None",
            "spectral_radius": np.max(np.abs(np.linalg.eigvals(self.weight.detach().numpy()))),
        }
        utils.print_dict("Hidden Layer", param_dict)
        utils.plot_connectivity_matrix_dist(self.weight.detach().numpy(), "Hidden Layer", False, not self.new_synapse)
