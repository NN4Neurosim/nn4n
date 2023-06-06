import torch
import numpy as np
import torch.nn as nn
import utils

class HiddenLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            dist,
            use_bias,
            spec_rad,
            mask,
            use_dale,
            plasticity,
            self_connections,
            ) -> None:
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden units
            @param dist: distribution of hidden weights
            @param use_bias: use bias or not
            @param plasticity: use plasticity or not
            @param spec_rad: spectral radius of hidden weights
            @param mask: mask for hidden weights, used to enforce plasticity and/or dale's law
            @param use_dale: use dale's law or not. If use_dale is True, mask must be provided
        """
        super().__init__()
        # some params are for verbose printing
        self.hidden_size = hidden_size
        self.dist = dist
        self.use_bias = use_bias
        self.spec_rad = spec_rad
        self.use_dale = use_dale
        self.plasticity = plasticity
        self.self_connections = self_connections

        # initialize constraints
        self.dale_mask, self.sparse_mask = None, None
        if mask is None:
            assert not use_dale, "mask must be provided if use_dale is True"
            assert not plasticity, "mask must be provided if plasticity is True"
        else:
            if use_dale:
                assert plasticity, "use_dale cannot be True if plasticity is False"
                self.dale_mask = np.where(mask > 0, 1, -1)
            if plasticity:
                self.sparse_mask = np.where(mask == 0, 0, 1)
        # whether to delete self connections
        if not self.self_connections: 
            if self.sparse_mask is None: 
                # if mask is not provided, create a mask
                self.sparse_mask = np.ones_like((self.hidden_size, self.hidden_size))
            self.sparse_mask = np.where(np.eye(self.hidden_size) == 1, 0, self.sparse_mask)
            

        # generate weights and bias
        self.weight = torch.nn.Parameter(self.generate_weight())
        if self.use_bias: self.bias = torch.nn.Parameter(self.generate_bias())
        else: self.bias = torch.nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        
        # enforce plasticity, dale's law, and spectral radius
        self.enforce_constraints()
        if self.dist == 'normal': self.enforce_spec_rad()


    def generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            k = 1/self.hidden_size
            w = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.hidden_size, self.hidden_size))
        elif self.dist == 'normal':
            w = np.random.normal(0, 1/3, (self.hidden_size, self.hidden_size))

        if self.dale:
            w = self.balance_excitatory_inhibitory(w)

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
        n_exc = np.sum(self.dale_mask == 1)
        n_inh = np.sum(self.dale_mask == -1)
        n_total = self.hidden_size**2/4

        dale_mask = self.dale_mask.copy()
        dale_mask[dale_mask == 1] = (n_total / n_exc)
        dale_mask[dale_mask == -1] = -(n_total / n_inh)

        return np.abs(w) * dale_mask


    def enforce_spec_rad(self):
        """ Enforce spectral radius """
        w = self.weight.detach().numpy()
        scale = self.spec_rad / np.max(np.abs(np.linalg.eigvals(w)))
        w *= scale
        w = torch.nn.Parameter(torch.from_numpy(w))
        self.weight.data.copy_(w)

        if self.use_bias:
            b = self.bias.detach().numpy()
            b *= scale
            b = torch.nn.Parameter(torch.from_numpy(b))
            self.bias.data.copy_(b)


    def enforce_constraints(self):
        """ Enforce constraints """
        if self.sparse_mask is not None:
            self.enforce_plasticity()
        if self.dale_mask is not None:
            self.enforce_dale()


    def enforce_dale(self):
        """ Enforce dale """
        w = self.weight.detach().numpy()
        w[self.dale_mask == 1] = w[self.dale_mask == 1].clip(min=0)
        w[self.dale_mask == -1] = w[self.dale_mask == -1].clip(max=0)
        w = torch.nn.Parameter(torch.from_numpy(w))
        self.weight.data.copy_(w)


    def enforce_plasticity(self):
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
        utils.plot_connectivity_matrix(self.weight.detach().numpy(), "Hidden Layer")
