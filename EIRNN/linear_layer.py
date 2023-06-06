import torch
import numpy as np
import torch.nn as nn
import utils

class LinearLayer(nn.Module):
    def __init__(
            self,
            input_size,
            output_size,
            dist,
            use_bias,
            mask,
            use_dale,
            plasticity,
        ) -> None:
        """ 
        Sparse Linear Layer
        TODO: add range, add warning when using normal dist as negative values may exist
        Parameters:
            @param use_bias: whether to use bias
            @param mask: mask for sparse connectivity
            @param dist: distribution of weights
            @param input_size: input size
            @param output_size: output size
            @param use_dale: whether to use Dale's law
            @param plasticity: whether to use plasticity
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.dist = dist
        self.use_bias = use_bias
        self.mask = mask
        self.use_dale = use_dale
        self.plasticity = plasticity

        # initialize constraints
        self.sparse_mask, self.dale_mask = None, None
        if self.mask is None:
            assert not use_dale, "mask must be provided if use_dale is True"
            assert not plasticity, "mask must be provided if plasticity is True"
        else:
            if use_dale:
                assert plasticity, "use_dale cannot be True if plasticity is False"
                self.dale_mask = np.where(mask > 0, 1, -1)
            if plasticity:
                self.sparse_mask = np.where(mask == 0, 0, 1)

        # generate weights
        self.weight = torch.nn.Parameter(self.generate_weight())
        if self.use_bias: self.bias = torch.nn.Parameter(self.generate_bias())
        else: self.bias = torch.nn.Parameter(torch.zeros(self.output_size), requires_grad=False)
        
        # enfore constraints
        self.enforce_constraints()


    def generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            k = 1/self.input_size
            w = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.output_size, self.input_size))
            # w = np.random.uniform(-k, k, (self.output_size, self.input_size))
            w = np.abs(w)
        elif self.dist == 'normal':
            w = np.random.normal(0, 1/3, (self.output_size, self.input_size))

        if self.sparse_mask is not None:
            w = self.rescale_weight_bias(w)

        return torch.from_numpy(w).float()


    def rescale_weight_bias(self, w):
        """ Rescale weight or bias """
        # check if input size is fully connected except for all zeros
        no_need_to_scale = np.any([self.sparse_mask.sum(axis=1) == self.input_size, self.sparse_mask.sum(axis=1) == 0])
        if no_need_to_scale: scale = 1
        else:
            n_entries = self.sparse_mask.sum()
            n_total = self.output_size * self.input_size
            scale = (n_total / n_entries)
        return w * scale


    def generate_bias(self):
        """ Generate random bias """
        if self.dist == 'uniform':
            k = 1/self.input_size
            b = np.random.uniform(-np.sqrt(k), np.sqrt(k), (self.output_size))
            # b = np.abs(b)
        elif self.dist == 'normal':
            b = np.random.normal(0, 1/3, (self.output_size))
        
        if self.sparse_mask is not None:
            b = self.rescale_weight_bias(b)

        return torch.from_numpy(b).float()


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
        self.enforce_constraints()
        result = x.float() @ self.weight.T + self.bias
        return result
    

    def get_weight(self):
        """ Get weight """
        return self.weight.detach().numpy()
    

    def print_layer(self):
        param_dict = {
            "in_size": self.input_size,
            "out_size": self.output_size,
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
            utils.plot_connectivity_matrix(self.weight.detach().numpy(), "Weight Matrix", False)
        else:
            utils.plot_connectivity_matrix(self.weight.detach().numpy().T, "Weight Matrix", False)