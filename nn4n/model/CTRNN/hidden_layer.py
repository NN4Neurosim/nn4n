import torch
import torch.nn as nn
import nn4n.utils as utils


class HiddenLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            dist,
            use_bias,
            scaling,
            mask,
            use_dale,
            new_synapses,
            self_connections,
            allow_negative,
            ei_balance,
            ) -> None:
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden units
            @param dist: distribution of hidden weights
            @param use_bias: use bias or not
            @param new_synapses: use new_synapses or not
            @param scaling: scaling of hidden weights
            @param mask: mask for hidden weights, used to enforce new_synapses and/or dale's law
            @param use_dale: use dale's law or not. If use_dale is True, mask must be provided
            @param self_connections: allow self connections or not
            @param allow_negative: allow negative weights or not, a boolean value
            @param ei_balance: method to balance e/i connections, based on number of neurons or number of synapses
        """
        super().__init__()
        # some params are for verbose printing
        self.hidden_size = hidden_size
        self.dist = dist
        self.use_bias = use_bias
        self.scaling = scaling
        self.use_dale = use_dale
        self.new_synapses = new_synapses
        self.self_connections = self_connections
        self.allow_negative = allow_negative
        self.ei_balance = ei_balance
        # generate weights and bias
        self.weight = self._generate_weight()
        self.bias = self._generate_bias()
        # init new_synapses, dale's law, and spectral radius
        self.dale_mask, self.sparse_mask = None, None
        self._init_constraints(mask)
        self._enforce_scaling()
        # parameterize the weights and bias
        self.weight = nn.Parameter(self.weight)
        self.bias = nn.Parameter(self.bias, requires_grad=self.use_bias)

    # INITIALIZATION
    # ======================================================================================
    def _generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            k = 1/self.hidden_size
            w = (torch.rand(self.hidden_size, self.hidden_size) * 2 - 1) * torch.sqrt(torch.tensor(k))
            if not self.allow_negative:
                w = (w + torch.abs(w.min())) / 2
        elif self.dist == 'normal':
            w = torch.randn(self.hidden_size, self.hidden_size) / torch.sqrt(torch.tensor(self.hidden_size))
            if not self.allow_negative:
                w = (w + torch.abs(w.min())) / 2
            # # Normalizing to [-1, 1] seems to be unnecessary, as it will be scaled later
            # # DISABLE ========================
            # w = (w - w.min()) / (w.max() - w.min()) * 2 - 1
            # if not self.allow_negative: w = (w + 1) / 2
        elif self.dist == 'zero':
            w = torch.zeros((self.output_dim, self.input_dim))
        else:
            raise NotImplementedError

        return w.float()

    def _generate_bias(self):
        # # Bias seem to have too much effect on the network
        # # DISABLE ========================
        # if self.use_bias:
        #     if self.dist == 'uniform':
        #         k = 1/self.hidden_size
        #         b = (torch.rand(self.hidden_size) * 2 - 1) * torch.sqrt(torch.tensor(k))
        #     elif self.dist == 'normal':
        #         b = torch.randn(self.hidden_size)
        #         # b = (b - b.min()) / (b.max() - b.min()) * 2 - 1
        # else:
        #     b = torch.zeros(self.hidden_size)
        # # ================================
        b = torch.zeros(self.hidden_size)

        return b.float()

    def _init_constraints(self, mask):
        """
        Initialize constraints
        It will also balance excitatory and inhibitory neurons
        """
        # Initialize dale's law and new_synapses masks
        if mask is None:
            assert not self.use_dale, "mask must be provided if use_dale is True"
            assert self.new_synapses, "mask must be provided if synapses are not plastic"
        else:
            mask = torch.from_numpy(mask).int()  # convert to torch.Tensor
            if self.use_dale:
                self._init_dale_mask(mask)  # initialize dale's mask
            if not self.new_synapses:
                self.sparse_mask = (mask != 0).int()  # convert to a binary mask
        # Whether to delete self connections
        if not self.self_connections:
            if self.sparse_mask is None:
                # if mask is not provided, create a mask
                self.sparse_mask = torch.ones((self.hidden_size, self.hidden_size))
            self.sparse_mask = torch.where(torch.eye(self.hidden_size) == 1, 0, self.sparse_mask)
        # Balance excitatory and inhibitory neurons and apply masks
        if self.dale_mask is not None:
            self.weight *= self.dale_mask
            self._balance_excitatory_inhibitory()
        if self.sparse_mask is not None:
            # TODO: re-write this part
            self.weight *= self.sparse_mask

    def _init_dale_mask(self, mask):
        """ initialize settings required for Dale's law """
        # Dale's law only applies to output edges
        # create a ei_list to store whether a neuron's output edges are all positive or all negative
        ei_list = torch.zeros(mask.shape[1])
        all_neg = torch.all(mask <= 0, dim=0)  # whether all output edges are negative
        all_pos = torch.all(mask >= 0, axis=0)  # whether all output edges are positive
        # check whether a neuron's output edges are all positive or all negative
        for i in range(mask.shape[1]):
            if all_neg[i]:
                ei_list[i] = -1
            elif all_pos[i]:
                ei_list[i] = 1
            else:
                assert False, "a neuron's output edges must be either all positive or all negative"
        self.ei_list = ei_list
        # create a mask for Dale's law
        self.dale_mask = torch.ones(mask.shape, dtype=int)
        self.dale_mask[:, self.ei_list == -1] = -1

    def _balance_excitatory_inhibitory(self):
        """ Balance excitatory and inhibitory weights """
        scale_mat = torch.ones_like(self.weight)
        # if using number of neurons to balance e/i connections
        if self.ei_balance == 'neuron':
            exc_pct = torch.count_nonzero(self.ei_list == 1.0) / self.hidden_size
            scale_mat[:, self.ei_list == 1] = 1 / exc_pct
            scale_mat[:, self.ei_list == -1] = 1 / (1 - exc_pct)
        # if using number of synapses to balance e/i connections
        elif self.ei_balance == 'synapse':
            exc_syn = torch.count_nonzero(self.weight > 0)
            inh_syn = torch.count_nonzero(self.weight < 0)
            exc_pct = exc_syn / (exc_syn + inh_syn)
            scale_mat[self.weight > 0] = 1 / exc_pct
            scale_mat[self.weight < 0] = 1 / (1 - exc_pct)
        # assert error if ei_balance is not 'neuron' or 'synapse'
        else:
            assert False, "ei_balance must be either 'neuron' or 'synapse'"
        # apply scaling
        self.weight *= scale_mat
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
        # scale = self.spec_rad / torch.abs(torch.linalg.eig(self.weight)[0]).max()
        # # Scale bias and weight
        # if self.use_bias:
        #     self.bias.data *= scale
        # self.weight.data *= scale

    def _enforce_scaling(self):
        """ Enforce scaling """
        # Calculate scale
        self.weight.data *= self.scaling

    def enforce_constraints(self):
        """ Enforce constraints """
        if self.sparse_mask is not None:
            self._enforce_sparsity()
        if self.dale_mask is not None:
            self._enforce_dale()

    def _enforce_sparsity(self):
        """ Enforce sparsity """
        # print('hidden sparse')
        # w = self.weight.detach()
        # w *= self.sparse_mask
        # w = torch.nn.Parameter(w)
        # self.weight.data.copy_(w)
        w = self.weight.detach().clone() * self.sparse_mask
        self.weight.data.copy_(torch.nn.Parameter(w))

    def _enforce_dale(self):
        """ Enforce Dale's law """
        # print('hidden dale')
        # w = self.weight.detach()
        # w[self.dale_mask == 1] = w[self.dale_mask == 1].clip(min=0)
        # w[self.dale_mask == -1] = w[self.dale_mask == -1].clip(max=0)
        # w = torch.nn.Parameter(w)
        # self.weight.data.copy_(w)
        w = self.weight.detach().clone()
        w[self.dale_mask == 1] = torch.clamp(w[self.dale_mask == 1], min=0)
        w[self.dale_mask == -1] = torch.clamp(w[self.dale_mask == -1], max=0)
        self.weight.data.copy_(torch.nn.Parameter(w))
    # ======================================================================================

    # HELPER FUNCTIONS
    # ======================================================================================
    def to(self, device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        if self.sparse_mask is not None:
            self.sparse_mask = self.sparse_mask.to(device)
        if self.dale_mask is not None:
            self.dale_mask = self.dale_mask.to(device)
        if not self.use_bias:
            self.bias = self.bias.to(device)

    def print_layer(self):
        # plot weight matrix
        param_dict = {
            "self_connections": self.self_connections,
            "input/output_dim": self.hidden_size,
            "distribution": self.dist,
            "bias": self.use_bias,
            "dale": self.use_dale,
            "shape": self.weight.shape,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "weight_mean": self.weight.mean().item(),
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.sparse_mask.sum() / self.sparse_mask.numel() if self.sparse_mask is not None else 1,
            "scaling": self.scaling,
            # "spectral_radius": torch.abs(torch.linalg.eig(self.weight)[0]).max().item(),
        }
        utils.print_dict("Hidden Layer", param_dict)
        weight = self.weight.cpu() if self.weight.device != torch.device('cpu') else self.weight
        utils.plot_connectivity_matrix_dist(weight.detach().numpy(), "Hidden Layer", False, not self.new_synapses)
        utils.plot_eigenvalues(weight.detach().numpy(), "Hidden Layer")
    # ======================================================================================
