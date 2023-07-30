import torch
import torch.nn as nn
import nn4n.utils as utils


class LinearLayer(nn.Module):
    def __init__(
            self,
            input_dim,
            output_dim,
            dist,
            use_bias,
            mask,
            use_dale,
            new_synapses,
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
            @param new_synapses: whether to use new_synapses
            @param allow_negative: whether to allow negative weights, a boolean value
            @param ei_balance: method to balance E/I neurons, default: "neuron"
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dist = dist
        self.use_bias = use_bias
        self.use_dale = use_dale
        self.new_synapses = new_synapses
        self.allow_negative = allow_negative
        self.ei_balance = ei_balance

        # generate weights
        self.weight = self._generate_weight()
        self.bias = self._generate_bias()

        # enfore constraints
        self.dale_mask, self.sparse_mask = None, None
        self._init_constraints(mask)

        # convert weight and bias to torch tensor
        self.weight = nn.Parameter(self.weight)
        self.bias = nn.Parameter(self.bias, requires_grad=self.use_bias)

    # INITIALIZATION
    # ======================================================================================
    def _init_constraints(self, mask):
        """
        Initialize constraints (because _enforce_dale will clip the weights, but we don't want that during initialization)
        It will also balance excitatory and inhibitory neurons
        """
        if mask is None:
            assert not self.use_dale, "mask must be provided if use_dale is True"
            assert self.new_synapses, "mask must be provided if synapses are not plastic"
        else:
            mask = torch.tensor(mask)
            if self.use_dale:
                self._init_dale_mask(mask)
            if not self.new_synapses:
                # TODO: re-write this part
                self.sparse_mask = (mask != 0).float()

        if self.dale_mask is not None:
            self.weight *= self.dale_mask
            # self._balance_excitatory_inhibitory()
        if self.sparse_mask is not None:
            self.weight *= self.sparse_mask
            self.weight = self._rescale_weight_bias(self.weight)
            # self.bias = self._rescale_weight_bias(self.bias)

    def _balance_excitatory_inhibitory(self):
        """ Balance excitatory and inhibitory weights """
        scale_mat = torch.ones_like(self.weight)
        # if using number of neurons to balance e/i connections
        if self.ei_balance == 'neuron':
            exc_pct = torch.count_nonzero(self.ei_list == 1.0) / self.ei_list.shape[0]
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

        self.weight *= scale_mat

    def _init_dale_mask(self, mask):
        """ initialize settings required for Dale's law """
        # Dale's law only applies to output edges
        # create a ei_list to store whether a neuron's output edges are all positive or all negative
        ei_list = torch.zeros(mask.shape[1])
        all_neg = torch.all(mask <= 0, axis=0)
        all_pos = torch.all(mask >= 0, axis=0)
        all_zero = torch.all(mask == 0, axis=0)  # readout layer may be sparse

        # check if all neurons are either all negative or all positive
        for i in range(mask.shape[1]):
            # if this neuron has all negative or all positive output edges, set it to -1 or 1
            if all_neg[i]:
                ei_list[i] = -1
            elif all_pos[i]:
                ei_list[i] = 1
            else:
                assert False, "a neuron's output edges must be either all positive or all negative"

            # if this neuron has no output edges, set it to 0 too
            if all_zero[i]:
                ei_list[i] = 0
        self.ei_list = ei_list

        # create a mask for Dale's law
        self.dale_mask = torch.ones(mask.shape, dtype=int)
        self.dale_mask[:, self.ei_list == -1] = -1

    def _generate_weight(self):
        """ Generate random weight """
        if self.dist == 'uniform':
            # if uniform, let w be uniform in [-sqrt(k), sqrt(k)]
            sqrt_k = torch.sqrt(torch.tensor(1/self.input_dim))
            if self.allow_negative:
                w = torch.rand(self.output_dim, self.input_dim) * 2 * sqrt_k - sqrt_k
            else:
                w = torch.rand(self.output_dim, self.input_dim) * sqrt_k
        elif self.dist == 'normal':
            w = torch.randn(self.output_dim, self.input_dim) / torch.sqrt(torch.tensor(self.input_dim))
            if not self.allow_negative:
                w = (w + torch.abs(w.min())) / 2
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
        if self.sparse_mask is not None:
            scale = self.sparse_mask.sum(axis=1).max() / self.input_dim
        else:
            scale = 1
        return mat / scale
    # ======================================================================================

    # FORWARD
    # ======================================================================================
    def enforce_constraints(self):
        """ Enforce mask """
        if self.sparse_mask is not None:
            self._enforce_sparsity()
        if self.dale_mask is not None:
            self._enforce_dale()

    def _enforce_sparsity(self):
        """ Enforce sparsity """
        # print('linear sparse')
        # w = self.weight.detach()
        # w *= self.sparse_mask
        # w = torch.nn.Parameter(w)
        # self.weight.data.copy_(w)
        w = self.weight.detach().clone() * self.sparse_mask
        self.weight.data.copy_(torch.nn.Parameter(w))

    def _enforce_dale(self):
        """ Enforce Dale's law """
        # print('linear dale')
        # w = self.weight.detach()
        # w[self.dale_mask == 1] = w[self.dale_mask == 1].clip(min=0)
        # w[self.dale_mask == -1] = w[self.dale_mask == -1].clip(max=0)
        # w = torch.nn.Parameter(w)
        # self.weight.data.copy_(w)
        w = self.weight.detach().clone()
        w[self.dale_mask == 1] = torch.clamp(w[self.dale_mask == 1], min=0)
        w[self.dale_mask == -1] = torch.clamp(w[self.dale_mask == -1], max=0)
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
        if self.sparse_mask is not None:
            self.sparse_mask = self.sparse_mask.to(device)
        if self.dale_mask is not None:
            self.dale_mask = self.dale_mask.to(device)
        if not self.use_bias:
            self.bias = self.bias.to(device)

    def print_layer(self):
        param_dict = {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "dist": self.dist,
            "bias": self.use_bias,
            "shape": self.weight.shape,
            "weight_min": self.weight.min().item(),
            "weight_max": self.weight.max().item(),
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
            "sparsity": self.sparse_mask.sum() / self.sparse_mask.numel() if self.sparse_mask is not None else 1,
        }
        utils.print_dict("Linear Layer", param_dict)

        # plot weight matrix
        weight = self.weight.cpu() if self.weight.device != torch.device('cpu') else self.weight
        if weight.size(0) < weight.size(1):
            utils.plot_connectivity_matrix_dist(weight.detach().numpy(), "Weight Matrix (Transposed)", False, not self.new_synapses)
        else:
            utils.plot_connectivity_matrix_dist(weight.detach().numpy().T, "Weight Matrix", False, not self.new_synapses)

    # def get_weight(self):
    #     """ Get weight """
    #     return self.weight.detach().numpy()
    # ======================================================================================
