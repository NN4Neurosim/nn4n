import torch
import numpy as np
import nn4n.utils as utils


class LinearLayer(torch.nn.Module):
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
        ei_mask: torch.Tensor = None,
        sparsity_mask: torch.Tensor = None,
        plasticity_mask: torch.Tensor = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight_dist = weight
        self.bias_dist = bias
        self.weight = self._generate_weight(self.weight_dist)
        self.bias = self._generate_bias(self.bias_dist)
        self.ei_mask = ei_mask.T if ei_mask is not None else None
        self.sparsity_mask = sparsity_mask.T if sparsity_mask is not None else None
        self.plasticity_mask = (
            plasticity_mask.T if plasticity_mask is not None else None
        )
        # All unique plasticity values in the plasticity mask
        self.plasticity_scales = (
            torch.unique(self.plasticity_mask)
            if self.plasticity_mask is not None
            else None
        )
        self._init_trainable()

    # INITIALIZATION
    # ======================================================================================
    @classmethod
    def from_dict(cls, layer_struct):
        """
        Alternative constructor to initialize LinearLayer from a dictionary.
        """
        # Create an instance using the dictionary values
        cls._check_keys(layer_struct)
        return cls(
            input_dim=layer_struct["input_dim"],
            output_dim=layer_struct["output_dim"],
            weight=layer_struct.get("weight", "uniform"),
            bias=layer_struct.get("bias", "uniform"),
            ei_mask=layer_struct.get("ei_mask"),
            sparsity_mask=layer_struct.get("sparsity_mask"),
            plasticity_mask=layer_struct.get("plasticity_mask"),
        )

    @staticmethod
    def _check_keys(layer_struct):
        required_keys = ["input_dim", "output_dim"]
        for key in required_keys:
            if key not in layer_struct:
                raise ValueError(f"Key '{key}' is missing in layer_struct")
        
        valid_keys = ["input_dim", "output_dim", "weight", "bias", "ei_mask", "sparsity_mask", "plasticity_mask"]
        for key in layer_struct.keys():
            if key not in valid_keys:
                raise ValueError(f"Key '{key}' is not a valid key in layer_struct")

    def _check_constaint_dims(self):
        """
        Check if the mask dimensions are valid
        """
        if self.sparsity_mask is not None:
            assert (
                self.sparsity_mask.shape == (self.output_dim, self.input_dim)
            ), f"Sparsity mask shape mismatch, expected {(self.output_dim, self.input_dim)}, got {self.sparsity_mask.shape}"
            self.sparsity_mask = self.sparsity_mask.clone().detach().int()
        if self.ei_mask is not None:
            assert (
                self.ei_mask.shape == (self.output_dim, self.input_dim)
            ), f"Excitatory/Inhibitory mask shape mismatch, expected {(self.output_dim, self.input_dim)}, got {self.ei_mask.shape}"
            self.ei_mask = self.ei_mask.clone().detach().float()
        if self.plasticity_mask is not None:
            assert (
                self.plasticity_mask.shape == (self.output_dim, self.input_dim)
            ), f"Plasticity mask shape mismatch, expected {(self.output_dim, self.input_dim)}, got {self.plasticity_mask.shape}"
            self.plasticity_mask = self.plasticity_mask.clone().detach().float()

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

    # INIT TRAINABLE
    # ======================================================================================
    def _init_trainable(self):
        # Enfore constraints
        self._init_constraints()
        # Convert weight and bias to learnable parameters
        self.weight = torch.nn.Parameter(
            self.weight, requires_grad=self.weight_dist is not None
        )
        self.bias = torch.nn.Parameter(self.bias, requires_grad=self.bias_dist is not None)

    def _init_constraints(self):
        """
        Initialize constraints
        It will also balance excitatory and inhibitory neurons
        """
        self._check_constaint_dims()
        if self.sparsity_mask is not None:
            self._enforce_sparsity()
        if self.ei_mask is not None:
            # Apply Dale's law
            self.weight[self.ei_mask == 1] = torch.clamp(
                self.weight[self.ei_mask == 1], min=0
            ) # For excitatory neurons, set negative weights to 0
            self.weight[self.ei_mask == -1] = torch.clamp(
                self.weight[self.ei_mask == -1], max=0
            ) # For inhibitory neurons, set positive weights to 0

            # Balance excitatory and inhibitory neurons weight magnitudes
            self._balance_excitatory_inhibitory()

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

    def _balance_excitatory_inhibitory(self):
        """Balance excitatory and inhibitory weights"""
        scale_mat = torch.ones_like(self.weight)
        ext_sum = self.weight[self.ei_mask == 1].sum()
        inh_sum = self.weight[self.ei_mask == -1].sum()
        if ext_sum == 0 or inh_sum == 0:
            # Automatically stop balancing if one of the sums is 0
            # devide by 10 to avoid recurrent explosion/decay
            self.weight /= 10
        else:
            if ext_sum > abs(inh_sum):
                _scale = abs(inh_sum).item() / ext_sum.item()
                scale_mat[self.ei_mask == 1] = _scale
            elif ext_sum < abs(inh_sum):
                _scale = ext_sum.item() / abs(inh_sum).item()
                scale_mat[self.ei_mask == -1] = _scale
            # Apply scaling
            self.weight *= scale_mat

    # TRAINING
    # ======================================================================================
    def to(self, device):
        """Move the network to the device (cpu/gpu)"""
        super().to(device)
        if self.sparsity_mask is not None:
            self.sparsity_mask = self.sparsity_mask.to(device)
        if self.ei_mask is not None:
            self.ei_mask = self.ei_mask.to(device)
        if self.bias.requires_grad:
            self.bias = self.bias.to(device)
        return self

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
                    raise RuntimeError(
                        "Weight gradient is None, possibly because the forward loop is non-differentiable"
                    )

    def freeze(self):
        """Freeze the layer"""
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def unfreeze(self):
        """Unfreeze the layer"""
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    # CONSTRAINTS
    # ======================================================================================
    def enforce_constraints(self):
        """
        Enforce constraints

        The constraints are:
            - sparsity_mask: mask for sparse connectivity
            - ei_mask: mask for Dale's law
        """
        if self.sparsity_mask is not None:
            self._enforce_sparsity()
        if self.ei_mask is not None:
            self._enforce_ei()

    def _enforce_sparsity(self):
        """Enforce sparsity"""
        if self.sparsity_mask is not None:
            # Apply mask directly without scaling
            w = self.weight.detach().clone()
            w = w * (self.sparsity_mask > 0).float()  # Ensure binary masking
            self.weight.data.copy_(w)

    def _enforce_ei(self):
        """Enforce Dale's law"""
        w = self.weight.detach().clone()
        w[self.ei_mask == 1] = torch.clamp(w[self.ei_mask == 1], min=0)
        w[self.ei_mask == -1] = torch.clamp(w[self.ei_mask == -1], max=0)
        self.weight.data.copy_(torch.nn.Parameter(w))

    # HELPER FUNCTIONS
    # ======================================================================================
    def set_weight(self, weight):
        """Set the value of weight"""
        assert (
            weight.shape == self.weight.shape
        ), f"Weight shape mismatch, expected {self.weight.shape}, got {weight.shape}"
        with torch.no_grad():
            self.weight.copy_(weight)

    def plot_layer(self, plot_type="weight"):
        """Plot the weights matrix and distribution of each layer"""
        weight = (
            self.weight.cpu()
            if self.weight.device != torch.device("cpu")
            else self.weight
        )
        if plot_type == "weight":
            utils.plot_connectivity_matrix(
                w=weight.detach().numpy(),
                title=f"Weight",
                colorbar=True,
            )
        elif plot_type == "dist":
            utils.plot_connectivity_distribution(
                w=weight.detach().numpy(),
                title=f"Weight",
                ignore_zeros=self.sparsity_mask is not None,
            )

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