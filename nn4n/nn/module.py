import nn4n
import torch
import numpy as np


class Module(torch.nn.Module):
    """
    An nn4n wrapper for torch.nn.Module

    TODO: temporarily remove plasticity_mask
    """
    def __init__(self, 
        sparsity_mask: torch.Tensor = None,
        positivity_mask: torch.Tensor = None,
        # plasticity_mask: torch.Tensor = None,
        **kwargs
    ):
        """
        Initialize the module

        Parameters:
            - sparsity_mask: sparsity mask
            - positivity_mask: positivity mask
        """
        super().__init__()

        # Initialize masks
        self.sparsity_mask = self._set_mask(sparsity_mask, "sparsity")
        self.positivity_mask = self._set_mask(positivity_mask, "positivity")
        # self.plasticity_mask = self._set_mask(plasticity_mask, "plasticity")

        # Initialize trainable parameters and enforce constraints
        self._init_trainable()
        self._enforce_sparsity()
        self._enforce_positivity()
        self._balance_excitatory_inhibitory()

        # Register the forward hook
        self.register_forward_pre_hook(self.enforce_constraints)

    # INIT MASKS
    # ======================================================================================
    def _set_mask(self, mask: torch.Tensor, mask_type: str):
        """
        Set the mask
        """
        mask = mask.T if mask is not None else None
        if mask is not None:
            self._check_mask(mask, mask_type)
            self.register_buffer(f"{mask_type}_mask", mask)
            return mask
        return None

    def _check_mask(self, mask: torch.Tensor, mask_type: str):
        """
        Check if the mask dimensions are valid
        """
        assert (
            mask.shape == self.weight.shape
        ), f"{mask_type} mask shape mismatch, expected {self.weight.shape}, got {mask.shape}"

    # INIT TRAINABLE
    # ======================================================================================
    def _init_trainable(self):
        # Convert weight and bias to learnable parameters
        self.weight = torch.nn.Parameter(self.weight)  # Not sure if not learnable would make sense, skip for now
        self.bias = torch.nn.Parameter(self.bias, requires_grad=self.bias_dist is not None)

    def _balance_excitatory_inhibitory(self):
        """Balance excitatory and inhibitory weights"""
        if self.positivity_mask is None:
            return  # No need to balance if no positivity mask
        scale_mat = torch.ones_like(self.weight)
        ext_sum = self.weight[self.positivity_mask == 1].sum()
        inh_sum = self.weight[self.positivity_mask == -1].sum()
        if ext_sum == 0 or inh_sum == 0:
            # Automatically stop balancing if one of the sums is 0
            # devide by 10 to avoid recurrent explosion/decay
            self.weight /= 10
        else:
            if ext_sum > abs(inh_sum):
                _scale = abs(inh_sum).item() / ext_sum.item()
                scale_mat[self.positivity_mask == 1] = _scale
            elif ext_sum < abs(inh_sum):
                _scale = ext_sum.item() / abs(inh_sum).item()
                scale_mat[self.positivity_mask == -1] = _scale
            # Apply scaling
            self.weight *= scale_mat

    def enforce_constraints(self, *args, **kwargs):
        """
        Enforce constraints
        """
        self._enforce_sparsity()
        self._enforce_positivity()

    # ENFORCE CONSTRAINTS
    # ======================================================================================
    def _enforce_sparsity(self):
        """Enforce sparsity"""
        if self.sparsity_mask is None:
            return
        w = self.weight.detach().clone()
        w = w * (self.sparsity_mask.T > 0).float()  # Ensure binary masking
        self.weight.data.copy_(w)

    def _enforce_positivity(self):
        """Enforce positivity"""
        if self.positivity_mask is None:
            return
        w = self.weight.detach().clone()
        w[self.positivity_mask.T == 1] = torch.clamp(w[self.positivity_mask.T == 1], min=0)
        w[self.positivity_mask.T == -1] = torch.clamp(w[self.positivity_mask.T == -1], max=0)
        self.weight.data.copy_(torch.nn.Parameter(w))

    # UTILITIES
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
            nn4n.utils.plot_connectivity_matrix(
                w=weight.detach().numpy(),
                title=f"Weight",
                colorbar=True,
            )
        elif plot_type == "dist":
            nn4n.utils.plot_connectivity_distribution(
                w=weight.detach().numpy(),
                title=f"Weight",
                ignore_zeros=self.sparsity_mask is not None,
            )