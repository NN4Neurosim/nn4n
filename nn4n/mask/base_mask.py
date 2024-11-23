import numpy as np
import nn4n.utils as utils
from nn4n.utils.help_functions import print_dict


class BaseMask:
    """Base class for all masks"""

    def __init__(self, **kwargs):
        """
        @kwarg hidden_size: number of hidden neurons in total, must be defined
        @kwarg input_dim: input dimension, default: 1
        @kwarg readout_dim: readout dimension, default: 1
        """
        self.dims = kwargs.get("dims", [1, 100, 1])
        self.hidden_size = self.dims[1]
        self.input_dim = self.dims[0]
        self.readout_dim = self.dims[2]

        # cannot be run as a child class
        assert (
            self.__class__.__name__ != "BaseMask"
        ), "BaseMask cannot be run as a child class"

    def _check_parameters(self):
        """
        Check if parameters are valid
        """
        # check hidden_size
        assert self.hidden_size is not None, "hidden_size must be defined"
        assert isinstance(self.hidden_size, int), "hidden_size must be int"
        assert self.hidden_size > 0, "hidden_size must be positive"

        # check input_dim
        assert self.input_dim is not None, "input_dim must be defined"
        assert isinstance(self.input_dim, int), "input_dim must be int"
        assert self.input_dim > 0, "input_dim must be positive"

        # check readout_dim
        assert self.readout_dim is not None, "readout_dim must be defined"
        assert isinstance(self.readout_dim, int), "readout_dim must be int"
        assert self.readout_dim > 0, "readout_dim must be positive"

    def _generate_masks(self):
        """
        Generate the mask for the multi-area network
        """
        self._generate_hidden_mask()
        self._generate_input_mask()
        self._generate_readout_mask()

    def _generate_hidden_mask(self):
        """
        Generate the mask for the hidden layer
        """
        raise NotImplementedError

    def _generate_input_mask(self):
        """
        Generate the mask for the input layer
        """
        raise NotImplementedError

    def _generate_readout_mask(self):
        """
        Generate the mask for the readout layer
        """
        raise NotImplementedError

    def get_input_indices(self):
        """
        Return the indices of neurons that receive input
        """
        raise NotImplementedError

    def get_non_input_indices(self):
        """
        Return the indices of neurons that do not receive input
        """
        raise NotImplementedError

    def get_readout_indices(self):
        """Return the indices of neurons that send readout"""
        raise NotImplementedError

    def get_areas(self):
        """Return the number of areas"""
        raise NotImplementedError

    def get_area_indices(self, area):
        """Return the indices of neurons in area"""
        raise NotImplementedError

    def _generate_sparse_matrix(self, n, m, p):
        """
        Generate a sparse matrix with size n x m and density p. 1 if connection exists, 0 otherwise
        """
        assert 0 <= p <= 1, "p must be between 0 and 1"
        mask = np.random.rand(n, m) < p
        return mask.astype(int)

    def plot_masks(self):
        if self.input_mask is not None:
            input_mask_ = (
                self.input_mask
                if self.input_mask.shape[1] > self.input_mask.shape[0]
                else self.input_mask.T
            )
            utils.plot_connectivity_matrix(input_mask_, "Input Layer Mask", False)

        if self.hidden_mask is not None:
            utils.plot_connectivity_matrix(self.hidden_mask, "Hidden Layer Mask", False)

        if self.readout_mask is not None:
            readout_mask_ = (
                self.readout_mask
                if self.readout_mask.shape[1] > self.readout_mask.shape[0]
                else self.readout_mask.T
            )
            utils.plot_connectivity_matrix(readout_mask_, "Readout Layer Mask", False)

    def get_masks(self):
        """Return the masks"""
        assert self.input_mask is not None, "input_mask is not generated"
        assert self.hidden_mask is not None, "hidden_mask is not generated"
        assert self.readout_mask is not None, "readout_mask is not generated"
        return [self.input_mask.T, self.hidden_mask.T, self.readout_mask.T]

    def get_specs(self):
        return {
            "dims": self.dims,
            "hidden_size": self.hidden_size,
            "input_dim": self.input_dim,
            "readout_dim": self.readout_dim,
        }

    def print_specs(self):
        print_dict(f"{self.__class__.__name__} Specs", self.get_specs())
