import numpy as np
from nn4n.mask.base_mask import BaseMask
from nn4n.utils.helper_functions import print_dict


class MultiIO(BaseMask):
    def __init__(self, **kwargs):
        """
        The MultiIO generate masks when there are multiple groups/types (e.g., a 1-dim olfactory signal +
        100-dim visual signal will be two groups) of signals that required to be projected to
        different hidden layer regions. The generated masks primarily works on the input/readout layer.

        @kwarg input_dims: a list denoting the dimensions of each group of input signals.
            E.g., if 1-dim olfactory signal + 100-dim visual signal will be two groups, then [1, 100],
            must sum-up to dims[0]
        @kwarg input_dims: a list denoting the dimensions of each group of readout signals.
            E.g., if 1-dim olfactory signal + 100-dim visual signal will be two groups, then [1, 100],
            must sum-up to dims[2]
        @kwarg input_table: a table denoting whether an input signal will be projected to a given
            hidden layer node. Must be of a table of shape (n_input_groups, hidden_size) and containing
            only 0s or 1s, default: all ones.
        @kwarg readout_table: a table denoting whether a hidden layer node will be used to generate a
            specific readout.  Must be of a table of shape (n_readout_groups, hidden_size) and containing
            only 0s or 1s, default: all ones.
        """
        super().__init__(**kwargs)
        self.input_dims = kwargs.get("input_dims", [self.dims[0]])
        self.readout_dims = kwargs.get("readout_dims", [self.dims[2]])
        # number of groups of input signals
        self.n_input_groups = len(self.input_dims)
        # number of groups of readout signals
        self.n_readout_groups = len(self.readout_dims)
        self.input_table = kwargs.get(
            "input_table", np.ones((self.n_input_groups, self.dims[1]))
        )
        self.readout_table = kwargs.get(
            "readout_table", np.ones((self.n_readout_groups, self.dims[1]))
        )

        # check parameters and generate masks
        self._check_parameters()
        self._generate_masks()

    def _check_parameters(self):
        """Check if parameters are valid"""
        super()._check_parameters()

        # The input/readout dims must be a list
        assert type(self.input_dims) == list and self._check_int_list(
            self.input_dims
        ), "input_dims must be a list of integers"
        assert type(self.readout_dims) == list and self._check_int_list(
            self.readout_dims
        ), "readout_dims must be a list of integers"

        # Check if the input_dims and readout_dims all sum up to self.dims[0] and self.dims[2]
        assert (
            np.sum(self.input_dims) == self.dims[0]
        ), "input_dims must sum-up to the full input dimension specified in self.dims[0]"
        assert (
            np.sum(self.readout_dims) == self.dims[2]
        ), "readout_dims must sum-up to the full readout dimension specified in self.dims[2]"

        # Check if the input/readout table dimension is valid
        assert self.input_table.shape == (self.n_input_groups, self.dims[1])
        assert self.readout_table.shape == (self.n_readout_groups, self.dims[1])

        # TODO: check if all input/readout table are zero/one.

    @staticmethod
    def _check_int_list(el_list):
        all_int = True
        for el in el_list:
            all_int = all_int and type(el) == int
        return all_int

    def _generate_hidden_mask(self):
        """Hidden mask is not important for this class, thus all ones by default"""
        hidden_mask = np.ones((self.dims[1], self.dims[1]))
        self.hidden_mask = hidden_mask.T  # TODO: remove this and flip other masks

    def _generate_input_mask(self):
        input_mask = np.zeros((self.dims[0], self.dims[1]))
        dim_counter = 0
        for i, d in enumerate(self.input_dims):
            input_idx = self.input_table[i].reshape(-1, 1)
            input_mask[dim_counter : dim_counter + d, :] = np.tile(input_idx, d).T
            dim_counter += d
        self.input_mask = input_mask.T  # TODO: remove this and flip other masks

    def _generate_readout_mask(self):
        readout_mask = np.zeros((self.dims[1], self.dims[2]))
        dim_counter = 0
        for i, d in enumerate(self.readout_dims):
            readout_idx = self.readout_table[i].reshape(-1, 1)
            readout_mask[:, dim_counter : dim_counter + d] = np.tile(readout_idx, d)
            dim_counter += d
        self.readout_mask = readout_mask.T  # TODO: remove this and flip other masks

    def get_specs(self):
        specs = super().get_specs()
        specs["input_dims"] = self.input_dims
        specs["readout_dims"] = self.readout_dims
        specs["input_table_shape"] = self.input_table.shape
        specs["readout_table_shape"] = self.readout_table.shape
        return specs
