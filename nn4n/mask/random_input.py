import numpy as np
from nn4n.mask.base_mask import BaseMask
from nn4n.utils.helper_functions import print_dict


class RandomInput(BaseMask):
    def __init__(self, **kwargs):
        """
        Generate a network with randomly injected input and readout neurons
        @kwarg input_spar: sparsity of the input layer, default: 0.1
        @kwarg readout_spar: sparsity of the readout layer, default: 0.1
        @kwarg overlap: overlap between input and readout neurons, default: True
        """
        super().__init__(**kwargs)
        self.input_spar = kwargs.get("input_spar", 1)
        self.readout_spar = kwargs.get("readout_spar", 1)
        self.hidden_spar = kwargs.get("hidden_spar", 1)
        self.overlap = kwargs.get("overlap", True)

        # run if it is not a child class
        if self.__class__.__name__ == "RandomInput":
            self._check_parameters()
            self._generate_masks()

    def _check_parameters(self):
        """
        Check if parameters are valid
        """
        super()._check_parameters()

        # check input_spar
        assert isinstance(self.input_spar, float), "input_spar must be float"
        assert 0 <= self.input_spar <= 1, "input_spar must be between 0 and 1"

        # check readout_spar
        assert isinstance(self.readout_spar, float), "readout_spar must be float"
        assert 0 <= self.readout_spar <= 1, "readout_spar must be between 0 and 1"

        # if not overlap, check if input_spar + readout_spar <= 1
        if not self.overlap:
            assert (
                self.input_spar + self.readout_spar <= 1
            ), "input_spar + readout_spar must be less than 1 if overlap is True"

    def _generate_hidden_mask(self):
        """
        Generate the mask for the hidden layer
        """
        self.hidden_mask = np.random.uniform(0, 1, (self.hidden_size, self.hidden_size))
        self.hidden_mask[self.hidden_mask > self.hidden_spar] = 0
        self.hidden_mask[self.hidden_mask > 0] = 1

    def _generate_input_mask(self):
        """
        Generate the mask for the input layer
        """
        self.input_idx = np.random.choice(
            self.hidden_size, int(self.hidden_size * self.input_spar), replace=False
        )
        self.non_input_idx = np.setdiff1d(np.arange(self.hidden_size), self.input_idx)
        self.input_mask = np.zeros((self.hidden_size, self.input_dim))
        self.input_mask[np.ix_(self.input_idx, np.arange(self.input_dim))] = (
            self._generate_sparse_matrix(len(self.input_idx), self.input_dim, 1)
        )

    def _generate_readout_mask(self):
        """
        Generate the mask for the readout layer
        """
        if self.overlap:
            self.readout_idx = np.random.choice(
                self.hidden_size,
                int(self.hidden_size * self.readout_spar),
                replace=False,
            )
        else:
            self.readout_idx = np.random.choice(
                self.non_input_idx,
                int(self.hidden_size * self.readout_spar),
                replace=False,
            )

        self.readout_mask = np.zeros((self.readout_dim, self.hidden_size))
        self.readout_mask[np.ix_(np.arange(self.readout_dim), self.readout_idx)] = (
            self._generate_sparse_matrix(self.readout_dim, len(self.readout_idx), 1)
        )

    def get_input_indices(self):
        """
        Return the indices of neurons that receive input
        """
        return self.input_idx

    def get_non_input_indices(self):
        """
        Return the indices of neurons that do not receive input
        """
        return self.non_input_idx

    def get_readout_indices(self):
        """
        Return the indices of neurons that send readout
        """
        return self.readout_idx

    def get_input_mask(self):
        """
        Return the mask for the input layer
        """
        return self.input_mask

    def get_areas(self):
        """
        Return the area of each layer
        """
        return ["area_input", "area_readout"]

    def get_area_indices(self, area):
        """
        Return the indices of neurons in each area
        """
        indicies = [self.input_idx, self.readout_idx]
        if isinstance(area, str):
            area = self.get_areas().index(area)
        return indicies[area]

    def get_specs(self):
        specs = super().get_specs()
        specs["input_spar"] = self.input_spar
        specs["readout_spar"] = self.readout_spar
        specs["hidden_spar"] = self.hidden_spar
        specs["overlap"] = self.overlap
        return specs
