import numpy as np
from .multi_area import MultiArea
from nn4n.utils.helper_functions import print_dict


class MultiAreaEI(MultiArea):
    """Multi-area network with ei constraints mask"""

    def __init__(self, **kwargs):
        """
        NOTE: On top of the keywards required for MultiArea:
        @kwarg exc_pct: percentage of excitatory neurons, default: 0.8
        @kwarg inter_area_connections: list of boolean, whether to have inter-area connections
            [exc_exc, exc_inh, inh_exc, inh_inh]
            default: [True, True, True, True]
        @kwarg inh_readout: whether to readout inhibitory neurons, default: True
        """
        super().__init__(**kwargs)
        # Initialize parameters
        self.exc_pct = kwargs.get("exc_pct", 0.8)
        self.inter_area_connections = kwargs.get(
            "inter_area_connections", [True, True, True, True]
        )
        self.inh_readout = kwargs.get("inh_readout", True)
        # Check parameters and generate mask
        self._check_parameters()
        self._generate_masks()

    def _check_parameters(self):
        super()._check_parameters()
        # Check exc_pct
        assert 0 <= self.exc_pct <= 1, "exc_pct must be between 0 and 1"
        # Check if inter_area_connections is list of 4 boolean
        assert (
            isinstance(self.inter_area_connections, list)
            and len(self.inter_area_connections) == 4
        ), "inter_area_connections must be list of 4 boolean"
        # Four elements are ‘exc-exc’, ‘exc-inh’, ‘inh-exc’, and ‘inh-inh’
        for i in range(4):
            assert isinstance(
                self.inter_area_connections[i], bool
            ), "inter_area_connections must be list of 4 boolean"

    def _generate_masks(self):
        """
        Generate the mask for the multi-area network
        """
        super()._generate_masks()
        self._generate_ei_assigment()
        self._masks_to_ei()

    def _generate_ei_assigment(self):
        """
        Generate the assignment of excitatory and inhibitory neurons
        """
        self.excitatory_neurons = np.zeros(self.n_areas, dtype=np.ndarray)
        self.inhibitory_neurons = np.zeros(self.n_areas, dtype=np.ndarray)

        for i in range(self.n_areas):
            area_i_size = len(self.node_assigment[i])
            n_exc = int(area_i_size * self.exc_pct)
            self.excitatory_neurons[i] = self.node_assigment[i][:n_exc]
            self.inhibitory_neurons[i] = self.node_assigment[i][n_exc:]

    def _masks_to_ei(self):
        """
        Convert the masks to ei masks
        """
        for i in range(self.n_areas):
            self.hidden_mask[:, self.inhibitory_neurons[i]] *= -1
            self.readout_mask[:, self.inhibitory_neurons[i]] *= -1

        # remove exc_exc connections between areas
        if not self.inter_area_connections[0]:
            for i in range(self.n_areas):
                for j in range(i + 1, self.n_areas):
                    self.hidden_mask[
                        np.ix_(self.excitatory_neurons[i], self.excitatory_neurons[j])
                    ] = 0
                    self.hidden_mask[
                        np.ix_(self.excitatory_neurons[j], self.excitatory_neurons[i])
                    ] = 0

        # remove exc_inh connections between areas
        if not self.inter_area_connections[1]:
            for i in range(self.n_areas):
                for j in range(self.n_areas):
                    if i != j:
                        self.hidden_mask[
                            np.ix_(
                                self.inhibitory_neurons[i], self.excitatory_neurons[j]
                            )
                        ] = 0

        # remove inh_exc connections between areas
        if not self.inter_area_connections[2]:
            for i in range(self.n_areas):
                for j in range(self.n_areas):
                    if i != j:
                        self.hidden_mask[
                            np.ix_(
                                self.excitatory_neurons[i], self.inhibitory_neurons[j]
                            )
                        ] = 0

        # remove inh_inh connections between areas
        if not self.inter_area_connections[3]:
            for i in range(self.n_areas):
                for j in range(i + 1, self.n_areas):
                    self.hidden_mask[
                        np.ix_(self.inhibitory_neurons[i], self.inhibitory_neurons[j])
                    ] = 0
                    self.hidden_mask[
                        np.ix_(self.inhibitory_neurons[j], self.inhibitory_neurons[i])
                    ] = 0

        # delete connections from inhibitory neurons to readout layers
        if not self.inh_readout:
            for i in range(self.n_areas):
                self.readout_mask[:, self.inhibitory_neurons[i]] = 0

    def get_sparsity_masks(self):
        masks = self.get_masks()
        # The sparsity masks will be binary version of the current masks, all 1 and -1 are 1, 0 remains 0
        sparsity_masks = []
        for mask in masks:
            sparsity_masks.append((mask != 0).astype(int))
        return sparsity_masks

    def get_specs(self):
        """
        Return the specifications of the network
        """
        specs = super().get_specs()
        specs["exc_pct"] = self.exc_pct
        specs["inter_area_connections"] = self.inter_area_connections
        specs["inh_readout"] = self.inh_readout
        return specs
