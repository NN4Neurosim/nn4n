import numpy as np
import utils

class MultiArea():
    def __init__(self, **kwargs):
        """
        Generate the multi-area network mask
        @kwarg n_areas: number of areas, list or int, default: 2
        @area_connectivities: connectivity between areas, list or np.ndarray, default: [0.1, 0.1]
        @hidden_size: number of hidden neurons in total, must be defined
        @input_size: number of input neurons, default: 1
        @output_size: number of output neurons, default: 1
        """
        self.n_areas = kwargs.get("n_areas", 2)
        self.area_connectivities = kwargs.get("area_connectivities", [0.1, 0.1])
        self.input_areas = np.array(kwargs.get("input_areas", None))
        self.output_areas = np.array(kwargs.get("output_areas", None))
        self.hidden_size = kwargs.get("hidden_size", None)
        self.input_size = kwargs.get("input_size", 1)
        self.output_size = kwargs.get("output_size", 1)

        # run if it is not a child class
        if self.__class__.__name__ == "MultiArea":
            self.check_parameters()
            self.generate_mask()


    def check_parameters(self):
        """
        Check if parameters are valid
        """
        ## check hidden_size
        assert self.hidden_size is not None, "hidden_size must be defined"
        assert isinstance(self.hidden_size, int), "hidden_size must be int"
        assert self.hidden_size > 0, "hidden_size must be positive"

        ## check n_areas
        if isinstance(self.n_areas, int):
            assert self.hidden_size % self.n_areas == 0, "hidden_size must be devideable by n_areas"
            # create a node assignment list
            node_assigment = np.zeros(self.n_areas, dtype=np.ndarray)
            for i in range(self.n_areas):
                node_assigment[i] = np.arange(self.hidden_size // self.n_areas) + i * (self.hidden_size // self.n_areas)
            self.node_assigment = node_assigment
        elif isinstance(self.n_areas, list):
            assert sum(self.n_areas) == self.hidden_size, "sum of n_areas must be equal to hidden_size"
            assert np.all(np.array(self.n_areas) > 0), "elements of n_areas must be larger than 0"
            # create a node assignment list
            node_assigment = np.zeros(len(self.n_areas), dtype=np.ndarray)
            for i in range(len(self.n_areas)):
                node_assigment[i] = np.arange(self.n_areas[i]) + sum(self.n_areas[:i])
            self.node_assigment = node_assigment
            self.n_areas = len(self.n_areas)
        else:
            assert False, "n_areas must be int or list"

        if self.n_areas == 1:
            self.input_areas = np.array([0])
            self.output_areas = np.array([0])

        ## check area_connectivities
        assert np.all(0 <= np.array(self.area_connectivities)) and np.all(np.array(self.area_connectivities) <= 1), "area_connectivities must be between 0 and 1"
        if isinstance(self.area_connectivities, list):
            assert len(self.area_connectivities) in [2, 3], "length of area_connectivities must be 2 or 3"
            # transform list to np.ndarray
            area_connectivities = np.zeros((self.n_areas, self.n_areas))
            area_connectivities[np.tril_indices(self.n_areas, -1)] = self.area_connectivities[0]
            area_connectivities[np.triu_indices(self.n_areas, 1)] = self.area_connectivities[1]
            area_connectivities[np.diag_indices(self.n_areas)] = self.area_connectivities[2] if len(self.area_connectivities) == 3 else 1
            self.area_connectivities = area_connectivities
        elif isinstance(self.area_connectivities, np.ndarray):
            assert self.area_connectivities.shape == (self.n_areas, self.n_areas), "shape of area_connectivities must be (n_areas, n_areas)"
        else:
            assert False, "area_connectivities must be list or np.ndarray"

        ## check input_areas and output_areas
        if self.input_areas is None:
            self.input_areas = np.arange(self.n_areas)
        else:
            assert np.all(0 <= self.input_areas) and np.all(self.input_areas < self.n_areas), "input_areas must be between 0 and n_areas"
        
        if self.output_areas is None:
            self.output_areas = np.arange(self.n_areas)
        else:
            assert np.all(0 <= self.output_areas) and np.all(self.output_areas < self.n_areas), "output_areas must be between 0 and n_areas"

    
    def generate_mask(self):
        """
        Generate the mask for the multi-area network
        """
        self.generate_hidden_mask()
        self.generate_input_mask()
        self.generate_output_mask()
 

    def generate_hidden_mask(self):
        """
        Generate the mask for the hidden layer
        """
        hidden_mask = np.zeros((self.hidden_size, self.hidden_size))
        for i in range(self.n_areas):
            for j in range(self.n_areas):
                if self.area_connectivities[i, j] > 0:
                    area_i_size = len(self.node_assigment[i])
                    area_j_size = len(self.node_assigment[j])
                    hidden_mask[np.ix_(self.node_assigment[i], self.node_assigment[j])] = self.generate_sparse_matrix(area_i_size, area_j_size, self.area_connectivities[i, j])
        self.hidden_mask = hidden_mask


    def generate_input_mask(self):
        """
        Generate the mask for the input layer
        """
        input_mask = np.zeros((self.hidden_size, self.input_size))
        for i in self.input_areas:
            area_i_size = len(self.node_assigment[i])
            input_mask[np.ix_(self.node_assigment[i], np.arange(self.input_size))] = self.generate_sparse_matrix(area_i_size, self.input_size, 1)
        self.input_mask = input_mask


    def generate_output_mask(self):
        """
        Generate the mask for the output layer
        """
        output_mask = np.zeros((self.output_size, self.hidden_size))
        for i in self.output_areas:
            area_i_size = len(self.node_assigment[i])
            output_mask[np.ix_(np.arange(self.output_size), self.node_assigment[i])] = self.generate_sparse_matrix(self.output_size, area_i_size, 1)
        self.output_mask = output_mask


    def generate_sparse_matrix(self, n, m, p):
        """
        Generate a sparse matrix with size n x m and density p. 1 if connection exists, 0 otherwise
        """
        assert 0 <= p <= 1, "p must be between 0 and 1"
        mask = np.random.rand(n, m) < p
        return mask.astype(int)
    

    def visualize(self):
        utils.plot_connectivity_matrix(self.hidden_mask, "Hidden Layer Connectivity", False)

        input_mask_ = self.input_mask if self.input_mask.shape[1] > self.input_mask.shape[0] else self.input_mask.T
        output_mask_ = self.output_mask if self.output_mask.shape[1] > self.output_mask.shape[0] else self.output_mask.T

        utils.plot_connectivity_matrix(input_mask_, "Input Layer Connectivity", False)
        utils.plot_connectivity_matrix(output_mask_, "Output Layer Connectivity", False)




class MultiAreaEI(MultiArea):
    def __init__(self, **kwargs):
        """
        Generate the multi-area network with ei constraints mask
        NOTE: On top of the keywards required for MultiArea:
        @kwarg exc_pct: percentage of excitatory neurons, default: 0.8
        @kwarg inter_area_connections: list of boolean, whether to have inter-area connections
            [exc_exc, exc_inh, inh_exc, inh_inh]
            default: [True, True, True, True]
        """
        super().__init__(**kwargs)

        self.exc_pct = kwargs.get("exc_pct", 0.8)
        self.inter_area_connections = kwargs.get("inter_area_connections", [True, True, True, True])

        self.check_parameters()
        self.generate_mask()


    def check_parameters(self):
        super().check_parameters()

        ## check exc_pct
        assert 0 <= self.exc_pct <= 1, "exc_pct must be between 0 and 1"

        ## check if inter_area_connections is list of 4 boolean
        assert isinstance(self.inter_area_connections, list) and len(self.inter_area_connections) == 4, "inter_area_connections must be list of 4 boolean"
        for i in range(4):
            assert isinstance(self.inter_area_connections[i], bool), "inter_area_connections must be list of 4 boolean"


    def generate_mask(self):
        """
        Generate the mask for the multi-area network
        """
        super().generate_mask()
        self.generate_ei_assigment()
        self.masks_to_ei()


    def generate_ei_assigment(self):
        """
        Generate the assignment of excitatory and inhibitory neurons
        """
        self.excitatory_neurons = np.zeros(self.n_areas, dtype=np.ndarray)
        self.inhibitory_neurons = np.zeros(self.n_areas, dtype=np.ndarray)

        for i in range(self.n_areas):
            area_i_size = len(self.node_assigment[i])
            n_exc = int(area_i_size * self.exc_pct)
            n_inh = area_i_size - n_exc
            self.excitatory_neurons[i] = self.node_assigment[i][:n_exc]
            self.inhibitory_neurons[i] = self.node_assigment[i][n_exc:]


    def masks_to_ei(self):
        """
        Convert the masks to ei masks
        """
        for i in range(self.n_areas):
            self.hidden_mask[:, self.inhibitory_neurons[i]] *= -1

        # remove exc_exc connections between areas
        if not self.inter_area_connections[0]:
            for i in range(self.n_areas):
                for j in range(i+1, self.n_areas):
                    self.hidden_mask[np.ix_(self.excitatory_neurons[i], self.excitatory_neurons[j])] = 0
                    self.hidden_mask[np.ix_(self.excitatory_neurons[j], self.excitatory_neurons[i])] = 0

        # remove exc_inh connections between areas
        if not self.inter_area_connections[1]:
            for i in range(self.n_areas):
                for j in range(self.n_areas):
                    if i != j:
                        self.hidden_mask[np.ix_(self.inhibitory_neurons[i], self.excitatory_neurons[j])] = 0

        # remove inh_exc connections between areas
        if not self.inter_area_connections[2]:
            for i in range(self.n_areas):
                for j in range(self.n_areas):
                    if i != j:
                        self.hidden_mask[np.ix_(self.excitatory_neurons[i], self.inhibitory_neurons[j])] = 0

        # remove inh_inh connections between areas
        if not self.inter_area_connections[3]:
            for i in range(self.n_areas):
                for j in range(i+1, self.n_areas):
                    self.hidden_mask[np.ix_(self.inhibitory_neurons[i], self.inhibitory_neurons[j])] = 0
                    self.hidden_mask[np.ix_(self.inhibitory_neurons[j], self.inhibitory_neurons[i])] = 0

    # def masks_to_ei(self):
    #     """
    #     Convert the masks to ei masks
    #     """


    # def __init__(self, **kwargs):
    #     """
    #     Generate the multi-area network with ei constraints mask
    #     @kwarg n_area: number of areas, default: 2
    #     @kwarg exc_pct: percentage of excitatory neurons, default: 0.8
    #     @kwarg fwd_pct: percentage of forward connections, default: 0.1
    #     @kwarg bwd_pct: percentage of backward connections, default: 0.1
    #     @kwarg n_neurons: number of neurons per area, default: 100
    #     @kwarg input_size: number of input neurons, default: 2
    #     @kwarg output_size: number of output neurons, default: 2
    #     """
    #     self.n_area = kwargs.get("n_area", 2)
    #     self.exc_pct = kwargs.get("exc_pct", 0.8)
    #     self.fwd_pct = kwargs.get("fwd_pct", 0.1)
    #     self.bwd_pct = kwargs.get("bwd_pct", 0.1)
    #     self.n_neurons = kwargs.get("n_neurons", 100)
    #     self.input_size = kwargs.get("input_size", 2)
    #     self.output_size = kwargs.get("output_size", 2)

    # def generate_random_sparse(self, sparsity, n_row, n_col):
    #     """
    #     Generate random sparse matrix
    #     """
    #     mask = np.random.rand(n_row, n_col)
    #     mask = np.where(mask > sparsity, 0, 1)
    #     return mask

    # def generate_sparse_mask(self):
    #     """
    #     Generate sparse mask for the network
    #     """
    #     exc_pct = self.exc_pct
    #     fwd_pct = self.fwd_pct
    #     bwd_pct = self.bwd_pct
    #     n_neurons = self.n_neurons

    #     # split excitatory and inhibitory neurons
    #     n_exc = int(n_neurons * exc_pct)
    #     n_inh = n_neurons - n_exc
    #     assert n_exc % 3 == 0, "Number of excitatory neurons must be divisible by 3"
    #     assert n_inh % 3 == 0, "Number of inhibitory neurons must be divisible by 3"
    #     n_exc_per_area = int(n_exc / 3)
    #     n_inh_per_area = int(n_inh / 3)

    #     # compute neuron indices
    #     exc_sensory_idx = np.arange(n_exc_per_area)
    #     exc_pre_motor_idx = np.arange(n_exc_per_area, 2 * n_exc_per_area)
    #     exc_motor_idx = np.arange(2 * n_exc_per_area, 3 * n_exc_per_area)
    #     inh_sensory_idx = np.arange(3 * n_exc_per_area, 3 * n_exc_per_area + n_inh_per_area)
    #     inh_pre_motor_idx = np.arange(3 * n_exc_per_area + n_inh_per_area, 3 * n_exc_per_area + 2 * n_inh_per_area)
    #     inh_motor_idx = np.arange(3 * n_exc_per_area + 2 * n_inh_per_area, 3 * n_exc_per_area + 3 * n_inh_per_area)

    #     self.exc_sensory_idx = exc_sensory_idx
    #     self.exc_pre_motor_idx = exc_pre_motor_idx
    #     self.exc_motor_idx = exc_motor_idx
    #     self.inh_sensory_idx = inh_sensory_idx
    #     self.inh_pre_motor_idx = inh_pre_motor_idx
    #     self.inh_motor_idx = inh_motor_idx

    #     # generate sparse masks
    #     hidden_mask = np.zeros((n_neurons, n_neurons))
    #     hidden_mask[np.ix_(exc_sensory_idx, exc_sensory_idx)] = 1
    #     hidden_mask[np.ix_(exc_pre_motor_idx, exc_pre_motor_idx)] = 1
    #     hidden_mask[np.ix_(exc_motor_idx, exc_motor_idx)] = 1

    #     hidden_mask[np.ix_(inh_sensory_idx, inh_sensory_idx)] = -1
    #     hidden_mask[np.ix_(inh_pre_motor_idx, inh_pre_motor_idx)] = -1
    #     hidden_mask[np.ix_(inh_motor_idx, inh_motor_idx)] = -1

    #     hidden_mask[np.ix_(exc_sensory_idx, exc_pre_motor_idx)] = self.generate_random_sparse(bwd_pct, n_exc_per_area, n_exc_per_area)
    #     hidden_mask[np.ix_(exc_pre_motor_idx, exc_motor_idx)] = self.generate_random_sparse(bwd_pct, n_exc_per_area, n_exc_per_area)
    #     hidden_mask[np.ix_(exc_pre_motor_idx, exc_sensory_idx)] = self.generate_random_sparse(fwd_pct, n_exc_per_area, n_exc_per_area)
    #     hidden_mask[np.ix_(exc_motor_idx, exc_pre_motor_idx)] = self.generate_random_sparse(fwd_pct, n_exc_per_area, n_exc_per_area)

    #     hidden_mask[np.ix_(exc_sensory_idx, inh_sensory_idx)] = -1
    #     hidden_mask[np.ix_(exc_pre_motor_idx, inh_pre_motor_idx)] = -1
    #     hidden_mask[np.ix_(exc_motor_idx, inh_motor_idx)] = -1

    #     hidden_mask[np.ix_(inh_sensory_idx, exc_sensory_idx)] = 1
    #     hidden_mask[np.ix_(inh_pre_motor_idx, exc_pre_motor_idx)] = 1
    #     hidden_mask[np.ix_(inh_motor_idx, exc_motor_idx)] = 1

    #     for i in range(n_neurons):
    #         hidden_mask[i, i] = 0

    #     input_mask = np.zeros((n_neurons, self.input_size))
    #     input_mask[np.ix_(exc_sensory_idx, np.arange(self.input_size))] = 1
    #     input_mask[np.ix_(inh_sensory_idx, np.arange(self.input_size))] = 1

    #     output_mask = np.zeros((self.output_size, n_neurons))
    #     output_mask[np.ix_(np.arange(self.output_size), exc_motor_idx)] = 1
    #     # output_mask[np.ix_(np.arange(self.output_size), inh_motor_idx)] = 1

    #     return input_mask, hidden_mask, output_mask
    
    # def generate_area_idx(self):
    #     """ 
    #     Generate E/I signature
    #     TODO: generalize this
    #     """
    #     idx1 = np.hstack((np.arange(0, 80), np.arange(240, 260)))
    #     idx2 = np.hstack((np.arange(80, 160), np.arange(260, 280)))
    #     idx3 = np.hstack((np.arange(160, 240), np.arange(280, 300)))
    #     return [idx1, idx2, idx3]
