import numpy as np
import nn4n.utils as utils

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