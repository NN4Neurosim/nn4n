import numpy as np

class MultiArea():
    def __init__(self, **kwargs):
        """
        Generate the multi-area network mask, a potential drawback is that the network only supports 3 areas
        It hasn't been generalized to n areas
        """
        self.exc_pct = kwargs.get("exc_pct", 0.8)
        self.fwd_pct = kwargs.get("fwd_pct", 0.1)
        self.bwd_pct = kwargs.get("bwd_pct", 0.1)
        self.n_neurons = kwargs.get("n_neurons", 100)
        self.input_size = kwargs.get("input_size", 2)
        self.output_size = kwargs.get("output_size", 2)

    def generate_random_sparse(self, sparsity, n_row, n_col):
        """
        Generate random sparse matrix
        """
        mask = np.random.rand(n_row, n_col)
        mask = np.where(mask > sparsity, 0, 1)
        return mask

    def generate_sparse_mask(self):
        """
        Generate sparse mask for the network
        """
        exc_pct = self.exc_pct
        fwd_pct = self.fwd_pct
        bwd_pct = self.bwd_pct
        n_neurons = self.n_neurons

        # split excitatory and inhibitory neurons
        n_exc = int(n_neurons * exc_pct)
        n_inh = n_neurons - n_exc
        assert n_exc % 3 == 0, "Number of excitatory neurons must be divisible by 3"
        assert n_inh % 3 == 0, "Number of inhibitory neurons must be divisible by 3"
        n_exc_per_area = int(n_exc / 3)
        n_inh_per_area = int(n_inh / 3)

        # compute neuron indices
        exc_sensory_idx = np.arange(n_exc_per_area)
        exc_pre_motor_idx = np.arange(n_exc_per_area, 2 * n_exc_per_area)
        exc_motor_idx = np.arange(2 * n_exc_per_area, 3 * n_exc_per_area)
        inh_sensory_idx = np.arange(3 * n_exc_per_area, 3 * n_exc_per_area + n_inh_per_area)
        inh_pre_motor_idx = np.arange(3 * n_exc_per_area + n_inh_per_area, 3 * n_exc_per_area + 2 * n_inh_per_area)
        inh_motor_idx = np.arange(3 * n_exc_per_area + 2 * n_inh_per_area, 3 * n_exc_per_area + 3 * n_inh_per_area)

        self.exc_sensory_idx = exc_sensory_idx
        self.exc_pre_motor_idx = exc_pre_motor_idx
        self.exc_motor_idx = exc_motor_idx
        self.inh_sensory_idx = inh_sensory_idx
        self.inh_pre_motor_idx = inh_pre_motor_idx
        self.inh_motor_idx = inh_motor_idx

        # generate sparse masks
        hidden_mask = np.zeros((n_neurons, n_neurons))
        hidden_mask[np.ix_(exc_sensory_idx, exc_sensory_idx)] = 1
        hidden_mask[np.ix_(exc_pre_motor_idx, exc_pre_motor_idx)] = 1
        hidden_mask[np.ix_(exc_motor_idx, exc_motor_idx)] = 1

        hidden_mask[np.ix_(inh_sensory_idx, inh_sensory_idx)] = -1
        hidden_mask[np.ix_(inh_pre_motor_idx, inh_pre_motor_idx)] = -1
        hidden_mask[np.ix_(inh_motor_idx, inh_motor_idx)] = -1

        hidden_mask[np.ix_(exc_sensory_idx, exc_pre_motor_idx)] = self.generate_random_sparse(bwd_pct, n_exc_per_area, n_exc_per_area)
        hidden_mask[np.ix_(exc_pre_motor_idx, exc_motor_idx)] = self.generate_random_sparse(bwd_pct, n_exc_per_area, n_exc_per_area)
        hidden_mask[np.ix_(exc_pre_motor_idx, exc_sensory_idx)] = self.generate_random_sparse(fwd_pct, n_exc_per_area, n_exc_per_area)
        hidden_mask[np.ix_(exc_motor_idx, exc_pre_motor_idx)] = self.generate_random_sparse(fwd_pct, n_exc_per_area, n_exc_per_area)

        hidden_mask[np.ix_(exc_sensory_idx, inh_sensory_idx)] = -1
        hidden_mask[np.ix_(exc_pre_motor_idx, inh_pre_motor_idx)] = -1
        hidden_mask[np.ix_(exc_motor_idx, inh_motor_idx)] = -1

        hidden_mask[np.ix_(inh_sensory_idx, exc_sensory_idx)] = 1
        hidden_mask[np.ix_(inh_pre_motor_idx, exc_pre_motor_idx)] = 1
        hidden_mask[np.ix_(inh_motor_idx, exc_motor_idx)] = 1

        for i in range(n_neurons):
            hidden_mask[i, i] = 0

        input_mask = np.zeros((n_neurons, self.input_size))
        input_mask[np.ix_(exc_sensory_idx, np.arange(self.input_size))] = 1
        input_mask[np.ix_(inh_sensory_idx, np.arange(self.input_size))] = 1

        output_mask = np.zeros((self.output_size, n_neurons))
        output_mask[np.ix_(np.arange(self.output_size), exc_motor_idx)] = 1
        # output_mask[np.ix_(np.arange(self.output_size), inh_motor_idx)] = 1

        return input_mask, hidden_mask, output_mask
    
    def generate_area_idx(self):
        """ 
        Generate E/I signature
        TODO: generalize this
        """
        idx1 = np.hstack((np.arange(0, 80), np.arange(240, 260)))
        idx2 = np.hstack((np.arange(80, 160), np.arange(260, 280)))
        idx3 = np.hstack((np.arange(160, 240), np.arange(280, 300)))
        return [idx1, idx2, idx3]
