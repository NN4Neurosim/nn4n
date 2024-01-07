import torch
import torch.nn as nn

from nn4n.model import BaseNN
from nn4n.layer import RecurrentLayer
from nn4n.layer import LinearLayer

param_list = ['dims', 'preact_noise', 'postact_noise', 'learnable', 
              'init_state', 'activation', 'tau', 'dt', 'weights',
              'biases', 'sparsity_masks', 'ei_masks', 'plasticity_masks']

dep_param_list = ['input_dim', 'output_dim', 'hidden_size', 'ei_balance', 
              'allow_negative', 'use_dale', 'new_synapses', 'positivity_constraints', 
              'sparsity_constraints', 'layer_distributions', 'layer_biases', 
              'layer_masks', 'scaling', 'self_connections']

class CTRNN(BaseNN):
    """ Recurrent network model """
    def __init__(self, **kwargs):
        """
        Base RNN constructor

        """
        super().__init__(**kwargs)

    # INITIALIZATION
    # ======================================================================================
    def _initialize(self, **kwargs):
        """ Initialize/Reinitialize the network """
        # parameters that used in all layers
        # base parameters
        self.dims = kwargs.pop("dims", [1, 100, 1])
        self.biases = kwargs.pop("biases", [None, None, None])
        self.weights = kwargs.pop("weights", ['uniform', 'uniform', 'uniform'])

        # network dynamics parameters
        self.sparsity_masks = kwargs.pop("sparsity_masks", [None])
        self.ei_masks = kwargs.pop("ei_masks", [None])
        self.plasticity_masks = kwargs.pop("plasticity_masks", [None])

        # check if all parameters meet the requirements
        self._check_parameters(kwargs)
        ctrnn_structs = self._build_structures()

        # layers
        self.recurrent_layer = RecurrentLayer(
            in_struct=ctrnn_structs[0],
            hid_struct=ctrnn_structs[1],
            **kwargs
        )
        self.readout_layer = LinearLayer(
            structure=ctrnn_structs[2],
        )

    def _handle_deprecated(self, **kwargs):
        """ Handle deprecated parameters """
        # check if there is any deprecated parameter
        if 'input_dim' in kwargs or 'output_dim' in kwargs or 'hidden_size' in kwargs:
            if 'dims' not in kwargs:
                self.dims = [kwargs.pop("input_dim", 1), kwargs.pop("hidden_size", 100), kwargs.pop("output_dim", 1)]
            print("WARNING: input_dim, output_dim, hidden_size are deprecated. Use dims instead.")
        if 'ei_balance' in kwargs:
            print("WARNING: ei_balance is deprecated. No ei_balance specification is needed.")
        if 'allow_negative' in kwargs:
            print("WARNING: allow_negative is deprecated. No allow_negative specification is needed.")
        if 'use_dale' in kwargs:
            print("WARNING: use_dale is deprecated. Use ei_masks instead. No Dale's law is applied.")
        if 'new_synapses' in kwargs:
            print("WARNING: new_synapses is deprecated. Use sparsity_masks instead. No synapse constraint is applied.")
        if 'learnable' in kwargs:
            print("WARNING: learnable is deprecated. Use `plasticity_masks` instead.\n\
                If you wish to define the learning behavior of the weights, generate a matrix \
                with the desired learning rate and pass it to the model via `plasticity_masks`.")
        if 'positivity_constraints' in kwargs:
            print("WARNING: positivity_constraints is deprecated. Use `ei_masks` instead.\n\
                If you wish to constraint the positivity of the weights, generate a matrix with the \
                desired positivity/netagivity and pass it to the model via `ei_masks`.")
        if 'sparsity_constraints' in kwargs:
            print("WARNING: sparsity_constraints is deprecated. Use sparsity_masks instead.\n\
                If you wish to constraint the sparsity of the weights, mask the element that you wish \
                to constrain to zero and pass the mask to the model via `sparsity_masks`.")
        if 'scaling' in kwargs:
            print("WARNING: scaling is deprecated. Use `weights` instead\n\
                If you wish to scale the weights, generate the weights with the desired scaling and pass them to the model via `weights`.")
        if 'self_connections' in kwargs:
            print("WARNING: self_connections is deprecated. Use sparsity_masks instead.\n\
                If you wish to constraint the sparsity of the weights, mask the element that you wish, \
                i.e. the diagonal elements, to constrain to zero and pass the mask to the model via `sparsity_masks`.")
        if 'layer_distributions' in kwargs:
            print("WARNING: layer_distributions is deprecated. Use `weights` instead.\n\
                The parameter `weights` inherits the functionality of `layer_distributions`.\
                Simply pass a list of distributions to `weights` and the model will generate the weights accordingly.")
        if 'layer_biases' in kwargs:
            print("WARNING: layer_biases is deprecated. Use `biases` instead.\n\
                The parameter `biases` inherits the functionality of `layer_biases`.\
                Simply pass a list of biases to `biases` and the model will generate the biases accordingly.")
        if 'layer_masks' in kwargs:
            print("WARNING: layer_masks is deprecated. The old functionality of `layer_masks` is now \
                split into `sparsity_masks`, `ei_masks`, and `plasticity_masks`.")

    @staticmethod
    def _check_ei_masks(ei_masks, dims):
        """ Check ei_masks """
        # must be an array of 3 masks/3 Nones/None
        if ei_masks == None: ei_masks = [None] * 3
        if type(ei_masks) != list:
            raise ValueError("ei_masks must be a list or None")
        # if ei_masks is a list, check if it is a list of 3 legit values
        if len(ei_masks) != 3:
            raise ValueError("ei_masks must be a list of 3 or None")
        else:
            # generate target dim
            target_dim = [(dims[0], dims[1]), (dims[1], dims[1]), (dims[1], dims[2])]
            # check if each mask is a valid mask
            for i in range(3):
                # skip if ei_masks[i] is None
                if ei_masks[i] != None:
                    if type(ei_masks[i]) != np.ndarray:
                        raise ValueError("ei_masks[{}] must be a numpy array".format(i))
                    if ei_masks[i].shape != target_dim[i]:
                        raise ValueError("ei_masks[{}] must be a numpy array of shape {}".format(i, target_dim[i]))
                    ei_masks[i] = np.where(ei_masks[i] > 0, 1, -1)
            return ei_masks

    @staticmethod
    def _check_weights(weights, dims):
        """ Check weights """
        # must be an array of 3 matrices/3 dists/1 dist
        # if weights is an signle value, convert it to a list of 3 values
        if type(weights) == str:
            weights = [weights] * 3
        if type(weights) != list:
            raise ValueError("weights must be a list or a string")
        # if weights is a list, check if it is a list of 3 legit values
        if len(weights) != 3:
            raise ValueError("weights must be a list of 3 or a single distribution")
        else:
            # generate target dim
            target_dim = [(dims[0], dims[1]), (dims[1], dims[1]), (dims[1], dims[2])]
            # check if each weight is a valid weight
            for i in range(3):
                if weights[i] != None:
                    if type(weights[i]) == str:
                        if weights[i] not in ['uniform', 'normal']:
                            raise ValueError("weights[{}] must be a string of 'uniform' or 'normal'".format(i))
                    elif type(weights[i]) == np.ndarray:
                        if weights[i].shape != target_dim[i]:
                            raise ValueError("weights[{}] must be a numpy array of shape {}".format(i, target_dim[i]))
                    else:
                        raise ValueError("weights[{}] must be a string of 'uniform' or 'normal' or a numpy array".format(i))
            return weights

    @staticmethod
    def _check_biases(biases, dims):
        """ Check biases """
        # must be an array of 3 values/3 dist/1 dist/3 Nones/1 None
        # if biases is an signle value, convert it to a list of 3 values
        if type(biases) == str or biases == None:
            biases = [biases] * 3
        if type(biases) != list:
            raise ValueError("biases must be a list, a string, or None")
        # if biases is a list, check if it is a list of 3 legit values
        if len(biases) != 3:
            raise ValueError("biases must be a list of 3 or a single distribution")
        else:
            # generate target dim
            target_dim = [dims[1], dims[1], dims[2]]
            # check if each bias is a valid bias
            for i in range(3):
                if biases[i] != None:
                    if type(biases[i]) == str:
                        if biases[i] not in ['uniform', 'normal']:
                            raise ValueError("biases[{}] must be a string of 'uniform' or 'normal'".format(i))
                    elif type(biases[i]) == np.ndarray:
                        if biases[i].shape != target_dim[i]:
                            raise ValueError("biases[{}] must be a numpy array of shape {}".format(i, target_dim[i]))
                    else:
                        raise ValueError("biases[{}] must be a string of 'uniform' or 'normal' or a numpy array".format(i))
            return biases

    @staticmethod
    def _check_sparsity_masks(sparsity_masks, dims):
        """ Check sparsity_masks """
        # must be an array of 3 masks/3 Nones/None
        if sparsity_masks == None: sparsity_masks = [None] * 3
        if type(sparsity_masks) != list:
            raise ValueError("sparsity_masks must be a list or None")
        # if sparsity_masks is a list, check if it is a list of 3 legit values
        if len(sparsity_masks) != 3:
            raise ValueError("sparsity_masks must be a list of 3 or None")
        else:
            # generate target dim
            target_dim = [(dims[0], dims[1]), (dims[1], dims[1]), (dims[1], dims[2])]
            # check if each mask is a valid mask
            for i in range(3):
                # skip if sparsity_masks[i] is None
                if sparsity_masks[i] != None:
                    if type(sparsity_masks[i]) != np.ndarray:
                        raise ValueError("sparsity_masks[{}] must be a numpy array".format(i))
                    if sparsity_masks[i].shape != target_dim[i]:
                        raise ValueError("sparsity_masks[{}] must be a numpy array of shape {}".format(i, target_dim[i]))
                    sparsity_masks[i] = np.where(sparsity_masks[i] == 0, 0, 1)
            return sparsity_masks

    @staticmethod
    def _check_plasticity_masks(plasticity_masks, dims):
        """ Check plasticity_masks """
        # must be an array of 3 masks/3 Nones/None
        if plasticity_masks == None: plasticity_masks = [None] * 3
        if type(plasticity_masks) != list:
            raise ValueError("plasticity_masks must be a list or None")
        # if plasticity_masks is a list, check if it is a list of 3 legit values
        if len(plasticity_masks) != 3:
            raise ValueError("plasticity_masks must be a list of 3 or None")
        else:
            # generate target dim
            target_dim = [(dims[0], dims[1]), (dims[1], dims[1]), (dims[1], dims[2])]
            # check if each mask is a valid mask
            for i in range(3):
                # skip if plasticity_masks[i] is None
                if plasticity_masks[i] != None:
                    if type(plasticity_masks[i]) != np.ndarray:
                        raise ValueError("plasticity_masks[{}] must be a numpy array".format(i))
                    if plasticity_masks[i].shape != target_dim[i]:
                        raise ValueError("plasticity_masks[{}] must be a numpy array of shape {}".format(i, target_dim[i]))
            
            # normalize plasticity_masks
            if any([mask is not None for mask in plasticity_masks]):
                min_val = min([mask.min() for mask in plasticity_masks if mask is not None])
                max_val = max([mask.max() for mask in plasticity_masks if mask is not None])
                for i in range(3):
                    if plasticity_masks[i] is not None:
                        plasticity_masks[i] = (plasticity_masks[i] - min_val) / (max_val - min_val)
                    else:
                        plasticity_masks[i] = np.ones(target_dim[i])
            return plasticity_masks

    def _check_parameters(self, kwargs):
        """ Check parameters """
        # check dims
        assert type(self.dims) == list, "dims must be a list"
        assert len(self.dims) == 3, "dims must be a list of length 3"
        for i in self.dims:
            assert type(i) == int, f"dims must be a list of integers, {i} is not an integer"
            assert i > 0, "dims must be a list of positive integers"

        # check ei_masks
        self.ei_masks = self._check_ei_masks(self.ei_masks, self.dims)
        # check weights
        self.weights = self._check_weights(self.weights, self.dims)
        # check biases
        self.biases = self._check_biases(self.biases, self.dims)
        # check sparsity_masks
        self.sparsity_masks = self._check_sparsity_masks(self.sparsity_masks, self.dims)
        # check plasticity_masks
        self.plasticity_masks = self._check_plasticity_masks(self.plasticity_masks, self.dims)
        
        # check all key in kwargs
        for key in kwargs:
            if not key in param_list:
                print("unrecognized parameter: {}".format(key))
    # ======================================================================================

    # FORWARD
    # ======================================================================================
    def train(self):
        self.recurrent_layer.preact_noise = self.preact_noise
        self.recurrent_layer.postact_noise = self.postact_noise
        self.training = True

    def eval(self):
        self.recurrent_layer.preact_noise = 0
        self.recurrent_layer.postact_noise = 0
        self.training = False

    def forward(self, x):
        """ Forwardly update network W_in -> n x W_rc -> W_out """
        # skip constraints if the model is not in training mode
        if self.training:
            self._enforce_constraints()
        hidden_states = self.recurrent_layer(x)
        output = self.readout_layer(hidden_states.float())
        return output, hidden_states

    def _enforce_constraints(self):
        self.recurrent_layer.enforce_constraints()
        self.readout_layer.enforce_constraints()

    # HELPER FUNCTIONS
    # ======================================================================================
    def to(self, device):
        """ Move the network to device """
        super().to(device)
        self.recurrent_layer.to(device)
        self.readout_layer.to(device)

    def print_layers(self):
        """ Print the parameters of each layer """
        self.recurrent_layer.print_layers()
        self.readout_layer.print_layers()

    def plot_layers(self, **kwargs):
        """ Plot the network """
        self.recurrent_layer.plot_layers(**kwargs)
        self.readout_layer.plot_layers(**kwargs)
    # ======================================================================================
