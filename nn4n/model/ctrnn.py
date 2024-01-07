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
        @kwarg dims: dimensions of the network, default: [1, 100, 1]
        @kwarg preact_noise: noise added to pre-activation, default: 0
        @kwarg postact_noise: noise added to post-activation, default: 0

        @kwarg biases: use bias or not for each layer, a list of 3 values or a single value
            if a single value is passed, it will be broadcasted to a list of 3 values, it can be:
                - None: no bias
                - 'zero' or 0: bias initialized to 0
                - 'normal': bias initialized from a normal distribution
                - 'uniform': bias initialized from a uniform distribution
            if a list of 3 values is passed, each value can be either the same as above or 
            a numpy array/torch tensor that directly specifies the bias
        @kwarg weights: distribution of weights for each layer, a list of 3 strings or 
            a single string, if a single string is passed, it will be broadcasted to a list of 3 strings
            it can be:
                - 'normal': weights initialized from a normal distribution
                - 'uniform': weights initialized from a uniform distribution
            if a list of 3 values is passed, each string can be either the same as above or 
            a numpy array/torch tensor that directly specifies the weights
        @kwarg sparsity_masks: use sparsity_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor 
            that directly specifies the sparsity_masks
        @kwarg ei_masks: use ei_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor 
            that directly specifies the ei_masks
        @kwarg plasticity_masks: use plasticity_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor 
            that directly specifies the plasticity_masks
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
        self.sparsity_masks = kwargs.pop("sparsity_masks", None)
        self.ei_masks = kwargs.pop("ei_masks", None)
        self.plasticity_masks = kwargs.pop("plasticity_masks", None)

        # temp storage
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

        # check if all parameters meet the requirements
        self._handle_deprecated(kwargs)
        self._check_parameters(kwargs)
        ctrnn_structs = self._build_structures(kwargs)

        # layers
        self.recurrent_layer = RecurrentLayer(layer_struct=ctrnn_structs[0])
        self.readout_layer = LinearLayer(layer_struct=ctrnn_structs[1])

    def _handle_deprecated(self, kwargs):
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
            print("WARNING: learnable is deprecated. Use `plasticity_masks` instead.\n"
                "   If you wish to define the learning behavior of the weights, generate a matrix "
                "with the desired learning rate and pass it to the model via `plasticity_masks`.")
        if 'positivity_constraints' in kwargs:
            print("WARNING: positivity_constraints is deprecated. Use `ei_masks` instead.\n"
                "   If you wish to constraint the positivity of the weights, generate a matrix with the "
                "desired positivity/netagivity and pass it to the model via `ei_masks`.")
        if 'sparsity_constraints' in kwargs:
            print("WARNING: sparsity_constraints is deprecated. Use sparsity_masks instead.\n"
                "   If you wish to constraint the sparsity of the weights, mask the element that you wish "
                "to constrain to zero and pass the mask to the model via `sparsity_masks`.")
        if 'scaling' in kwargs:
            print("WARNING: scaling is deprecated. Use `weights` instead\n"
                "   If you wish to scale the weights, generate the weights with the desired "
                "scaling and pass them to the model via `weights`.")
        if 'self_connections' in kwargs:
            print("WARNING: self_connections is deprecated. Use sparsity_masks instead.\n"
                "   If you wish to constraint the sparsity of the weights, mask the element that you wish, "
                "i.e. the diagonal elements, to constrain to zero and pass the mask to the model via `sparsity_masks`.")
        if 'layer_distributions' in kwargs:
            print("WARNING: layer_distributions is deprecated. Use `weights` instead.\n"
                "   The parameter `weights` inherits the functionality of `layer_distributions`."
                "Simply pass a list of distributions to `weights` and the model will generate the weights accordingly.")
        if 'layer_biases' in kwargs:
            print("WARNING: layer_biases is deprecated. Use `biases` instead.\n"
                "   The parameter `biases` inherits the functionality of `layer_biases`."
                "Simply pass a list of biases to `biases` and the model will generate the biases accordingly.")
        if 'layer_masks' in kwargs:
            print("WARNING: layer_masks is deprecated.\n   The old parameter `layer_masks` is now "
                "splitted into `sparsity_masks`, `ei_masks`, and `plasticity_masks`.")

    def _check_masks(self, param, param_type, dims):
        """ General function to check different parameter types. """
        target_dim = [(dims[0], dims[1]), (dims[1], dims[1]), (dims[1], dims[2])]
        target_dim_biases = [dims[1], dims[1], dims[2]]
 
        # Handle None cases
        if param is None:
            if param_type in ["ei_masks", "sparsity_masks", "plasticity_masks"]:
                param = [None] * 3
            else: raise ValueError(f"{param_type} cannot be None when param_type is {param_type}")
        elif param is not None and type(param) != list and param_type in ["weights", "biases"]:
            param = [param] * 3

        if type(param) != list:
            raise ValueError(f"{param_type} is/can not be broadcasted to a list")
        if len(param) != 3:
            raise ValueError(f"{param_type} is/can not be broadcasted to a list of length 3")

        for i in range(3):
            if param[i] is not None:
                if param_type in ["ei_masks", "sparsity_masks", "plasticity_masks"]:
                    param[i] = self._check_array(param[i], target_dim[i], param_type, i)
                    if param_type == "ei_masks":
                        param[i] = np.where(param[i] > 0, 1, -1)
                    elif param_type == "sparsity_masks":
                        param[i] = np.where(param[i] == 0, 0, 1)
                    elif param_type == "plasticity_masks":
                        # Normalize plasticity_masks
                        min_val, max_val = param[i].min(), param[i].max()
                        param[i] = (param[i] - min_val) / (max_val - min_val)
                elif param_type in ["weights", "biases"]:
                    self._check_distribution_or_array(param[i], target_dim_biases[i] if param_type == "biases" else target_dim[i], param_type, i)
        return param

    def _check_array(self, param, param_type, dim, index):
        if type(param) != np.ndarray:
            if type(param) == torch.Tensor: return param.numpy()
            else: raise ValueError(f"{param_type}[{index}] must be a numpy array")
        if param.shape != dim:
            raise ValueError(f"{param_type}[{index}] must be a numpy array of shape {dim}")
        return param

    def _check_distribution_or_array(self, param, param_type, dim, index):
        if type(param) == str:
            if param not in ['uniform', 'normal']:
                raise ValueError(f"{param_type}[{index}] must be a string of 'uniform' or 'normal'")
        elif type(param) == np.ndarray:
            # its already being converted to numpy array if it is a torch tensor, so no need to check
            if param.shape != dim:
                raise ValueError(f"{param_type}[{index}] must be a numpy array of shape {dim}")
        else:
            raise ValueError(f"{param_type}[{index}] must be a string of 'uniform' or 'normal' \
                or a numpy array/torch tensor with shape {dim}")

    def _check_parameters(self, kwargs):
        """ Check parameters """
        # check dims
        assert type(self.dims) == list, "dims must be a list"
        assert len(self.dims) == 3, "dims must be a list of length 3"
        for i in self.dims:
            assert type(i) == int, f"dims must be a list of integers, {i} is not an integer"
            assert i > 0, "dims must be a list of positive integers"

        # check ei_masks
        self.ei_masks = self._check_masks(self.ei_masks, "ei_masks", self.dims)
        # check weights
        self.weights = self._check_masks(self.weights, "weights", self.dims)
        # check biases
        self.biases = self._check_masks(self.biases, "biases", self.dims)
        # check sparsity_masks
        self.sparsity_masks = self._check_masks(self.sparsity_masks, "sparsity_masks", self.dims)
        # check plasticity_masks
        self.plasticity_masks = self._check_masks(self.plasticity_masks, "plasticity_masks", self.dims)
        
        # check all key in kwargs
        for key in kwargs:
            if not key in param_list and not key in dep_param_list:
                print("unrecognized parameter: {}".format(key))

    def _build_structures(self, kwargs):
        """ Build structures """
        # build structures
        rc_struct = {
            "init_state": kwargs.pop("init_state", 'zero'),
            "activation": kwargs.pop("activation", "relu"),
            "dt": kwargs.pop("dt", 1),
            "tau": kwargs.pop("tau", 1),
            "preact_noise": self.preact_noise,
            "postact_noise": self.postact_noise,
            "in_struct": {
                "input_dim": self.dims[0],
                "output_dim": self.dims[1],
                "weights": self.weights[0],
                "biases": self.biases[0],
                "sparsity_mask": self.sparsity_masks[0],
                "ei_mask": self.ei_masks[0],
                "plasticity_mask": self.plasticity_masks[0],
            },
            "hid_struct": {
                "input_dim": self.dims[1],
                "output_dim": self.dims[1],
                "weights": self.weights[1],
                "biases": self.biases[1],
                "sparsity_mask": self.sparsity_masks[1],
                "ei_mask": self.ei_masks[1],
                "plasticity_mask": self.plasticity_masks[1],
            }
        }
        out_struct = {
            "input_dim": self.dims[1],
            "output_dim": self.dims[2],
            "weights": self.weights[2],
            "biases": self.biases[2],
            "sparsity_mask": self.sparsity_masks[2],
            "ei_mask": self.ei_masks[2],
            "plasticity_mask": self.plasticity_masks[2],
        }
        return rc_struct, out_struct
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
