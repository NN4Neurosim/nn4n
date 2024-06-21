import torch
import torch.nn as nn
import numpy as np
import warnings

from nn4n.model import BaseNN
from nn4n.layer import RecurrentLayer
from nn4n.layer import LinearLayer

param_list = ['dims', 'preact_noise', 'postact_noise', 
              'init_state', 'activation', 'tau', 'dt', 'weights',
              'biases', 'sparsity_masks', 'ei_masks', 'plasticity_masks']

dep_param_list = ['input_dim', 'output_dim', 'hidden_size', 'ei_balance', 'learnable',
              'allow_negative', 'use_dale', 'new_synapses', 'positivity_constraints', 
              'sparsity_constraints', 'layer_distributions', 'layer_biases', 
              'layer_masks', 'scaling', 'self_connections']

class CTRNN(BaseNN):
    """
    Recurrent network model

    Keyword Arguments:
        - dims: dimensions of the network, default: [1, 100, 1]
        - preact_noise: noise added to pre-activation, default: 0
        - postact_noise: noise added to post-activation, default: 0
        - init_state: initial state of the network. It defines the hidden state at t=0.
        - activation: activation function, default: "relu", can be "relu", "sigmoid", "tanh", "retanh"
        - dt: time step, default: 10
        - tau: time constant, default: 100
        - biases: use bias or not for each layer, a list of 3 values or a single value
            if a single value is passed, it will be broadcasted to a list of 3 values, it can be:
            - None: no bias
            - 'zero' or 0: bias initialized to 0
            - 'normal': bias initialized from a normal distribution
            - 'uniform': bias initialized from a uniform distribution
            if a list of 3 values is passed, each value can be either the same as above or
            a numpy array/torch tensor that directly specifies the bias
        - weights: distribution of weights for each layer, a list of 3 strings or
            a single string, if a single string is passed, it will be broadcasted to a list of 3 strings
            it can be:
            - 'normal': weights initialized from a normal distribution
            - 'uniform': weights initialized from a uniform distribution
            if a list of 3 values is passed, each string can be either the same as above or
            a numpy array/torch tensor that directly specifies the weights
        - sparsity_masks: use sparsity_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor
            that directly specifies the sparsity_masks
        - ei_masks: use ei_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor
            that directly specifies the ei_masks
        - plasticity_masks: use plasticity_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor
            that directly specifies the plasticity_masks
        - synapse_growth_masks: use synapse_growth_masks or not, a list of 3 values or a single None
            if a single None is passed, it will be broadcasted to a list of 3 None
            if a list of 3 values is passed, each value can be either None or a numpy array/torch tensor
            that directly specifies the probability of growing a synapse at the selected location if there is no synapse.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # INITIALIZATION
    # ======================================================================================
    def _initialize(self, **kwargs):
        """ Initialize/Reinitialize the network """
        super()._initialize(**kwargs)
        # parameters that used in all layers
        # base parameters
        self.dims = kwargs.pop("dims", [1, 100, 1])
        self.biases = kwargs.pop("biases", None)
        self.weights = kwargs.pop("weights", 'uniform')

        # network dynamics parameters
        self.sparsity_masks = kwargs.pop("sparsity_masks", None)
        self.ei_masks = kwargs.pop("ei_masks", None)
        self.plasticity_masks = kwargs.pop("plasticity_masks", None)

        # temp storage
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

        # suppress warnings
        self.suppress_warnings = kwargs.pop("suppress_warnings", False)

        # check if all parameters meet the requirements
        if not self.suppress_warnings:
            self._handle_warnings(kwargs)
        self._check_parameters()
        rc_struct, out_struct = self._build_structures(kwargs)

        # layers
        self.recurrent_layer = RecurrentLayer(layer_struct=rc_struct)
        self.readout_layer = LinearLayer(layer_struct=out_struct)

    @property
    def layers(self):
        layer_list = [
            self.recurrent_layer.input_layer, 
            self.recurrent_layer.hidden_layer, 
            self.readout_layer
        ]
        return layer_list

    def _handle_warnings(self, kwargs):
        """ Handle deprecated parameters """
        slevel = 5 # output the warning at the level where kwargs is passed
        # check if there is any deprecated parameter
        if 'input_dim' in kwargs or 'output_dim' in kwargs or 'hidden_size' in kwargs:
            if 'dims' not in kwargs:
                self.dims = [kwargs.pop("input_dim", 1), kwargs.pop("hidden_size", 100), kwargs.pop("output_dim", 1)]
            warnings.warn("input_dim, output_dim, hidden_size are deprecated. Use dims instead.", UserWarning, stacklevel=slevel)
        if 'ei_balance' in kwargs:
            warnings.warn("ei_balance is deprecated. No ei_balance specification is needed.", UserWarning, stacklevel=slevel)
        if 'allow_negative' in kwargs:
            warnings.warn("allow_negative is deprecated. No allow_negative specification is needed.", UserWarning, stacklevel=slevel)
        if 'use_dale' in kwargs:
            warnings.warn("use_dale is deprecated. Use ei_masks instead. No Dale's law is applied.", UserWarning, stacklevel=slevel)
        if 'new_synapses' in kwargs:
            warnings.warn("new_synapses is deprecated. Use sparsity_masks instead. No synapse constraint is applied.", UserWarning, stacklevel=slevel)
        if 'learnable' in kwargs:
            warnings.warn("learnable is deprecated. Use `plasticity_masks` instead.\n"
                "   If you wish to define the learning behavior of the weights, generate a matrix "
                "with the desired learning rate and pass it to the model via `plasticity_masks`.", UserWarning, stacklevel=slevel)
        if 'positivity_constraints' in kwargs:
            warnings.warn("positivity_constraints is deprecated. Use `ei_masks` instead.\n"
                "   If you wish to constraint the positivity of the weights, generate a matrix with the "
                "desired positivity/netagivity and pass it to the model via `ei_masks`.", UserWarning, stacklevel=slevel)
        if 'sparsity_constraints' in kwargs:
            warnings.warn("sparsity_constraints is deprecated. Use sparsity_masks instead.\n"
                "   If you wish to constraint the sparsity of the weights, mask the element that you wish "
                "to constrain to zero and pass the mask to the model via `sparsity_masks`.", UserWarning, stacklevel=slevel)
        if 'scaling' in kwargs:
            warnings.warn("scaling is deprecated. Use `weights` instead.\n"
                "   If you wish to scale the weights, generate the weights with the desired "
                "scaling and pass them to the model via `weights`.", UserWarning, stacklevel=slevel)
        if 'self_connections' in kwargs:
            warnings.warn("self_connections is deprecated. Use sparsity_masks instead.\n"
                "   If you wish to constraint the sparsity of the weights, mask the element that you wish, "
                "i.e. the diagonal elements, to constrain to zero and pass the mask to the model via `sparsity_masks`.", UserWarning, stacklevel=slevel)
        if 'layer_distributions' in kwargs:
            warnings.warn("layer_distributions is deprecated. Use `weights` instead.\n"
                "   The parameter `weights` inherits the functionality of `layer_distributions`."
                "Simply pass a list of distributions to `weights` and the model will generate the weights accordingly.", UserWarning, stacklevel=slevel)
        if 'layer_biases' in kwargs:
            warnings.warn("layer_biases is deprecated. Use `biases` instead.\n"
                "   The parameter `biases` inherits the functionality of `layer_biases`."
                "Simply pass a list of biases to `biases` and the model will generate the biases accordingly.", UserWarning, stacklevel=slevel)
        if 'layer_masks' in kwargs:
            warnings.warn("layer_masks is deprecated. Use `sparsity_masks`, `ei_masks`, and `plasticity_masks` instead.\n"
                "   The parameter `sparsity_masks`, `ei_masks`, and `plasticity_masks` inherits the functionality of `layer_masks`."
                "Simply pass a list of masks to `sparsity_masks`, `ei_masks`, and `plasticity_masks` and the model will generate the masks accordingly.", UserWarning, stacklevel=slevel)
        
        for key in kwargs:
            if not key in param_list and not key in dep_param_list:
                print("unrecognized parameter: {}".format(key))

    def _check_masks(self, param, param_type, dims):
        """ General function to check different parameter types. """
        target_dim = [(dims[0], dims[1]), (dims[1], dims[1]), (dims[1], dims[2])]
        target_dim_biases = [dims[1], dims[1], dims[2]]
 
        # Handle None cases
        if param is None:
            if param_type in ["ei_masks", "sparsity_masks", "plasticity_masks", "biases"]:
                param = [None] * 3
            else: raise ValueError(f"{param_type} cannot be None when param_type is {param_type}") # weights
        elif param is not None and type(param) != list and param_type in ["weights"]:
            param = [param] * 3

        if type(param) != list:
            if param_type in ["ei_masks", "sparsity_masks", "plasticity_masks"]:
                raise ValueError(f"{param_type} must be a list of 3 values")
            else:
                param = [param] * 3
        if len(param) != 3:
            raise ValueError(f"{param_type} is/can not be broadcasted to a list of length 3")

        # param_type are all legal because it is passed by non-user code
        if param_type == "plasticity_masks": param = self._regularize_plas_masks(param, target_dim)
        else:
            # if its not plasticity_masks, then it must be a list of 3 values
            for i in range(3):
                if param[i] is not None:
                    if param_type in ["ei_masks", "sparsity_masks"]:
                        param[i] = self._check_array(param[i], param_type, target_dim[i], i)
                        if param_type == "ei_masks":
                            param[i] = torch.where(param[i] > 0, torch.tensor(1), torch.tensor(-1))
                        elif param_type == "sparsity_masks":
                            param[i] = torch.where(param[i] == 0, torch.tensor(0), torch.tensor(1))
                    elif param_type in ["weights", "biases"]:
                        self._check_distribution_or_array(param[i], param_type, target_dim_biases[i] if param_type == "biases" else target_dim[i], i)
        return param

    def _regularize_plas_masks(self, masks, target_dim):
        if any(mask is not None for mask in masks):
            min_plas, max_plas = [], []
            for mask in masks:
                if mask is not None:
                    min_plas.append(mask.min())
                    max_plas.append(mask.max())
            min_plas, max_plas = min(min_plas), max(max_plas)
            if min_plas != max_plas:
                params = []
                for i in range(3):
                    if masks[i] is None: params.append(torch.ones(target_dim[i]))
                    else:
                        _temp_mask = (masks[i] - min_plas) / (max_plas - min_plas)
                        params.append(self._check_array(_temp_mask, "plasticity_masks", target_dim[i], i))
                # check the total number of unique plasticity values
                plasticity_scales = torch.unique(torch.cat([param.flatten() for param in params]))
                if len(plasticity_scales) > 5:
                    raise ValueError("The number of unique plasticity values cannot be larger than 5")
                return params
        return [torch.ones(target_dim[i]) for i in range(3)]

    def _check_array(self, param, param_type, dim, index):
        if type(param) != torch.Tensor:
            if type(param) == np.ndarray:
                param = torch.from_numpy(param)
            else: raise ValueError(f"{param_type}[{index}] must be a numpy array or torch tensor")
        if param.shape != dim:
            raise ValueError(f"{param_type}[{index}] must be a numpy array of shape {dim}")
        return param

    def _check_distribution_or_array(self, param, param_type, dim, index):
        if type(param) == str:
            if param not in ['uniform', 'normal', 'zero']:
                raise ValueError(f"{param_type}[{index}] must be a string of 'uniform' or 'normal'")
        elif type(param) == torch.Tensor:
            # its already being converted to torch.Tensor, so no need to check np.ndarray case
            if param.shape != dim:
                raise ValueError(f"{param_type}[{index}] must be a numpy array of shape {dim}")
        else:
            raise ValueError(f"{param_type}[{index}] must be a string of 'uniform' or 'normal' \
                or a numpy array/torch tensor with shape {dim}")

    def _check_parameters(self):
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

    def _build_structures(self, kwargs):
        """ Build structures """
        # build structures
        rc_struct = {
            "init_state": kwargs.pop("init_state", 'zero'),
            "activation": kwargs.pop("activation", "relu"),
            "dt": kwargs.pop("dt", 10),
            "tau": kwargs.pop("tau", 100),
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
    def to(self, device):
        """ Move the network to device """
        super().to(device)
        self.recurrent_layer.to(device)
        self.readout_layer.to(device)

    def forward(self, x):
        """ 
        Forwardly update network

        Inputs:
            - x: input, shape: (n_timesteps, batch_size, input_dim)
        """
        # skip constraints if the model is not in training mode
        if self.training:
            self._enforce_constraints()
        hidden_states = self.recurrent_layer(x)
        output = self.readout_layer(hidden_states.float())
        return output, [hidden_states]

    def train(self):
        """
        Set pre-activation and post-activation noise to the specified value
        and resume enforcing constraints
        """
        self.recurrent_layer.preact_noise = self.preact_noise
        self.recurrent_layer.postact_noise = self.postact_noise
        self.training = True

    def eval(self):
        """ 
        Set pre-activation and post-activation noise to zero 
        and pause enforcing constraints
        """
        self.recurrent_layer.preact_noise = 0
        self.recurrent_layer.postact_noise = 0
        self.training = False

    def apply_plasticity(self):
        """
        Apply plasticity to the weight gradient such that the weights representing the synapses
        will update at different rates according to the plasticity mask.
        """
        # no need to consider the case where plasticity_mask is None as 
        # it will be automatically converted to a tensor of ones in parameter initialization
        self.recurrent_layer.apply_plasticity()
        self.readout_layer.apply_plasticity()

    def _enforce_constraints(self):
        self.recurrent_layer.enforce_constraints()
        self.readout_layer.enforce_constraints()

    # HELPER FUNCTIONS
    # ======================================================================================
    def print_layers(self):
        """ Print the specs of each layer """
        self.recurrent_layer.print_layers()
        self.readout_layer.print_layers()

    def plot_layers(self):
        """ Plot the weights matrix and distribution of each layer """
        self.recurrent_layer.plot_layers()
        self.readout_layer.plot_layers()
    # ======================================================================================
