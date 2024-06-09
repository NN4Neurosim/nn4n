import torch
import torch.nn as nn

from nn4n.model import BaseNN
from nn4n.layer import LinearLayer
from nn4n.utils import get_activation


class MLP(BaseNN):
    """
    Multi layer perceptron

    Keyword Arguments:
        - dims: dimensions of the network, default: [1, 100, 1]
        - preact_noise: noise added to pre-activation, default: 0
        - postact_noise: noise added to post-activation, default: 0
        - activation: activation function, default: "relu"
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
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, **kwargs):
        """ Initialize/Reinitialize the network """
        super()._initialize(**kwargs)
        # parameters that used in all layers
        # base parameters
        self.dims = kwargs.pop("dims", [10, 10])
        self.biases = kwargs.pop("biases", None)
        self.weights = kwargs.pop("weights", 'uniform')
        self.activation = get_activation(kwargs.pop("activation", "relu"))

        # network dynamics parameters
        self.sparsity_masks = kwargs.pop("sparsity_masks", None)
        self.ei_masks = kwargs.pop("ei_masks", None)
        self.plasticity_masks = kwargs.pop("plasticity_masks", None)

        # noise parameters
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

        # suppress warnings
        self.suppress_warnings = kwargs.pop("suppress_warnings", False)

        # check if all parameters meet the requirements
        if not self.suppress_warnings:
            self._handle_warnings(kwargs)
        self._check_parameters()

        self._init_layers()

    def _init_layers(self):
        self.layers = nn.ModuleList()
        for i in range(len(self.dims)-1):
            layer_struct = {
                "input_dim": self.dims[i],
                "output_dim": self.dims[i+1],
                "weights": self.weights[i],
                "biases": self.biases[i],
                "sparsity_mask": self.sparsity_masks[i],
                "ei_mask": self.ei_masks[i],
                "plasticity_mask": self.plasticity_masks[i],
            }
            l = LinearLayer(layer_struct=layer_struct)
            self.layers.append(l)

    def _handle_warnings(self, kwargs):
        """ Handle deprecated parameters """
        pass

    def _check_parameters(self):
        """ Check if the parameters meet the requirements """
        # check if dims is a list
        if not isinstance(self.dims, list):
            raise ValueError("dims must be a list of integers")
        n_weights = len(self.dims)

        self.biases = self._broadcast_values(self.biases, n_weights)
        self.weights = self._broadcast_values(self.weights, n_weights)
        self.sparsity_masks = self._broadcast_values(self.sparsity_masks, n_weights, is_mask=True)
        self.ei_masks = self._broadcast_values(self.ei_masks, n_weights, is_mask=True)
        self.plasticity_masks = self._broadcast_values(self.plasticity_masks, n_weights, is_mask=True)
        self.plasticity_masks = self._regularize_plas_masks(self.plasticity_masks, self.dims)
        self.preact_noise = self._broadcast_values(self.preact_noise, n_weights-1)  # no noise for the readout & input layer
        self.postact_noise = self._broadcast_values(self.postact_noise, n_weights-1)  # no noise for the readout & input layer

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

    def _standardize_masks(self, masks):
        assert masks is not None, "Masks cannot be None"
        if len(masks) != len(self.dims):
            raise ValueError("The length of the mask must be the same as the length of dims")

        for i in range(len(masks)):
            if masks[i] is None:
                continue
            if isinstance(masks[i], np.ndarray):
                masks[i] = torch.tensor(masks[i], dtype=torch.float32)
            if isinstance(masks[i], torch.Tensor):
                masks[i] = masks[i].float()
                # check shape is the same as dims
                if masks[i].shape != (self.dims[i], self.dims[i+1]):
                    raise ValueError(f"Mask shape mismatch, expected: {(self.dims[i], self.dims[i+1])}, got: {masks[i].shape}")
    
    def apply_plasticity(self):
        pass

    @staticmethod
    def _broadcast_values(value, length, is_mask=False):
        """ Broadcast a single value to a list if it's not already a list """
        if not isinstance(value, list):
            if is_mask and value != None:
                raise ValueError("Mask value cannot be auto-broadcasted")
            return [value] * length
        else:
            # check if the length of the list is correct
            if len(value) != length:
                raise ValueError(f"Expected a list of length {length}, got a list of length {len(value)}")
            return value

    def forward(self, x):
        """
        Inputs:
            - x: size=(batch_size, input_dim)
        """
        hidden_states = []
        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                x = layer(x)
                break
            x = layer(x)
            x += torch.randn_like(x) * self.preact_noise[i]
            x = self.activation(x) + torch.randn_like(x) * self.postact_noise[i]
            hidden_states.append(x)
        
        return x, hidden_states

    def print_layers(self):
        """
        Print the layer information
        """
        for i, layer in enumerate(self.layers):
            layer.print_layers()

    def plot_layers(self):
        """
        Plot the layer information
        """
        for i, layer in enumerate(self.layers):
            layer.plot_layers()