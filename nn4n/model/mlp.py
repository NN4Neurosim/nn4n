import torch

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
        assert False, "MLP is deprecated for now"
        super().__init__(**kwargs)

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
        self._check_parameters(kwargs)
        ctrnn_structs = self._build_structures(kwargs)

        # layers
        self.recurrent_layer = RecurrentLayer(layer_struct=ctrnn_structs[0])
        self.readout_layer = LinearLayer(layer_struct=ctrnn_structs[1])
        # self.input_dim = kwargs.pop("input_dim", 1)
        # self.output_dim = kwargs.pop("output_dim", 1)
        # self.hidden_size = kwargs.pop("hidden_size", 100)
        # self.dims = kwargs.pop("dims", [self.input_dim, self.hidden_size, self.output_dim])
        # self.learnable = kwargs.pop("learnable", [True, True])
        # self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'uniform'])
        # self.layer_biases = kwargs.pop("layer_biases", [True, True])
        # # dynamics parameters
        # self.preact_noise = kwargs.pop("preact_noise", 0)
        # self.postact_noise = kwargs.pop("postact_noise", 0)
        # self.layer_masks = kwargs.pop("layer_masks", [None, None])

        # if len(self.layer_distributions) == 3:
        #     self.layer_distributions.pop(1)
        # if len(self.layer_biases) == 3:
        #     self.layer_biases.pop(1)

        # self.act = kwargs.pop("act", "relu")
        # self.activation = get_activation(self.act)

        self.input_layer = LinearLayer(
            input_dim=self.input_dim,
            output_dim=self.hidden_size,
            dist=self.layer_distributions[0],
            use_bias=self.layer_biases[0],
            mask=self.layer_masks[0],
            positivity_constraints=False,
            sparsity_constraints=True if self.layer_masks[0] is None else False,
            learnable=self.learnable[0],
        )

        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            output_dim=self.output_dim,
            dist=self.layer_distributions[1],
            use_bias=self.layer_biases[1],
            mask=self.layer_masks[0],
            positivity_constraints=False,
            sparsity_constraints=True if self.layer_masks[1] is None else False,
            learnable=self.learnable[1],
        )

    def _handle_warnings(self, kwargs):
        """ Handle deprecated parameters """


    def forward(self, input):
        """
        Inputs:
            - x: size=(batch_size, input_dim)
        """
        v = self.input_layer(input)
        if self.preact_noise > 0:
            v = v + torch.randn_like(v) * self.preact_noise
        fr = self.activation(v)
        if self.postact_noise > 0:
            fr = fr + torch.randn_like(fr) * self.postact_noise
        v = self.readout_layer(fr)
        return v, fr

    def print_layers(self):
        """
        Print the layer information
        """
        self.input_layer.print_layers()
        self.readout_layer.print_layers()