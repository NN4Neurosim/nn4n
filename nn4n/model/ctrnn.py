import torch
import torch.nn as nn

from nn4n.model import BaseNN
from nn4n.layer import RecurrentLayer
from nn4n.layer import LinearLayer

param_list = ['input_dim', 'output_dim', 'hidden_size', 'positivity_constraints', 
              'sparsity_constraints', 'layer_distributions', 'layer_biases', 
              'layer_masks', 'preact_noise', 'postact_noise', 'learnable', 
              'init_state', 'activation', 'scaling', 'tau', 'dt', 'self_connections']

class CTRNN(BaseNN):
    """ Recurrent network model """
    def __init__(self, **kwargs):
        """
        Base RNN constructor
        Keyword Arguments:
            @kwarg positivity_constraints: whether to enforce positivity constraint
            @kwarg sparsity_constraints: use sparsity_constraints or not
            @kwarg hidden_size: number of hidden neurons
            @kwarg output_dim: output dimension
            @kwarg layer_masks: masks for each layer, a list of 3 masks
            @kwarg layer_distributions: distribution of weights for each layer, a list of 3 strings
            @kwarg layer_biases: use bias or not for each layer, a list of 3 boolean values
        """
        super().__init__(**kwargs)

    # INITIALIZATION
    # ======================================================================================
    def _initialize(self, **kwargs):
        """ Initialize/Reinitialize the network """
        # parameters that used in all layers
        # base parameters
        self.hidden_size = kwargs.pop("hidden_size", 100)
        self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'normal', 'uniform'])
        self.layer_biases = kwargs.pop("layer_biases", [True, True, True])
        # network dynamics parameters
        self.positivity_constraints = kwargs.pop("positivity_constraints", [False, False, False])
        self.sparsity_constraints = kwargs.pop("sparsity_constraints", [False, False, False])
        self.layer_masks = kwargs.pop("layer_masks", [None, None, None])
        self.learnable = kwargs.pop("learnable", [True, True, True])
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

        if 'ei_balance' in kwargs:
            print("WARNING: ei_balance is deprecated. No ei_balance specification is needed.")
        if 'allow_negative' in kwargs:
            print("WARNING: allow_negative is deprecated. No allow_negative specification is needed.")
        if 'use_dale' in kwargs:
            print("WARNING: use_dale is deprecated. Use positivity_constraints instead.")
            self.positivity_constraints = kwargs.pop("use_dale")
        if 'new_synapses' in kwargs:
            print("WARNING: new_synapses is deprecated. Use sparsity_constraints instead.")
            self.sparsity_constraints = kwargs.pop("new_synapses")

        # check if all parameters meet the requirements
        self._check_parameters(kwargs)

        # layers
        self.recurrent_layer = RecurrentLayer(
            hidden_size=self.hidden_size,
            positivity_constraints=self.positivity_constraints,
            sparsity_constraints=self.sparsity_constraints,
            layer_distributions=self.layer_distributions,
            layer_biases=self.layer_biases,
            layer_masks=self.layer_masks,
            preact_noise=self.preact_noise,
            postact_noise=self.postact_noise,
            learnable=self.learnable,
            **kwargs
        )
        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            positivity_constraints=self.positivity_constraints[2],
            output_dim=kwargs.pop("output_dim", 1),
            dist=self.layer_distributions[2],
            use_bias=self.layer_biases[2],
            mask=self.layer_masks[2],
            sparsity_constraints=self.sparsity_constraints[2],
            learnable=self.learnable[2],
        )

    def _check_parameters(self, kwargs):
        """ Check parameters """
        # check positivity_constraints
        if type(self.positivity_constraints) == bool:
            self.positivity_constraints = [self.positivity_constraints] * 3
        elif type(self.positivity_constraints) == list:
            assert len(self.positivity_constraints) == 3, "positivity_constraints must be a list of length 3"
            for i in self.positivity_constraints:
                assert type(i) == bool, "positivity_constraints must be a list of booleans"
        else:
            raise ValueError("positivity_constraints must be a boolean or a list of booleans")

        # check hidden_size
        assert type(self.hidden_size) == int, "hidden_size must be an integer"
        assert self.hidden_size > 0, "hidden_size must be a positive integer"

        # check sparsity_constraints
        if type(self.sparsity_constraints) == bool:
            self.sparsity_constraints = [self.sparsity_constraints] * 3
        elif type(self.sparsity_constraints) == list:
            assert len(self.sparsity_constraints) == 3, "sparsity_constraints must be a list of length 3"
            for i in self.sparsity_constraints:
                assert type(i) == bool, "sparsity_constraints must be a list of booleans"
        else:
            raise ValueError("sparsity_constraints must be a boolean or a list of booleans")
        
        # if layer_masks is not None, check if it is a list of 3 masks
        if self.layer_masks is not None:
            assert type(self.layer_masks) == list, "layer_masks must be a list of 3 masks"
            assert len(self.layer_masks) == 3, "layer_masks must be a list of 3 masks"
            # check if either positivity_constraints or any of the sparsity_constraints is False
            if not any(self.sparsity_constraints) and not any(self.positivity_constraints):
                print("WARNING: layer_masks is ignored because neither positivity_constraints nor sparsity_constraints is set to True")
        
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
