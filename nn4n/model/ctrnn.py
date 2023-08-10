import torch
import torch.nn as nn

from nn4n.model import BaseNN
from nn4n.layer import RecurrentLayer
from nn4n.layer import LinearLayer


class CTRNN(BaseNN):
    """ Recurrent network model """
    def __init__(self, **kwargs):
        """
        Base RNN constructor
        Keyword Arguments:
            @kwarg use_dale: use dale's law or not
            @kwarg new_synapses: use new_synapses or not
            @kwarg hidden_size: number of hidden neurons
            @kwarg output_dim: output dimension
            @kwarg allow_negative: allow negative weights or not
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
        # structure parameters
        self.hidden_size = kwargs.pop("hidden_size")
        self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'normal', 'uniform'])
        self.layer_biases = kwargs.pop("layer_biases", [True, True, True])
        # dynamics parameters
        self.use_dale = kwargs.pop("use_dale", False)
        self.new_synapses = kwargs.pop("new_synapses", True)
        self.allow_negative = kwargs.pop("allow_negative", True)
        self.layer_masks = kwargs.pop("layer_masks", [None, None, None])
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

        if 'ei_balance' in kwargs:
            print("WARNING: ei_balance is deprecated. No ei_balance specification is needed.")

        # check if all parameters meet the requirements
        self._check_parameters()

        # layers
        self.recurrent_layer = RecurrentLayer(
            hidden_size=self.hidden_size,
            use_dale=self.use_dale,
            new_synapses=self.new_synapses,
            allow_negative=self.allow_negative,
            layer_distributions=self.layer_distributions,
            layer_biases=self.layer_biases,
            layer_masks=self.layer_masks,
            preact_noise=self.preact_noise,
            postact_noise=self.postact_noise,
            **kwargs
        )
        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            use_dale=self.use_dale,
            output_dim=kwargs.pop("output_dim"),
            dist=self.layer_distributions[2],
            use_bias=self.layer_biases[2],
            mask=self.layer_masks[2],
            new_synapses=self.new_synapses[2],
            allow_negative=self.allow_negative[2],
        )

        # if using dale's law
        if self.use_dale:
            hidden_ei_list = self.recurrent_layer.hidden_layer.ei_list
            readout_ei_list = self.readout_layer.ei_list
            nonzero_idx = torch.nonzero(readout_ei_list).squeeze()
            assert torch.all(hidden_ei_list[nonzero_idx] == readout_ei_list[nonzero_idx]), \
                "ei_list of hidden layer and readout layer must be the same when use_dale is True"

    def _check_parameters(self):
        """ Check parameters """
        # check use_dale
        assert type(self.use_dale) == bool, "use_dale must be a boolean"

        # check hidden_size
        assert type(self.hidden_size) == int, "hidden_size must be an integer"
        assert self.hidden_size > 0, "hidden_size must be a positive integer"

        # check new_synapses
        assert type(self.new_synapses) == bool, "new_synapses must be a boolean"

        # check allow_negative
        if type(self.allow_negative) == bool:
            self.allow_negative = [self.allow_negative] * 3
        elif type(self.allow_negative) == list:
            assert len(self.allow_negative) == 3, "allow_negative must be a list of length 3"
            for i in self.allow_negative:
                assert type(i) == bool, "allow_negative must be a list of booleans"
        else:
            raise ValueError("allow_negative must be a boolean or a list of booleans")
        if self.use_dale and self.allow_negative != [False, False, False]:
            print("WARNING: allow_negative is ignored because use_dale is set to True")
            self.allow_negative = [False, False, False]

        # check new_synapses
        if type(self.new_synapses) == bool:
            self.new_synapses = [self.new_synapses] * 3
        elif type(self.new_synapses) == list:
            assert len(self.new_synapses) == 3, "new_synapses must be a list of length 3"
            for i in self.new_synapses:
                assert type(i) == bool, "new_synapses must be a list of booleans"
        else:
            raise ValueError("new_synapses must be a boolean or a list of booleans")
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
        self.recurrent_layer.print_layer()
        self.readout_layer.print_layer()
    # ======================================================================================
