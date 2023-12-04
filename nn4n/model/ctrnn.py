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
            @kwarg positivity_constraint: whether to enforce positivity constraint
            @kwarg sparsity_constraint: use sparsity_constraint or not
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
        # structure parameters
        self.hidden_size = kwargs.pop("hidden_size", 100)
        self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'normal', 'uniform'])
        self.layer_biases = kwargs.pop("layer_biases", [True, True, True])
        # dynamics parameters
        self.positivity_constraint = kwargs.pop("positivity_constraint", [False, False, False])
        self.sparsity_constraint = kwargs.pop("sparsity_constraint", [False, False, False])
        print(self.positivity_constraint, self.sparsity_constraint)
        self.layer_masks = kwargs.pop("layer_masks", [None, None, None])
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

        # TODO: add auto-compatibility
        if 'ei_balance' in kwargs:
            print("WARNING: ei_balance is deprecated. No ei_balance specification is needed.")
        if 'allow_negative' in kwargs:
            print("WARNING: allow_negative is deprecated. No allow_negative specification is needed.")
        if 'use_dale' in kwargs:
            print("WARNING: use_dale is deprecated. Use positivity_constraint instead.")
            self.positivity_constraint = kwargs.pop("use_dale")
        if 'new_synapses' in kwargs:
            print("WARNING: new_synapses is deprecated. Use sparsity_constraint instead.")
            self.sparsity_constraint = kwargs.pop("new_synapses")

        # check if all parameters meet the requirements
        self._check_parameters()

        # layers
        self.recurrent_layer = RecurrentLayer(
            hidden_size=self.hidden_size,
            positivity_constraint=self.positivity_constraint,
            sparsity_constraint=self.sparsity_constraint,
            layer_distributions=self.layer_distributions,
            layer_biases=self.layer_biases,
            layer_masks=self.layer_masks,
            preact_noise=self.preact_noise,
            postact_noise=self.postact_noise,
            **kwargs
        )
        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            positivity_constraint=self.positivity_constraint[2],
            output_dim=kwargs.pop("output_dim", 1),
            dist=self.layer_distributions[2],
            use_bias=self.layer_biases[2],
            mask=self.layer_masks[2],
            sparsity_constraint=self.sparsity_constraint[2]
        )

        # # if using dale's law, check if ei_list of hidden layer and readout layer are the same
        # if self.positivity_constraint:
        #     hidden_ei_list = self.recurrent_layer.hidden_layer.ei_list
        #     readout_ei_list = self.readout_layer.ei_list
        #     nonzero_idx = torch.nonzero(readout_ei_list).squeeze()
        #     assert torch.all(hidden_ei_list[nonzero_idx] == readout_ei_list[nonzero_idx]), \
        #         "ei_list of hidden layer and readout layer must be the same when positivity_constraint is True"

    def _check_parameters(self):
        """ Check parameters """
        # check positivity_constraint
        if type(self.positivity_constraint) == bool:
            self.positivity_constraint = [self.positivity_constraint] * 3
        elif type(self.positivity_constraint) == list:
            assert len(self.positivity_constraint) == 3, "positivity_constraint must be a list of length 3"
            for i in self.positivity_constraint:
                assert type(i) == bool, "positivity_constraint must be a list of booleans"
        else:
            raise ValueError("positivity_constraint must be a boolean or a list of booleans")

        # check hidden_size
        assert type(self.hidden_size) == int, "hidden_size must be an integer"
        assert self.hidden_size > 0, "hidden_size must be a positive integer"

        # check sparsity_constraint
        if type(self.sparsity_constraint) == bool:
            self.sparsity_constraint = [self.sparsity_constraint] * 3
        elif type(self.sparsity_constraint) == list:
            assert len(self.sparsity_constraint) == 3, "sparsity_constraint must be a list of length 3"
            for i in self.sparsity_constraint:
                assert type(i) == bool, "sparsity_constraint must be a list of booleans"
        else:
            raise ValueError("sparsity_constraint must be a boolean or a list of booleans")
        
        # if layer_masks is not None, check if it is a list of 3 masks
        if self.layer_masks is not None:
            assert type(self.layer_masks) == list, "layer_masks must be a list of 3 masks"
            assert len(self.layer_masks) == 3, "layer_masks must be a list of 3 masks"
            # TODO: re-write this part
            # check if either positivity_constraint or any of the sparsity_constraint is False
            # if not (self.positivity_constraint or not all(self.sparsity_constraint)):
            #     print("WARNING: layer_masks is ignored because positivity_constraint and sparsity_constraint are both set to False")
            #     self.layer_masks = [None, None, None]
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
