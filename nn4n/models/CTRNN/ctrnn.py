import torch
import torch.nn as nn
import numpy as np

from .recurrent_layer import RecurrentLayer
from .linear_layer import LinearLayer

class CTRNN(nn.Module):
    """ Recurrent network model """
    def __init__(self, **kwargs):
        """
        Base RNN constructor
        Keyword Arguments:
            @kwarg use_dale: use dale's law or not
            @kwarg new_synapse: use new_synapse or not
            @kwarg hidden_size: number of hidden neurons
            @kwarg allow_negative: allow negative weights or not
            @kwarg training: training or not

            @kwarg output_size: number of output neurons
            @kwarg output_dist: distribution of output layer weights
            @kwarg output_bias: use bias for output layer or not
            @kwarg output_mask: mask for output layer, optional
        """
        super().__init__()
        self.kwargs_checkpoint = kwargs.copy()
        self.initialize(**kwargs)


    def initialize(self, **kwargs):
        """
        Initialize/Reinitialize the network
        """
        # parameters that used in all layers
        self.use_dale = kwargs.pop("use_dale", False)
        self.new_synapse = kwargs.pop("new_synapse", True)
        self.hidden_size = kwargs.pop("hidden_size", 100)
        self.allow_negative = kwargs.pop("allow_negative", [True, True, True])
        self.training = kwargs.get("training", True)

        self.check_params()

        # layers
        self.recurrent = RecurrentLayer(
            hidden_size = self.hidden_size,
            use_dale = self.use_dale,
            new_synapse = self.new_synapse,
            allow_negative = self.allow_negative,
            **kwargs
        )
        self.readout_layer = LinearLayer(
            input_size = self.hidden_size,
            use_dale = self.use_dale,
            output_size = kwargs.get("output_size", 1),
            dist = kwargs.get("output_dist", "uniform"),
            use_bias = kwargs.get("output_bias", False),
            mask = kwargs.get("output_mask", None),
            new_synapse = self.new_synapse,
            allow_negative = self.allow_negative[2]
        )

        if self.use_dale:
            hidden_ei_list = self.recurrent.hidden_layer.ei_list
            readout_ei_list = self.readout_layer.ei_list
            nonzero_idx = np.where(readout_ei_list != 0)[0]
            assert np.all(hidden_ei_list[nonzero_idx] == readout_ei_list[nonzero_idx]), "ei_list of hidden layer and readout layer must be the same when use_dale is True"


    def check_params(self):
        """
        Check parameters
        """
        ## check use_dale
        assert type(self.use_dale) == bool, "use_dale must be a boolean"

        ## check allow_negative
        assert len(self.allow_negative) == 3, "allow_negative must be a list of length 3"
        for i in self.allow_negative:
            assert type(i) == bool, "allow_negative must be a list of booleans"
        if self.use_dale and self.allow_negative != [False, False, False]:
            print("WARNING: allow_negative is ignored because use_dale is set to True")
            self.allow_negative = [False, False, False]

        ## check hidden_size
        assert type(self.hidden_size) == int, "hidden_size must be an integer"
        assert self.hidden_size > 0, "hidden_size must be a positive integer"

        ## check new_synapse
        assert type(self.new_synapse) == bool, "new_synapse must be a boolean"


    def forward(self, x):
        """
        Forwardly update network W_in -> n x W_rc -> W_out
        """
        # skip constraints if not training
        if self.training:
            self.enforce_constraints()
        hidden_activity, _ = self.recurrent(x)
        output = self.readout_layer(hidden_activity.float())
        return output, hidden_activity


    def print_layers(self):
        self.recurrent.print_layer()
        self.readout_layer.print_layer()


    def enforce_constraints(self):
        self.recurrent.enforce_constraints()
        self.readout_layer.enforce_constraints()

    def save(self, path):
        # save model and kwargs to the same file
        assert type(path) == str, "path must be a string"
        assert path[-4:] == ".pth", "path must end with .pth"
        torch.save({
            "model_state_dict": self.state_dict(),
            "kwargs": self.kwargs_checkpoint
        }, path)

    def load(self, path):
        # load model and kwargs from the same file
        assert type(path) == str, "path must be a string"
        assert path[-4:] == ".pth" or path[-3:] == ".pt", "path must end with .pth or .pt"
        checkpoint = torch.load(path)
        self.kwargs_checkpoint = checkpoint["kwargs"]
        self.initialize(**self.kwargs_checkpoint)
        self.load_state_dict(checkpoint["model_state_dict"])
