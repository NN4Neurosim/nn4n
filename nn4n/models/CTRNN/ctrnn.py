import torch
import torch.nn as nn

from .recurrent_layer import RecurrentLayer
from .linear_layer import LinearLayer

class CTRNN(nn.Module):
    """ Recurrent network model """
    def __init__(self, **kwargs):
        """
        Base RNN constructor
        Keyword Arguments:
            @kwarg use_dale: use dale's law or not, default: True
            @kwarg plasticity: use plasticity or not, default: False
            @kwarg hidden_size: number of hidden neurons, default: 100
            @kwarg allow_neg: allow negative weights or not, default: [True, True, True]

            @kwarg output_size: number of output neurons, default: 1
            @kwarg output_dist: distribution of output layer weights, default: "uniform"
            @kwarg output_bias: use bias for output layer or not, default: False
            @kwarg output_mask: mask for output layer, optional, default: None
        """
        super().__init__()
        # parameters that used in all layers
        self.use_dale = kwargs.pop("use_dale", False)
        self.plasticity = kwargs.pop("plasticity", False)
        self.hidden_size = kwargs.pop("hidden_size", 100)
        self.allow_neg = kwargs.pop("allow_neg", [True, True, True])

        self.check_params()

        # layers
        self.recurrent = RecurrentLayer(
            hidden_size = self.hidden_size,
            use_dale = self.use_dale,
            plasticity = self.plasticity,
            allow_neg = self.allow_neg,
            **kwargs
        )
        self.readout_layer = LinearLayer(
            input_size = self.hidden_size,
            use_dale = self.use_dale,
            output_size = kwargs.get("output_size", 1),
            dist = kwargs.get("output_dist", "uniform"),
            use_bias = kwargs.get("output_bias", False),
            mask = kwargs.get("output_mask", None),
            plasticity = kwargs.get("plasticity", False),
            allow_neg = self.allow_neg[2]
        )

        if self.use_dale:
            assert self.recurrent.ei_list == self.readout_layer.ei_list, "E/I list of recurrent and readout layer must be the same"


    def check_params(self):
        """
        Check parameters
        """
        ## check allow_neg
        assert type(self.use_dale) == bool, "use_dale must be a boolean"

        ## check allow_neg
        assert len(self.allow_neg) == 3, "allow_neg must be a list of length 3"
        for i in self.allow_neg:
            assert type(i) == bool, "allow_neg must be a list of booleans"
        if self.use_dale and self.allow_neg != [False, False, False]:
            print("Warning: allow_neg is ignored because use_dale is set to True")
            self.allow_neg = [False, False, False]

        ## check hidden_size
        assert type(self.hidden_size) == int, "hidden_size must be an integer"
        assert self.hidden_size > 0, "hidden_size must be a positive integer"

        ## check plasticity
        assert type(self.plasticity) == bool, "plasticity must be a boolean"


    def forward(self, x):
        """
        Forwardly update network W_in -> n x W_rc -> W_out
        """
        hidden_activity, _ = self.recurrent(x)
        output = self.readout_layer(hidden_activity.float())
        return output, hidden_activity


    def print_layers(self):
        self.recurrent.print_layer()
        self.readout_layer.print_layer()


    def enforce_mask(self):
        self.recurrent.enforce_mask()
        self.readout_layer.enforce_mask()