import torch
import torch.nn as nn

from recurrent_layer import RecurrentLayer
from linear_layer import LinearLayer

class EIRNN(nn.Module):
    """ Recurrent network model """
    def __init__(self, **kwargs):
        """
        Base RNN constructor
        Keyword Arguments:
            @kwarg use_dale: use dale's law or not, default: True
            @kwarg plasticity: use plasticity or not, default: False
            @kwarg hidden_size: number of hidden neurons, default: 100

            @kwarg output_size: number of output neurons, default: 1
            @kwarg output_dist: distribution of output layer weights, default: "uniform"
            @kwarg output_bias: use bias for output layer or not, default: False
            @kwarg output_mask: mask for output layer, optional, default: None
        """
        super().__init__()
        # parameters that used in all layers
        self.use_dale = kwargs.get("use_dale", True)
        self.plasticity = kwargs.get("plasticity", False)
        self.hidden_size = kwargs.get("hidden_size", 100)

        # layers
        self.recurrent = RecurrentLayer(
            hidden_size = self.hidden_size,
            use_dale = self.use_dale,
            plasticity = self.plasticity,
            **kwargs
        )
        self.readout_layer = LinearLayer(
            input_size = self.hidden_size,
            output_size = kwargs.get("output_size", 1),
            dist = kwargs.get("output_dist", "uniform"),
            use_bias = kwargs.get("output_bias", False),
            mask = kwargs.get("output_mask", None),
            use_dale = kwargs.get("use_dale", True),
            plasticity = kwargs.get("plasticity", False),
        )


    def forward(self, x):
        """
        Forwardly update network W_in -> n x W_rc -> W_out
        """
        hidden_activity, _ = self.recurrent(x)
        output = self.readout_layer(hidden_activity.float())
        return output, hidden_activity


    def print_layer(self):
        self.recurrent.print_layer()
        self.readout_layer.print_layer()


    def enforce_mask(self):
        self.recurrent.enforce_mask()
        self.readout_layer.enforce_mask()