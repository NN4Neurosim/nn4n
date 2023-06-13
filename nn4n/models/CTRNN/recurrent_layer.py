import torch
import numpy as np
import torch.nn as nn
from nn4n.utils import *

from .hidden_layer import HiddenLayer
from .linear_layer import LinearLayer

class RecurrentLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            use_dale,
            new_synapse,
            allow_negative,
            ei_balance,
            layer_distributions,
            layer_biases,
            layer_masks,
            **kwargs
        ):
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden neurons
            @param use_dale: use dale's law or not
            @param new_synapse: use new_synapse or not
            @param allow_negative: allow negative weights or not, a list of 3 boolean values
            @param ei_balance: method to balance e/i connections, based on number of neurons or number of synapses
            @param layer_distributions: distribution of weights for each layer, a list of 3 strings
            @param layer_biases: use bias or not for each layer, a list of 3 boolean values

        Keyword Arguments:
            @kwarg activation: activation function, default: "relu"
            @kwarg recurrent_noise: noise for recurrent connections, default: 0.05
            @kwarg dt: time step, default: 1
            @kwarg tau: time constant, default: 1
            @kwarg ei_balance: method to balance E/I neurons, default: "neuron"

            @kwarg input_size: number of input neurons, default: 1

            @kwarg hidden_dist: distribution of hidden layer weights, default: "normal"
            @kwarg self_connections: allow self connections or not, default: False
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.use_dale = use_dale
        self.set_activation(kwargs.get("activation", "relu"))
        self.recurrent_noise = kwargs.get("recurrent_noise", 0.05)
        self.alpha = kwargs.get("dt", 1) / kwargs.get("tau", 1)
        self.ei_balance = ei_balance
        self.layer_distributions = layer_distributions
        self.layer_biases = layer_biases
        self.layer_masks = layer_masks

        self.input_layer = LinearLayer(
            use_dale = False,
            new_synapse = new_synapse[0],
            output_size = self.hidden_size,
            input_size = kwargs.get("input_size", 1),
            use_bias = self.layer_biases[0],
            dist = self.layer_distributions[0],
            mask = self.layer_masks[0],
            allow_negative = allow_negative[0],
            ei_balance = self.ei_balance,
        )
        self.hidden_layer = HiddenLayer(
            hidden_size = self.hidden_size,
            new_synapse = new_synapse[1],
            use_dale = self.use_dale,
            dist = self.layer_distributions[1],
            use_bias = self.layer_biases[1],
            spec_rad = kwargs.get("spec_rad", 1),
            mask = self.layer_masks[1],
            self_connections = kwargs.get("self_connections", False),
            allow_negative = allow_negative[1],
            ei_balance = self.ei_balance,
        )

    
    def set_activation(self, act):
        self.act = act
        if self.act == "relu":
            self.activation = torch.relu
        elif self.act == "tanh":
            self.activation = torch.tanh
        elif self.act == "sigmoid":
            self.activation = torch.sigmoid
    

    def recurrence(self, input, hidden):
        """
        Hidden layer updates 
        @param input: shape=(batch_size, 4)
        @param hidden: hidden layer of the CTRNN
        """
        hidden_out = self.hidden_layer(self.activation(hidden)) # r(t) @ W_rec + b
        # print(hidden_out.mean(), hidden_out.max(), hidden_out.min())
        new_input = self.input_layer(input) # u(t) @ W_in
        if self.recurrent_noise > 0:
            noise = torch.from_numpy(np.random.normal(0, self.recurrent_noise, self.hidden_size))
            hidden_new = (1-self.alpha)*hidden + self.alpha*(hidden_out+new_input+noise)
        else:
            hidden_new = (1-self.alpha)*hidden + self.alpha*(hidden_out+new_input)

        return hidden_new


    def forward(self, input, hidden=None):
        """
        Propogate input through the network.
        @param input: shape=(seq_len, batch, input_size), network input
        @return output: shape=(seq_len, batch, hidden_size), stack of hidden layer status
        @return hidden: shape=(batch, hidden_size), hidden layer final status
        """
        # initialize hidden
        # if hidden is None:
        hidden = torch.zeros(self.hidden_size)

        # update hidden and append to output
        output = []
        for i in range(input.size(0)):
            hidden = self.recurrence(input[i], hidden)
            output.append(hidden)
        output = torch.stack(output, dim=0)
        
        return output, hidden
    

    def print_layer(self):
        param_dict = {
            "recurrent_noise": self.recurrent_noise,
            "activation": self.act,
            "alpha": self.alpha,
        }
        self.input_layer.print_layer()
        print_dict("Recurrence", param_dict)
        self.hidden_layer.print_layer()


    def enforce_constraints(self):
        self.input_layer.enforce_constraints()
        self.hidden_layer.enforce_constraints()