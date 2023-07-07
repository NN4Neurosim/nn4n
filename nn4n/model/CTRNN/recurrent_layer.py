import torch
import torch.nn as nn
from nn4n.utils import *

from .hidden_layer import HiddenLayer
from .linear_layer import LinearLayer

class RecurrentLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            use_dale,
            new_synapses,
            allow_negative,
            ei_balance,
            layer_distributions,
            layer_biases,
            layer_masks,
            recurrent_noise,
            **kwargs
        ):
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden neurons
            @param use_dale: use dale's law or not
            @param new_synapses: use new_synapses or not
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

            @kwarg input_dim: input dimension, default: 1

            @kwarg hidden_dist: distribution of hidden layer weights, default: "normal"
            @kwarg self_connections: allow self connections or not, default: False
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.use_dale = use_dale
        self._set_activation(kwargs.get("activation", "relu"))
        self.recurrent_noise = recurrent_noise
        self.alpha = kwargs.get("dt", 1) / kwargs.get("tau", 1)
        self.ei_balance = ei_balance
        self.layer_distributions = layer_distributions
        self.layer_biases = layer_biases
        self.layer_masks = layer_masks
        self.hidden_state = torch.zeros(self.hidden_size)

        self.input_layer = LinearLayer(
            use_dale = self.use_dale,
            new_synapses = new_synapses[0],
            output_dim = self.hidden_size,
            input_dim = kwargs.get("input_dim", 1),
            use_bias = self.layer_biases[0],
            dist = self.layer_distributions[0],
            mask = self.layer_masks[0],
            allow_negative = allow_negative[0],
            ei_balance = self.ei_balance,
        )
        self.hidden_layer = HiddenLayer(
            hidden_size = self.hidden_size,
            new_synapses = new_synapses[1],
            use_dale = self.use_dale,
            dist = self.layer_distributions[1],
            use_bias = self.layer_biases[1],
            scaling = kwargs.get("scaling", 1.0),
            mask = self.layer_masks[1],
            self_connections = kwargs.get("self_connections", False),
            allow_negative = allow_negative[1],
            ei_balance = self.ei_balance,
        )


    # INITIALIZATION
    # ==================================================================================================
    def _set_activation(self, act):
        self.act = act
        if self.act == "relu":
            self.activation = torch.relu
        elif self.act == "tanh":
            self.activation = torch.tanh
        elif self.act == "sigmoid":
            self.activation = torch.sigmoid
        elif self.act == "retanh":
            self.activation = lambda x: torch.maximum(torch.tanh(x),torch.tensor(0))
    # ==================================================================================================


    # FORWARD
    # ==================================================================================================
    def reset_state(self):
        self.hidden_state = torch.zeros(self.hidden_size, device=self.hidden_state.device)


    def recurrence(self, input, hidden):
        """
        Hidden layer updates 
        @param input: shape=(batch_size, 4)
        @param hidden: hidden layer of the CTRNN
        """
        hidden_out = self.hidden_layer(hidden) # r(t) @ W_hid + b
        new_input = self.input_layer(input) # u(t) @ W_in
        if self.recurrent_noise > 0:
            noise = torch.randn(self.hidden_size, device=hidden.device) * self.recurrent_noise
            hidden_new = (1-self.alpha)*hidden + self.alpha*(hidden_out+new_input+noise)
        else:
            hidden_new = (1-self.alpha)*hidden + self.alpha*(hidden_out+new_input)

        return self.activation(hidden_new)


    def enforce_constraints(self):
        self.input_layer.enforce_constraints()
        self.hidden_layer.enforce_constraints()


    def forward(self, input):
        """
        Propogate input through the network.
        @param input: shape=(seq_len, batch, input_dim), network input
        @return stacked_states: shape=(seq_len, batch, hidden_size), stack of hidden layer status
        """
        
        hidden_state = torch.zeros(self.hidden_size, device=input.device)
        # update hidden state and append to stacked_states
        stacked_states = []
        for i in range(input.size(0)):
            hidden_state = self.recurrence(input[i], hidden_state)
            stacked_states.append(hidden_state)
        stacked_states = torch.stack(stacked_states, dim=0)
        
        return stacked_states
    # ==================================================================================================


    # HELPER FUNCTIONS
    # ==================================================================================================
    def to(self, device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        self.input_layer.to(device)
        self.hidden_layer.to(device)
        self.hidden_state = self.hidden_state.to(device)


    def print_layer(self):
        param_dict = {
            "recurrent_noise": self.recurrent_noise,
            "activation": self.act,
            "alpha": self.alpha,
        }
        self.input_layer.print_layer()
        print_dict("Recurrence", param_dict)
        self.hidden_layer.print_layer()
    # ==================================================================================================
