import torch
import torch.nn as nn
from nn4n.utils import print_dict

from .hidden_layer import HiddenLayer
from .linear_layer import LinearLayer


class RecurrentLayer(nn.Module):
    def __init__(
            self,
            hidden_size,
            use_dale,
            new_synapses,
            allow_negative,
            layer_distributions,
            layer_biases,
            layer_masks,
            preact_noise,
            postact_noise,
            **kwargs
            ):
        """
        Hidden layer of the RNN
        Parameters:
            @param hidden_size: number of hidden neurons
            @param use_dale: use dale's law or not
            @param new_synapses: use new_synapses or not
            @param allow_negative: allow negative weights or not, a list of 3 boolean values
            @param layer_distributions: distribution of weights for each layer, a list of 3 strings
            @param layer_biases: use bias or not for each layer, a list of 3 boolean values

        Keyword Arguments:
            @kwarg activation: activation function, default: "relu"
            @kwarg preact_noise: noise added to pre-activation, default: 0
            @kwarg postact_noise: noise added to post-activation, default: 0
            @kwarg dt: time step, default: 1
            @kwarg tau: time constant, default: 1

            @kwarg input_dim: input dimension, default: 1

            @kwarg hidden_dist: distribution of hidden layer weights, default: "normal"
            @kwarg self_connections: allow self connections or not, default: False
            @kwarg init_state: initial state of the network, 'zero', 'keep', or 'learn'
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.use_dale = use_dale
        self.preact_noise = preact_noise
        self.postact_noise = postact_noise
        self.alpha = kwargs.get("dt", 10) / kwargs.get("tau", 100)
        self.layer_distributions = layer_distributions
        self.layer_biases = layer_biases
        self.layer_masks = layer_masks
        self.hidden_state = torch.zeros(self.hidden_size)
        self.init_state = kwargs.get("init_state", 'zero')
        self._set_activation(kwargs.get("activation", "relu"))
        self._set_hidden_state()

        self.input_layer = LinearLayer(
            use_dale=self.use_dale,
            new_synapses=new_synapses[0],
            output_dim=self.hidden_size,
            input_dim=kwargs.get("input_dim", 1),
            use_bias=self.layer_biases[0],
            dist=self.layer_distributions[0],
            mask=self.layer_masks[0],
            allow_negative=allow_negative[0],
        )
        self.hidden_layer = HiddenLayer(
            hidden_size=self.hidden_size,
            new_synapses=new_synapses[1],
            use_dale=self.use_dale,
            dist=self.layer_distributions[1],
            use_bias=self.layer_biases[1],
            scaling=kwargs.get("scaling", 1.0),
            mask=self.layer_masks[1],
            self_connections=kwargs.get("self_connections", False),
            allow_negative=allow_negative[1],
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
            self.activation = lambda x: torch.maximum(torch.tanh(x), torch.tensor(0))
        else:
            raise NotImplementedError

    def _set_hidden_state(self):
        """
        Add the hidden layer to the parameter
        """
        if self.init_state == 'learn':
            self.hidden_state = torch.nn.Parameter(torch.zeros(self.hidden_size), requires_grad=True)
        else:
            self.hidden_state = torch.nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
    # ==================================================================================================

    # FORWARD
    # ==================================================================================================
    def _reset_state(self):
        if self.init_state == 'learn' or self.init_state == 'keep':
            return self.hidden_state
        else:
            return torch.zeros(self.hidden_size)

    def enforce_constraints(self):
        self.input_layer.enforce_constraints()
        self.hidden_layer.enforce_constraints()

    def recurrence(self, fr_t, v_t, u_t):
        """ Recurrence function """
        # through input layer
        w_in_u_t = self.input_layer(u_t)  # u_t @ W_in
        # through hidden layer
        w_hid_fr_t = self.hidden_layer(fr_t)  # fr_t @ W_hid + b
        # update hidden state
        v_t = (1-self.alpha)*v_t + self.alpha*(w_hid_fr_t+w_in_u_t)

        # add pre-activation noise
        if self.preact_noise > 0:
            preact_epsilon = torch.randn((u_t.size(0), self.hidden_size), device=u_t.device) * self.preact_noise
            v_t = v_t + self.alpha*preact_epsilon
    
        # apply activation function
        fr_t = self.activation(v_t)
    
        # add post-activation noise
        if self.postact_noise > 0:
            postact_epsilon = torch.randn((u_t.size(0), self.hidden_size), device=u_t.device) * self.postact_noise
            fr_t = fr_t + postact_epsilon
        
        return fr_t, v_t

    def forward(self, input):
        """
        Propogate input through the network.
        @param input: shape=(seq_len, batch, input_dim), network input
        @return stacked_states: shape=(seq_len, batch, hidden_size), stack of hidden layer status
        """
        v_t = self._reset_state().to(input.device)
        fr_t = self.activation(v_t)
        # update hidden state and append to stacked_states
        stacked_states = []
        for i in range(input.size(0)):
            fr_t, v_t = self.recurrence(fr_t, v_t, input[i])
            # append to stacked_states
            stacked_states.append(fr_t)

        # if keeping the last state, save it to hidden_state
        if self.init_state == 'keep':
            self.hidden_state = fr_t.detach().clone()  # TODO: haven't tested this yet

        return torch.stack(stacked_states, dim=0)
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
            "hidden_min": self.hidden_state.min(),
            "hidden_max": self.hidden_state.max(),
            "hidden_mean": self.hidden_state.mean(),
            "preact_noise": self.preact_noise,
            "postact_noise": self.postact_noise,
            "activation": self.act,
            "alpha": self.alpha,
        }
        self.input_layer.print_layer()
        print_dict("Recurrence", param_dict)
        self.hidden_layer.print_layer()
    # ==================================================================================================
