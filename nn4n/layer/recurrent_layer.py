import torch
import torch.nn as nn
from nn4n.utils import print_dict, get_activation

from nn4n.layer import HiddenLayer
from nn4n.layer import LinearLayer


class RecurrentLayer(nn.Module):
    """
    Recurrent layer of the RNN. The layer is initialized by passing specs in layer_struct.
    
    Required keywords in layer_struct:
        - activation: activation function, default: "relu"
        - preact_noise: noise added to pre-activation
        - postact_noise: noise added to post-activation
        - dt: time step, default: 10
        - tau: time constant, default: 100
        - init_state: initial state of the network. It defines the hidden state at t=0.
            - 'zero': all zeros
            - 'keep': keep the last state
            - 'learn': learn the initial state
        - in_struct: input layer layer_struct
        - hid_struct: hidden layer layer_struct
    """
    def __init__(self, layer_struct, **kwargs):
        super().__init__()
        self.alpha = layer_struct['dt']/layer_struct['tau']
        self.hidden_size = layer_struct['hid_struct']['input_dim']
        self.hidden_state = torch.zeros(self.hidden_size)
        self.init_state = layer_struct['init_state']
        self.act = layer_struct['activation']
        self.activation = get_activation(self.act)
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)
        self._set_hidden_state()

        self.input_layer = LinearLayer(layer_struct=layer_struct['in_struct'])
        self.hidden_layer = HiddenLayer(layer_struct=layer_struct['hid_struct'])

    # INITIALIZATION
    # ==================================================================================================
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
    def to(self, device):
        """ Move the network to the device (cpu/gpu) """
        super().to(device)
        self.input_layer.to(device)
        self.hidden_layer.to(device)
        self.hidden_state = self.hidden_state.to(device)

    def forward(self, x):
        """ 
        Forwardly update network

        Inputs:
            - x: input, shape: (n_timesteps, batch_size, input_dim)

        Returns:
            - states: shape: (n_timesteps, batch_size, hidden_size)
        """
        v_t = self._reset_state().to(x.device)
        fr_t = self.activation(v_t)
        # update hidden state and append to stacked_states
        stacked_states = []
        for i in range(x.size(0)):
            fr_t, v_t = self._recurrence(fr_t, v_t, x[i])
            # append to stacked_states
            stacked_states.append(fr_t)

        # if keeping the last state, save it to hidden_state
        if self.init_state == 'keep':
            self.hidden_state = fr_t.detach().clone()  # TODO: haven't tested this yet

        return torch.stack(stacked_states, dim=0)

    def _reset_state(self):
        if self.init_state == 'learn' or self.init_state == 'keep':
            return self.hidden_state
        else:
            return torch.zeros(self.hidden_size)

    def apply_plasticity(self):
        """ Apply plasticity masks to the weight gradients """
        self.input_layer.apply_plasticity()
        self.hidden_layer.apply_plasticity()

    def enforce_constraints(self):
        """ 
        Enforce sparsity and excitatory/inhibitory constraints if applicable.
        This is by default automatically called after each forward pass,
        but can be called manually if needed
        """
        self.input_layer.enforce_constraints()
        self.hidden_layer.enforce_constraints()

    def _recurrence(self, fr_t, v_t, u_t):
        """ Recurrence function """
        # through input layer
        v_in_u_t = self.input_layer(u_t)  # u_t @ W_in
        # through hidden layer
        v_hid_fr_t = self.hidden_layer(fr_t)  # fr_t @ W_hid + b
        # update hidden state
        v_t = (1-self.alpha)*v_t + self.alpha*(v_hid_fr_t+v_in_u_t)

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
    # ==================================================================================================

    # HELPER FUNCTIONS
    # ==================================================================================================
    def plot_layers(self, **kwargs):
        """ Plot the weights matrix and distribution of each layer """
        self.input_layer.plot_layers()
        self.hidden_layer.plot_layers()

    def print_layers(self):
        """ Print the weights matrix and distribution of each layer """
        param_dict = {
            "init_hidden_min": self.hidden_state.min(),
            "init_hidden_max": self.hidden_state.max(),
            "preact_noise": self.preact_noise,
            "postact_noise": self.postact_noise,
            "activation": self.act,
            "alpha": self.alpha,
            "init_state": self.init_state,
            "init_state_learnable": self.hidden_state.requires_grad,
        }
        self.input_layer.print_layers()
        print_dict("Recurrence", param_dict)
        self.hidden_layer.print_layers()
    # ==================================================================================================
