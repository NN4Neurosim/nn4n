import torch
import torch.nn as nn
from typing import List
from nn4n.utils import print_dict, get_activation
from nn4n.layer import LinearLayer


class RNN(nn.Module):
    """
    Recurrent layer of the RNN. The layer is initialized by passing specs in layer_struct.

    Parameters:
        - layers: list of layers in the network
    """

    def __init__(self,
                 hidden_layers: List[nn.Module],
                 readout_layer: LinearLayer = None,
                 device: str = "cpu"):
        """
        Initialize the recurrent layer

        Parameters:
            - hidden_layers: list of hidden layers
            - readout_layer: readout layer, optional
        """
        super().__init__()
        if not isinstance(hidden_layers, list) or not all(isinstance(_l, nn.Module) for _l in hidden_layers):
            raise ValueError("`layers` must be a list of nn.Module instances.")
        self.hidden_layers = nn.ModuleList(hidden_layers)
        self.readout_layer = readout_layer
        self.device = torch.device(device)
        self.hidden_sizes = [layer.hidden_size for layer in hidden_layers]

    # FORWARD
    # ==================================================================================================
    def to(self, device: torch.device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        self.device = device
        for layer in self.hidden_layers:
            layer.to(device)
        if self.readout_layer is not None:
            self.readout_layer.to(device)

    def _generate_init_state(
        self,
        dim: int,
        batch_size: int,
        i_val: float = 0
    ) -> torch.Tensor:
        """Generate initial state"""
        return torch.full((batch_size, dim), i_val, device=self.device)

    def forward(
        self,
        x: torch.Tensor,
        init_states: List[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Forwardly update network

        Inputs:
            - x: input, shape: (batch_size, n_timesteps, input_dim)
            - init_states: a list of initial states of the network, each element 
                           has shape: (batch_size, hidden_size_i), i-th hidden layer

        Returns:
            - hidden_state_list: hidden states of the network, list of tensors, each element
        """
        # Initialize hidden states as a list of tensors
        _bs, _T, _ = x.size()
        hidden_states = [
            torch.zeros(_bs, _T+1, hidden_size, device=self.device)
            for hidden_size in self.hidden_sizes
        ]

        # Set the hidden state at t=0 if provided
        if init_states is not None:
            assert len(init_states) == len(self.hidden_layers), \
                "Number of initial states must match the number of hidden layers."
            for i, _s in enumerate(init_states):
                hidden_states[i][:, 0, :] = _s

        # Initialize two lists to store membrane potentials and firing rates
        v_t_list = [
            hidden_states[i][:, 0, :].clone()  # shape: (batch_size, hidden_size_i)
            for i in range(len(self.hidden_layers))
        ]
        fr_t_list = [
            hidden_states[i][:, 0, :].clone()  # shape: (batch_size, hidden_size_i)
            for i in range(len(self.hidden_layers))
        ]


        # Forward pass through time
        for t in range(_T):
            for i, layer in enumerate(self.hidden_layers):
                # Input to the current layer, use input if its the first layer
                u_in_t = x[:, t, :] if i == 0 else fr_t_list[i-1]
                fr_t_list[i], v_t_list[i] = layer(fr_t_list[i], v_t_list[i], u_in_t)

                # Update hidden states and membrane potentials
                hidden_states[i][:, t+1, :] = fr_t_list[i].clone()

        # Trim the hidden states to remove the initial state
        hidden_states = [state[:, 1:, :] for state in hidden_states]

        # Readout layer
        output = self.readout_layer(hidden_states[-1]) if self.readout_layer is not None else None

        return output, hidden_states

    def apply_plasticity(self):
        """Apply plasticity masks to the weight gradients"""
        pass

    def enforce_constraints(self):
        """
        Enforce sparsity and excitatory/inhibitory constraints if applicable.
        This is by default automatically called after each forward pass,
        but can be called manually if needed
        """
        pass

    # ==================================================================================================

    # HELPER FUNCTIONS
    # ==================================================================================================
    def plot_layer(self, **kwargs):
        """Plot the weight matrix and distribution of each layer"""
        pass

    def print_layer(self):
        """Print the weight matrix and distribution of each layer"""
        pass
    # ==================================================================================================
