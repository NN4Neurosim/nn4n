import torch
from typing import List


class RNN(torch.nn.Module):
    """
    Recurrent layer of the RNN.

    Parameters:
        - recurrent_layers: list of recurrent layers in the network
    """

    def __init__(self,
                 recurrent_layers: List[torch.nn.Module],
                 readout_layer: torch.nn.Module = None
                 ):
        """
        Initialize the recurrent layer

        Parameters:
            - recurrent_layers: list of recurrent_layers
            - readout_layer: readout layer
        """
        super().__init__()
        if not isinstance(recurrent_layers, list) or \
           not all(isinstance(l, torch.nn.Module) for l in recurrent_layers):
            raise ValueError("`recurrent_layers` must be a list of torch.nn.Module instances.")
        self.recurrent_layers = torch.nn.ModuleList(recurrent_layers)
        self.readout_layer = readout_layer

    # FORWARD
    # ==================================================================================================
    def to(self, device: torch.device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        self.device = device
        for layer in self.recurrent_layers:
            layer.to(device)
        return self

    def _generate_init_state(
        self,
        dim: int,
        batch_size: int,
        i_val: float = 0
    ) -> torch.Tensor:
        """Generate initial state"""
        return torch.full((batch_size, dim), i_val, device=self.device)

    def _get_input_shape(self, x: torch.Tensor) -> int:
        """Get the input size"""
        return x.shape
    
    def _get_input(self, x: torch.Tensor, t: int) -> torch.Tensor:
        """Get the input at time t"""
        return x[:, t]
    
    def _get_output(self, layer_states: List[torch.Tensor]) -> torch.Tensor:
        """Get the output from the layer states"""
        return self.readout_layer(layer_states[-1]) if self.readout_layer is not None else None

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
                           has shape: (batch_size, leaky_layer_i_size), i-th leaky layer

        Returns:
            - hidden_state_list: hidden states of the network, list of tensors, each element
        """
        # Initialize hidden states as a list of tensors
        # Temporarily add an extra time step to store the initial state
        # The initial state will be removed at the end
        bs, T, _ = self._get_input_shape(x)  # For code reuse in BlockRNN
        layer_states = [torch.zeros(bs, T+1, l.size, device=self.device) for l in self.recurrent_layers]

        # Set the hidden state at t=0 if provided
        # TODO: this is a bit problematic because sometime we might want to only set part of the initial states
        if init_states is not None:
            assert len(init_states) == len(self.recurrent_layers), \
                "Number of initial states must match the number of hidden layers."
            for i, init_s in enumerate(init_states):
                layer_states[i][:, 0] = init_s

        # Initialize two lists to store membrane potentials and firing rates for one time step
        # The list is over the sequential hidden layers, not time steps
        # Each item i is of shape (batch_size, hidden_size(i))
        v_list = [layer_states[i][:, 0].clone() for i in range(len(self.recurrent_layers))]
        fr_list = [layer_states[i][:, 0].clone() for i in range(len(self.recurrent_layers))]

        # Forward pass through time
        for t in range(T):
            for i, layer in enumerate(self.recurrent_layers):
                # If the first layer, use the actual input, otherwise use the previous layer's output
                u_in = self._get_input(x, t) if i == 0 else fr_list[i-1]
                fr_list[i], v_list[i] = layer(fr_list[i], v_list[i], u_in)

                # Update hidden states and membrane potentials
                layer_states[i][:, t+1, :] = fr_list[i].clone()

        # Trim the hidden states to remove the initial state
        layer_states = [state[:, 1:, :] for state in layer_states]

        output = self._get_output(layer_states)

        return output, layer_states

    # HELPER FUNCTIONS
    # ==================================================================================================
    def plot_layer(self, **kwargs):
        """Plot the weight matrix and distribution of each layer"""
        for i, layer in enumerate(self.hidden_layers):
            layer.plot_layer(**kwargs)

    def print_layer(self):
        """Print the weight matrix and distribution of each layer"""
        pass
    # ==================================================================================================


class BlockRNN(RNN):
    def __init__(self, recurrent_layers: List[torch.nn.Module], block_readout_layer: torch.nn.Module = None, *args, **kwargs):
        super().__init__(recurrent_layers=recurrent_layers)
        """
        Initialize the block RNN

        Parameters:
            - recurrent_layers: list of recurrent layers
            - block_readout_layer: block readout layer
        """
        if block_readout_layer is not None:
            assert len(block_readout_layer) == self.recurrent_layers[-1].n_blocks, \
                "Number of block readout layers must match the number of blocks in the last recurrent layer."
            self.block_readout_layer = torch.nn.ModuleList(block_readout_layer)
        else:
            self.block_readout_layer = None
        
    def _get_input_shape(self, x: List[torch.Tensor]) -> int:
        """Get the input size"""
        return x[0].shape
    
    def _get_input(self, x: List[torch.Tensor], t: int) -> List[torch.Tensor]:
        """Get the input at time t"""
        return [x[i][:, t] for i in range(len(x))]
    
    def _get_output(self, layer_states: List[torch.Tensor]) -> torch.Tensor:
        """Get the output from the layer states"""
        outputs = []
        for i, layer in enumerate(self.recurrent_layers):
            if self.block_readout_layer is not None:
                outputs.append(self.block_readout_layer[i](layer_states[-1][:, :, layer.block_indices(i)]))
        return outputs
