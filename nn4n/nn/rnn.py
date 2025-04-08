import torch
from typing import List, Optional
from .tensor_pack import TensorPack

class RNN(torch.nn.Module):
    """
    Recurrent layer of the RNN.

    Parameters:
        - recurrent_layers: list of recurrent layers in the network
    """

    def __init__(self,
                 recurrent_layers: Optional[List[torch.nn.Module]] = None,
                 readout_layer: Optional[torch.nn.Module] = None,
                 device: torch.device = "cpu"
                 ):
        """
        Initialize the recurrent layer

        Parameters:
            - recurrent_layers: list of recurrent_layers
            - readout_layer: readout layer
            - device: device to move the network to
        """
        super().__init__()
        self.recurrent_layers = torch.nn.ModuleList(recurrent_layers) if recurrent_layers is not None else None
        self.readout_layer = readout_layer
        self.device = device

    # FORWARD
    # ==================================================================================================
    def to(self, device: torch.device):
        """
        Move the network to the device (cpu/gpu)
        """
        super().to(device)
        self.device = device
        if self.recurrent_layers is not None:
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
    
    def _get_output(self, layer_states: TensorPack) -> TensorPack:
        """Get the output from the layer states"""
        return self.readout_layer(layer_states[-1]) if self.readout_layer is not None else None

    def _assign_init_states(self, layer_states: TensorPack, init_states: TensorPack):
        """Assign initial states to the layer states"""
        assert len(layer_states) == len(init_states), \
            "Number of initial states must match the number of hidden layers."
        for i, init_s in enumerate(init_states):
            if init_s is not None:
                layer_states[i][:, 0] = init_s

    def add_recurrent_layer(self, layer: torch.nn.Module, idx: int = -1):
        """Add a recurrent layer to the network"""
        # If the recurrent layers are not initialized, initialize them
        if self.recurrent_layers is None:
            self.recurrent_layers = []

        # Insert the layer at the specified index
        self.recurrent_layers.insert(idx, layer)
        
        # If the recurrent layers are not a ModuleList, convert them to one
        if not isinstance(self.recurrent_layers, torch.nn.ModuleList):
            self.recurrent_layers = torch.nn.ModuleList(self.recurrent_layers)

    def set_recurrent_layer(self, idx: int, layer: torch.nn.Module):
        """Set a recurrent layer at a specific index"""
        self.recurrent_layers[idx] = layer

    def freeze(self):
        """Freeze the network"""
        for layer in self.recurrent_layers:
            layer.freeze()
        if self.readout_layer is not None:
            self.readout_layer.freeze()

    def unfreeze(self):
        """Unfreeze the network"""
        for layer in self.recurrent_layers:
            layer.unfreeze()
        if self.readout_layer is not None:
            self.readout_layer.unfreeze()

    def forward(
        self,
        x: torch.Tensor,
        init_states: TensorPack = None
    ) -> TensorPack:
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
        layer_states = [torch.zeros(bs, T+1, l.hidden_size, device=self.device) for l in self.recurrent_layers]

        # Set the hidden state at t=0 if provided
        if init_states is not None:
            self._assign_init_states(layer_states, init_states)

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
        layer_states = TensorPack([state[:, 1:, :] for state in layer_states])

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
    def __init__(
        self, 
        recurrent_layers: Optional[List[torch.nn.Module]] = None, 
        block_readout_layer: Optional[List[torch.nn.Module]] = None, 
        *args, 
        **kwargs
    ):
        super().__init__(recurrent_layers=recurrent_layers, *args, **kwargs)
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

    def _get_input_shape(self, x: TensorPack) -> int:
        """Get the input size"""
        for i in range(len(x)):
            if x[i] is not None:
                return x[i].shape
        raise ValueError("At least one input must not be None.")

    def _get_input(self, x: TensorPack, t: int) -> TensorPack:
        """Get the input at time t"""
        return [x[i][:, t] if x[i] is not None else None for i in range(len(x))]

    def _assign_init_states(self, layer_states: TensorPack, init_states: TensorPack):
        """Assign initial states to the layer states"""
        assert len(layer_states) == len(init_states), \
            "Number of initial states must match the number of hidden layers."
        for i in range(len(init_states)):
            for j in range(len(init_states[i])):
                if init_states[i, j] is not None:
                    block_indices = self.recurrent_layers[i].block_indices(j)
                    layer_states[i][:, 0, block_indices] = init_states[i, j]

    def _get_output(self, layer_states: TensorPack) -> TensorPack:
        """Get the output from the layer states"""
        outputs = []
        if self.block_readout_layer is not None:
            n_blocks = self.recurrent_layers[-1].n_blocks
            for i in range(n_blocks):
                layer = self.recurrent_layers[-1]
                readout_layer = self.block_readout_layer[i]
                if readout_layer is not None:
                    outputs.append(readout_layer(layer_states[-1][:, :, layer.block_indices(i)]))
                else:
                    outputs.append(None)
        return TensorPack(outputs)
