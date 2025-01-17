import torch
from typing import List
from .linear_layer import LinearLayer


class BlockMatrix:
    def __init__(self, n_blocks: int):
        """
        Initializes a BlockMatrix.
        
        Args:
            n (int): Size of the matrix (n x n).
        """
        self.n_blocks = n_blocks
        self.matrix = [[None for _ in range(n_blocks)] for _ in range(n_blocks)]

    def __getitem__(self, idx):
        """
        Get item using matrix-style indexing.

        Parameters:
            - idx (tuple): A tuple (i, j) representing the row and column indices.

        Returns:
            The module or value at position (i, j).
        """
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError("Index must be a tuple (i, j)")
        
        i, j = idx
        if not (0 <= i < self.n_blocks) or not (0 <= j < self.n_blocks):
            raise IndexError("Index out of bounds")
        
        return self.matrix[i][j]

    def __setitem__(self, idx, value):
        """
        Set item using matrix-style indexing.

        Args:
            idx (tuple): A tuple (i, j) representing the row and column indices.
            value: The value to set at position (i, j).
        """
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError("Index must be a tuple (i, j)")
        
        i, j = idx
        if not (0 <= i < self.n_blocks) or not (0 <= j < self.n_blocks):
            raise IndexError("Index out of bounds")
        
        self.matrix[i][j] = value


class BlockRecurrentLayer(torch.nn.Module):
    def __init__(
        self,
        n_blocks: int,
    ):
        """
        Hidden layer of the network. The layer is initialized by passing specs in layer_struct.

        Parameters:
            - n_blocks: number of blocks in the layer
        """
        super().__init__()
        self.blocks = BlockMatrix(n_blocks)

    @property
    def n_blocks(self) -> int:
        return self.blocks.n_blocks

    @property
    def size(self) -> int:
        return sum(self.list_sizes())

    def list_sizes(self) -> List[int]:
        """
        Get the sizes of the blocks
        """
        return [self.blocks[i, i].input_dim for i in range(self.n_blocks)]

    def to(self, device):
        """Move the network to the device (cpu/gpu)"""
        super().to(device)
        for i in range(self.n_blocks):
            for j in range(self.n_blocks):
                block = self.blocks[i, j]
                if block is not None and isinstance(block, torch.nn.Module):
                    block.to(device)
        return self

    # FORWARD
    # =================================================================================
    def forward(
        self, 
        fr_hid_t: torch.Tensor,
        v_hid_t: torch.Tensor, 
        u_in_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Forwardly update network

        Parameters:
            - fr_hid_t: hidden state (post-activation), shape: (batch_size, hidden_size)
            - v_hid_t: hidden state (pre-activation), shape: (batch_size, hidden_size)
            - u_in_t: input, shape: (batch_size, input_size)

        Returns:
            - fr_t_next: hidden state (post-activation), shape: (batch_size, hidden_size)
            - v_t_next: hidden state (pre-activation), shape: (batch_size, hidden_size)
        """
        v_in_t = self.input_layer(u_in_t) if self.input_layer is not None else u_in_t
        v_hid_t_next = self.linear_layer(fr_hid_t)
        v_t_next = (1 - self.alpha) * v_hid_t + self.alpha * (v_hid_t_next + v_in_t)
        if self.preact_noise > 0:
            _preact_noise = self._generate_noise(v_t_next.size(), self.preact_noise)
            v_t_next = v_t_next + _preact_noise
        fr_t_next = self.activation(v_t_next)
        if self.postact_noise > 0:
            _postact_noise = self._generate_noise(fr_t_next.size(), self.postact_noise)
            fr_t_next = fr_t_next + _postact_noise
        return fr_t_next, v_t_next

    def enforce_constraints(self):
        """
        Enforce constraints on the layer
        """
        self.linear_layer.enforce_constraints()
        self.input_layer.enforce_constraints()
    
    def apply_plasticity(self):
        """
        Apply plasticity masks to the weight gradients
        """
        self.linear_layer.apply_plasticity()
        self.input_layer.apply_plasticity()

    def train(self):
        # TODO: change the noise to regular level
        pass

    def eval(self):
        # TODO: change the noise to zero
        pass

    # HELPER FUNCTIONS
    # ======================================================================================
    def plot_layer(self, **kwargs):
        """
        Plot the layer
        """
        self.linear_layer.plot_layer(**kwargs)
        if self.input_layer is not None:
            self.input_layer.plot_layer(**kwargs)

    def _get_specs(self):
        """
        Get specs of the layer
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_size": self.hidden_size,
            "alpha": self.alpha,
            "learn_alpha": self.learn_alpha,
            "preact_noise": self.preact_noise,
            "postact_noise": self.postact_noise,
        }
