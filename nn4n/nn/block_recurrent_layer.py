import torch
from typing import List, Tuple
from .recurrent_layer import RecurrentLayer

class BlockMatrix(torch.nn.Module):
    def __init__(self, n_blocks: int):
        super().__init__()
        self.n_blocks = n_blocks
        self.matrix = torch.nn.ModuleList(
            [torch.nn.ModuleList([None for _ in range(n_blocks)]) for _ in range(n_blocks)]
        )

    def __getitem__(self, idx: tuple):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError("Index must be a tuple (i, j)")
        i, j = idx
        if not (0 <= i < self.n_blocks) or not (0 <= j < self.n_blocks):
            raise IndexError("Index out of bounds")
        return self.matrix[i][j]

    def __setitem__(self, idx: tuple, value: torch.nn.Module):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError("Index must be a tuple (i, j)")
        i, j = idx
        if not (0 <= i < self.n_blocks):
            raise IndexError(f"Index {i} out of bounds for n_blocks {self.n_blocks}")
        if not (0 <= j < self.n_blocks):
            raise IndexError(f"Index {j} out of bounds for n_blocks {self.n_blocks}")
        if i == j:
            assert isinstance(value, RecurrentLayer), "Diagonal blocks must be an instance of nn4n.nn.RecurrentLayer"
        else:
            assert isinstance(value, torch.nn.Module), "Off-diagonal blocks must be an instance of torch.nn.Module"
            assert not isinstance(value, RecurrentLayer), "Off-diagonal blocks cannot be an instance of nn4n.nn.RecurrentLayer"
        self.matrix[i][j] = value

class BlockRecurrentLayer(torch.nn.Module):
    def __init__(self, n_blocks: int, **kwargs):
        super().__init__()
        self.block_recurrent = BlockMatrix(n_blocks=n_blocks)
        self.initialized = False
    
    @property
    def size(self) -> int:
        return sum(self.block_sizes())

    @property
    def n_blocks(self) -> int:
        return self.block_recurrent.n_blocks
    
    def block_indices(self, block_idx: int) -> torch.Tensor:
        ranges = self.block_ranges[block_idx]
        return torch.arange(ranges[0], ranges[1])

    def _compute_block_ranges(self):
        block_ranges = []
        start_idx = 0
        for block_idx in range(self.n_blocks):
            block_size = self.block_sizes()[block_idx]
            if block_size == 0:
                block_ranges.append(None)
            else:
                block_ranges.append((start_idx, start_idx + block_size))
                start_idx += block_size
        return block_ranges

    def block_sizes(self) -> List[int]:
        block_sizes = []
        for block_idx in range(self.n_blocks):
            diagonal_block = self.block_recurrent[block_idx, block_idx]
            block_size = diagonal_block.size if diagonal_block is not None else 0
            block_sizes.append(block_size)
        return block_sizes
    
    def set_projection(self, from_idx, to_idx, layer):
        # NOTE: this is "inversed" which is slightly confusing
        self[to_idx, from_idx] = layer

    def set_recurrent(self, idx, layer):
        self[idx, idx] = layer
    
    def __getitem__(self, idx: tuple):
        return self.block_recurrent[idx]
    
    def __setitem__(self, idx: tuple, value: torch.nn.Module):
        # Set the value then check the network is initialized every time
        self.block_recurrent[idx] = value
        self.initialized = all(isinstance(self.block_recurrent[i, i], RecurrentLayer) for i in range(self.n_blocks))
        self.block_ranges = self._compute_block_ranges()

    # FORWARD
    # =================================================================================
    def _parse_fr_v(self, fr: torch.Tensor, v: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Parse the fr and v into a list of tensors
        """
        fr_list = []
        v_list = []
        for i in range(self.n_blocks):
            fr_list.append(fr[:, self.block_indices(i)])
            v_list.append(v[:, self.block_indices(i)])
        return fr_list, v_list

    def forward(
        self, 
        fr: torch.Tensor,
        v: torch.Tensor, 
        u_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forwardly update network

        Parameters:
            - fr: hidden state (post-activation), shape: (batch_size, total_hidden_size)
            - v: hidden state (pre-activation), shape: (batch_size, total_hidden_size)
            - u_list: list of input tensors, each of shape (batch_size, input_dim)

        Returns:
            - fr_t_next: hidden state (post-activation), shape: (batch_size, total_hidden_size)
            - v_t_next: hidden state (pre-activation), shape: (batch_size, total_hidden_size)
        """
        if not self.initialized:
            raise ValueError("BlockRecurrentLayer is not initialized. All diagonal blocks must be set before forward pass.")

        fr_list, v_list = self._parse_fr_v(fr, v)
        u_aux_list = [torch.zeros_like(_fr, device=_fr.device) for _fr in fr_list]
        fr_n_list, v_n_list = [None for _ in range(self.n_blocks)], [None for _ in range(self.n_blocks)]

        for from_idx in range(self.n_blocks):
            for to_idx in range(self.n_blocks):
                if to_idx == from_idx:
                    continue
                layer = self.block_recurrent[to_idx, from_idx]
                if layer is not None:
                    u_aux_list[to_idx] += layer(fr_list[from_idx])

        for diag_idx in range(self.n_blocks):
            layer = self.block_recurrent[diag_idx, diag_idx]
            fr_n_list[diag_idx], v_n_list[diag_idx] = layer(fr_list[diag_idx], v_list[diag_idx], u_list[diag_idx], u_aux_list[diag_idx])

        fr_next = torch.cat(fr_n_list, dim=-1)
        v_next = torch.cat(v_n_list, dim=-1)
        return fr_next, v_next

    # HELPER FUNCTIONS
    # ======================================================================================
    def plot_layer(self, **kwargs):
        """
        Plot the layer
        """
        raise NotImplementedError("Plotting is not implemented for BlockRecurrentLayer")

    def _get_specs(self):
        """
        Get specs of the layer
        """
        raise NotImplementedError("Getting specs is not implemented for BlockRecurrentLayer")
