import torch
from .tensor_pack import TensorPack
from typing import List, Tuple, Union
from .recurrent_layer import RecurrentLayer


class BlockMatrix(torch.nn.Module):
    """
    A block matrix of size n_blocks x n_blocks
    
    **Specification**:
        - Diagonal blocks must be an instance of nn4n.nn.RecurrentLayer
        - Off-diagonal blocks must be an instance of torch.nn.Module
        - Off-diagonal blocks cannot be an instance of nn4n.nn.RecurrentLayer

    **Parameters**:
        - n_blocks: number of blocks
    """
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
            assert isinstance(value, (RecurrentLayer, BlockRecurrentLayer)), "Diagonal blocks must be an instance of nn4n.nn.RecurrentLayer or nn4n.nn.BlockRecurrentLayer"
        else:
            assert isinstance(value, torch.nn.Module), "Off-diagonal blocks must be an instance of torch.nn.Module"
            assert not isinstance(value, RecurrentLayer), "Off-diagonal blocks cannot be an instance of nn4n.nn.RecurrentLayer"
        self.matrix[i][j] = value


class BlockRecurrentLayer(torch.nn.Module):
    """
    A block recurrent layer of size n_blocks x n_blocks, this is a wrapper around BlockMatrix
    For recurrent neural networks with multiple recurrent blocks, in usual use case we don't have feedback
    connections between blocks, so the block matrix is a lower triangular matrix. However, feedback connections
    may play an essential role in the dynamics of the network, so we provide a general BlockMatrix class to
    support full matrix that can be used to model feedback connections.
    
    **Specification**:
        - Diagonal blocks must be an instance of nn4n.nn.RecurrentLayer, these are the 
          recurrent blocks that has hidden states
        - Off-diagonal blocks must be an instance of torch.nn.Module, these are the 
          projection blocks between recurrent blocks. These are effectively linear projections

    **Parameters**:
        n_blocks: number of blocks
    """
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
        """
        Get the size of each block

        **Returns**:
            block_sizes: list of int, the size of each block
        """
        block_sizes = []
        for block_idx in range(self.n_blocks):
            diagonal_block = self.block_recurrent[block_idx, block_idx]
            block_size = diagonal_block.size if diagonal_block is not None else 0
            block_sizes.append(block_size)
        return block_sizes
    
    def set_projection(self, from_idx, to_idx, projection_layer):
        """
        Set the projection block between from_idx and to_idx

        **Examples**:

        Set the projection block between block 0 and block 1::

        >>> block_recurrent.set_projection(0, 1, nn4n.nn.Linear(10, 10))
        """
        # NOTE: this is "inversed" which is slightly confusing
        self[to_idx, from_idx] = projection_layer

    def set_recurrent(self, idx, recurrent_layer):
        """
        Set the recurrent block at index idx

        **Examples**:

        Set the recurrent block at index 0::
        
        >>> block_recurrent.set_recurrent(0, nn4n.nn.RecurrentLayer(10))
        """
        self[idx, idx] = recurrent_layer

    def __getitem__(self, idx: tuple):
        return self.block_recurrent[idx]
    
    def __setitem__(self, idx: tuple, value: torch.nn.Module):
        # Set the value then check the network is initialized every time
        self.block_recurrent[idx] = value
        self.initialized = all(isinstance(self.block_recurrent[i, i], RecurrentLayer) for i in range(self.n_blocks))
        self.block_ranges = self._compute_block_ranges()

    # FORWARD
    # =================================================================================
    def _parse_fr_v(
        self, 
        fr: torch.Tensor, 
        v: torch.Tensor
    ) -> Tuple[TensorPack, TensorPack]:
        """
        In alignment with the simple RNN, which passes fr and v as a single tensor, we also pass fr and v as a single tensor
        to this BlockRecurrentLayer. The passed tensor is expected to have feature dimension equals to sum(block_sizes),
        and this function will parse it into a 1D TensorPack containing the fr and v of each block

        **Parameters**:
            fr: hidden state (post-activation), shape: (batch_size, total_hidden_size)
            v: hidden state (pre-activation), shape: (batch_size, total_hidden_size)

        **Returns**:
            block_fr: hidden state (post-activation), shape: (batch_size, n_blocks, block_size)
            block_v: hidden state (pre-activation), shape: (batch_size, n_blocks, block_size)
        """
        return self._parse_into_blocks(fr), self._parse_into_blocks(v)
    
    def _parse_into_blocks(self, tensor: torch.Tensor) -> TensorPack:
        """
        For a tensor with feature dimension equals to sum(block_sizes), parse it into a list of tensors
        where each tensor has feature dimension equals to the size of the corresponding block

        **Parameters**:
            tensor: tensor to be parsed, shape: (batch_size, total_hidden_size)

        **Returns**:
            block_tensor_pack: a TensorPack of tensors, each of shape (n_blocks,) of torch.Tensor. 
        """
        block_list = []
        for i in range(self.n_blocks):
            block_list.append(tensor[:, self.block_indices(i)])
        return TensorPack(block_list)

    def forward(
        self, 
        fr: torch.Tensor,
        v: torch.Tensor, 
        u: Union[TensorPack, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        """
        Forwardly update network

        **Parameters**:
            fr: hidden state (post-activation), shape: (batch_size, total_hidden_size)
            v: hidden state (pre-activation), shape: (batch_size, total_hidden_size)
            u: list of input tensors, each of shape (batch_size, input_dim)
            **kwargs: additional arguments that will be ignored

        **Returns**:
            fr_n: hidden state (post-activation), shape: (batch_size, total_hidden_size)
            v_n: hidden state (pre-activation), shape: (batch_size, total_hidden_size)
        """
        if not self.initialized:
            raise ValueError("BlockRecurrentLayer is not initialized. All diagonal blocks must be set before forward pass.")

        block_fr = self._parse_into_blocks(fr)
        block_v = self._parse_into_blocks(v)
        block_u_aux = TensorPack([torch.zeros_like(_fr, device=_fr.device) for _fr in block_fr])
        block_fr_n = TensorPack([None for _ in range(self.n_blocks)])
        block_v_n = TensorPack([None for _ in range(self.n_blocks)])

        # Update the auxiliary input for each block
        for from_idx in range(self.n_blocks):
            for to_idx in range(self.n_blocks):
                if to_idx == from_idx:
                    continue
                layer = self.block_recurrent[to_idx, from_idx]
                if layer is not None:
                    block_u_aux[to_idx] += layer(block_fr[from_idx])

        # Pass the updated auxiliary input and external input to each block
        for diag_idx in range(self.n_blocks):
            diag_layer = self.block_recurrent[diag_idx, diag_idx]
            block_fr_n[diag_idx], block_v_n[diag_idx] = diag_layer(
                fr=block_fr[diag_idx], 
                v=block_v[diag_idx], 
                u=u[diag_idx], 
                u_aux=block_u_aux[diag_idx]
            )

        # Return the concatenated updated hidden states of each block
        return torch.cat(list(block_fr_n), dim=-1), torch.cat(list(block_v_n), dim=-1)

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
