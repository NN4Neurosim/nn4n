import torch
from .tensor_pack import TensorPack
from typing import List, Tuple, Union, Optional
from .recurrent_layer import RecurrentLayer
from .linear_layer import LinearLayer
from .leaky_linear_layer import LeakyLinearLayer


def check_initialized(func):
    """
    Decorator to check if the network is initialized
    """
    def wrapper(self, *args, **kwargs):
        if not self.initialized:
            raise ValueError("The network is not initialized. Cannot call this function.")
        return func(self, *args, **kwargs)
    return wrapper


class BlockRecurrentLayer(torch.nn.Module):
    """
    A block recurrent layer of size n_blocks x n_blocks.
    For recurrent neural networks with multiple recurrent blocks, in usual use case we don't have feedback
    connections between blocks, so the block matrix is a lower triangular matrix. However, feedback connections
    may play an essential role in the dynamics of the network, so we provide a general structure that
    can be used to model feedback connections.
    
    **Specification**:
        - Diagonal blocks must be an instance of nn4n.nn.RecurrentLayer or nn4n.nn.BlockRecurrentLayer, these are the 
          recurrent blocks that has hidden states
        - Off-diagonal blocks must be an instance of torch.nn.Module, these are the 
          projection blocks between recurrent blocks. These are effectively linear projections

    **Parameters**:
        n_blocks: number of blocks
    """
    def __init__(self, n_blocks: Optional[int] = None, **kwargs):
        super().__init__()
        self.n_blocks = n_blocks if n_blocks is not None else 0
        if n_blocks is not None:
            self.block_matrix = torch.nn.ModuleList(
                [torch.nn.ModuleList([None for _ in range(n_blocks)]) for _ in range(n_blocks)]
            )
        else:
            self.block_matrix = torch.nn.ModuleList()
        self.initialized = False
        self.network_assembled = False

    def __getitem__(self, idx: tuple):
        if not isinstance(idx, tuple) or len(idx) != 2:
            raise IndexError("Index must be a tuple (i, j)")
        i, j = idx
        if not (0 <= i < self.n_blocks) or not (0 <= j < self.n_blocks):
            raise IndexError("Index out of bounds")
        return self.block_matrix[i][j]

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
        self.block_matrix[i][j] = value
        
        # Check initialization status after each set operation
        self.block_ranges = self._compute_block_ranges()
        self.initialized = all(isinstance(self.block_matrix[i][i], RecurrentLayer) for i in range(self.n_blocks))

    def add_block(self, n_add: int):
        """
        Add a new block to the block matrix
        """
        # First, iterate through each row of the current block matrix and add n_add None to each row
        for i in range(self.n_blocks):
            self.block_matrix[i].extend([None for _ in range(n_add)])
        
        # Then, add n_add new rows to the block matrix
        for _ in range(n_add):
            self.block_matrix.append(torch.nn.ModuleList([None for _ in range(self.n_blocks + n_add)]))

        # Update n_blocks
        self.n_blocks += n_add

    def list_all_blocks(self):
        """
        Get the values of the block matrix
        """
        return [self.block_matrix[i][j] for i in range(self.n_blocks) for j in range(self.n_blocks)]
    
    def freeze(self):
        """
        Freeze the layer
        """
        if not self.network_assembled:
            self.assembled_network.freeze()
        else:
            for block in self.list_all_blocks():
                if block is not None:
                    block.freeze()

    def unfreeze(self):
        """
        Unfreeze the layer
        """
        if not self.network_assembled:
            self.assembled_network.unfreeze()
        else:
            for block in self.list_all_blocks():
                if block is not None:
                    block.unfreeze()

    @property
    def hidden_size(self) -> int:
        return sum(self.block_sizes())

    def size(self) -> Tuple[int, int]:
        hidden_size = self.hidden_size
        return hidden_size, hidden_size
    
    def block_indices(self, block_idx: int) -> torch.Tensor:
        ranges = self.block_ranges[block_idx]
        return torch.arange(ranges[0], ranges[1])
    
    def block_slices(self, block_idx: int) -> Tuple[slice, slice]:
        ranges = self.block_ranges[block_idx]
        return slice(ranges[0], ranges[1])

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
            diagonal_block = self.block_matrix[block_idx][block_idx]
            block_size = diagonal_block.hidden_size if diagonal_block is not None else 0
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

    def add_recurrent(self, recurrent_layer):
        """
        Add a new recurrent block to the network
        """
        self.add_block(1)
        self[self.n_blocks - 1, self.n_blocks - 1] = recurrent_layer

    @check_initialized
    def assemble(self):
        """
        Initialize the full network
        """
        if not self.initialized:
            raise ValueError("The initialization of the network is not complete")

        # Get reference block for comparison
        ref_block = self.block_matrix[0][0]
        
        # Properties that must be consistent across blocks
        properties = {
            'activation': lambda b: type(b.leaky_layer.activation),
            'learn_alpha': lambda b: b.leaky_layer.learn_alpha,
            'preact_noise': lambda b: b.leaky_layer.preact_noise,
            'postact_noise': lambda b: b.leaky_layer.postact_noise
        }

        # Check each block against reference block
        for i in range(self.n_blocks):
            block = self.block_matrix[i][i]
            for prop_name, prop_getter in properties.items():
                cur_prop = prop_getter(block)
                ref_prop = prop_getter(ref_block)
                if cur_prop != ref_prop:
                    raise ValueError(
                        f"All diagonal blocks must use the same {prop_name}. "
                        f"Block {i} uses {cur_prop}, "
                        f"which is different from {ref_prop}"
                    )
        
        # Gather all network specs
        # Helper functions to extract dimensions
        def get_input_dim(block):
            return block.projection_layer.input_dim if block.projection_layer is not None else 0
        
        def get_hidden_dim(block):
            return block.leaky_layer.size
        
        # Extract dimensions from diagonal blocks
        diagonal_blocks = [self.block_matrix[i][i] for i in range(self.n_blocks)]
        input_dims = [get_input_dim(block) for block in diagonal_blocks]
        hidden_dims = [get_hidden_dim(block) for block in diagonal_blocks]
        
        # Create input slices for mapping
        input_slices = []
        current_pos = 0
        for dim in input_dims:
            if dim == 0:
                input_slices.append(None)
            else:
                input_slices.append(slice(current_pos, current_pos + dim))
                current_pos += dim
        
        # Calculate total dimensions
        total_input_dim = sum(input_dims)
        total_hidden_dim = sum(hidden_dims)
        
        # Initialize matrices
        input_mat = torch.zeros(total_hidden_dim, total_input_dim)
        hidden_mat = torch.zeros(total_hidden_dim, total_hidden_dim)
        input_sparsity_mask = torch.ones(total_hidden_dim, total_input_dim)
        hidden_sparsity_mask = torch.ones(total_hidden_dim, total_hidden_dim)
        
        # Initialize matrices with block values
        self._initialize_hidden_mat(hidden_mat, hidden_sparsity_mask)
        self._initialize_input_mat(input_mat, input_sparsity_mask, input_slices)

        # Generate the assembled network
        rec_lin = LinearLayer(
            input_dim=total_hidden_dim,
            output_dim=total_hidden_dim,
            sparsity_mask=hidden_sparsity_mask.T,
        )
        rec_lin.weight.data = hidden_mat
        ref_layer = self.block_matrix[0][0].leaky_layer
        proj_lin = LinearLayer(
            input_dim=total_input_dim,
            output_dim=total_hidden_dim,
            sparsity_mask=input_sparsity_mask.T,
        )
        proj_lin.weight.data = input_mat
        self.assembled_network = RecurrentLayer(
            leaky_layer=LeakyLinearLayer(
                linear_layer=rec_lin,
                activation=ref_layer.activation,
                alpha=ref_layer.alpha[0].item(),
                learn_alpha=ref_layer.learn_alpha,
                preact_noise=ref_layer.preact_noise,
                postact_noise=ref_layer.postact_noise,
            ),
            projection_layer=proj_lin,
        )
        self._clear_block_matrix()
        self.network_assembled = True

        return self
    
    def _clear_block_matrix(self):
        """
        Since that we are now using the assembled network, we will remove all
        the learnable parameters in the block matrix but keep the structure
        """
        # Copy the structure but set all blocks to None
        regular_matrix = []
        for i in range(self.n_blocks):
            regular_matrix.append([])
            for j in range(self.n_blocks):
                block = self.block_matrix[i][j]
                if block is not None:
                    block.clear_parameters()
                regular_matrix[i].append(block)

        # Replace the ModuleList with a regular list
        del self.block_matrix
        self.block_matrix = regular_matrix

    def _initialize_input_mat(
            self, 
            input_mat: torch.Tensor, 
            input_sparsity_mask: torch.Tensor, 
            input_slices: List[slice]
        ):
        """
        Initialize the input matrix
        """
        for i in range(self.n_blocks):
            b_slice = self.block_slices(i)
            block = self.block_matrix[i][i]
            if block is not None and block.projection_layer is not None:    
                if input_slices[i] is not None:
                    input_mat[b_slice, input_slices[i]] = block.projection_layer.weight.clone()
                else:
                    input_sparsity_mask[b_slice, input_slices[i]] = 0

    def _initialize_hidden_mat(
            self, hidden_mat: torch.Tensor, hidden_sparsity_mask: torch.Tensor
        ):
        """
        Initialize the weights of the recurrent blocks
        """
        for i in range(self.n_blocks):
            i_slice = self.block_slices(i)
            for j in range(self.n_blocks):
                j_slice = self.block_slices(j)
                block = self.block_matrix[i][j]
                if i == j:
                    hidden_mat[i_slice, j_slice] = block.leaky_layer.linear_layer.weight.clone()
                else:
                    if block is not None:
                        hidden_mat[i_slice, j_slice] = block.weight.clone()
                    else:
                        hidden_sparsity_mask[i_slice, j_slice] = 0

    def get_recurrent(self, idx: int):
        """
        Get the recurrent block at index idx

        **Parameters**:
            idx: index of the block

        **Returns**:
            block: the recurrent block at index idx
        """
        return self.block_matrix[idx][idx]

    def get_projection(self, from_idx: int, to_idx: int):
        """
        Get the projection block between from_idx and to_idx

        **Parameters**:
            from_idx: index of the from block
            to_idx: index of the to block

        **Returns**:
            projection: the projection block between from_idx and to_idx
        """
        return self.block_matrix[to_idx][from_idx]

    def get_block(self, idx: Tuple[int, int]):
        """
        Get the recurrent block at index idx

        **Parameters**:
            idx: index of the block

        **Returns**:
            block: the recurrent block at index idx
        """
        return self[idx]

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
        if self.network_assembled:
            return self._simple_forward(fr, v, u, **kwargs)
        else:
            return self._block_forward(fr, v, u, **kwargs)

    def _simple_forward(
        self, 
        fr: torch.Tensor,
        v: torch.Tensor, 
        u: Union[TensorPack, torch.Tensor],
        **kwargs
    ) -> torch.Tensor:
        return self.assembled_network(fr, v, u, **kwargs)

    def _block_forward(
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
                layer = self.block_matrix[to_idx][from_idx]
                if layer is not None:
                    block_u_aux[to_idx] += layer(block_fr[from_idx])

        # Pass the updated auxiliary input and external input to each block
        for diag_idx in range(self.n_blocks):
            diag_layer = self.block_matrix[diag_idx][diag_idx]
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
