import torch
import numpy as np


class TensorPack:
    """
    TensorPack is a flexible container for PyTorch Tensors.
    It can store one or more tensors, with no dimension restrictions.

    **Construction rules**:
    
        - If ``data`` is None, we store an empty list.
        - If ``data`` is a single ``torch.Tensor``, we store it in a list.
        - If ``data`` is a single ``np.ndarray``, we convert it to a ``torch.Tensor`` and store it in a list.
        - If ``data`` is a list/tuple, we check each element:
            - If they are all Tensors, NumPy arrays, or None, we store them individually.
        - Otherwise, we attempt to interpret ``data`` as numeric/nested lists and convert it to a single ``torch.Tensor``.

    **Examples**:

    Single PyTorch tensor::

        >>> tc = TensorPack(torch.randn(3, 4))
        >>> tc
        TensorPack(num_tensors=1, tensors=[tensor([...])])

    List of tensors::

        >>> tc_list = TensorPack([torch.randn(2, 2), torch.randn(5)])
        >>> tc_list
        TensorPack(num_tensors=2, tensors=[tensor([[...]]), tensor([...])])

    NumPy arrays::

        >>> import numpy as np
        >>> arr1 = np.random.randn(3, 3)
        >>> arr2 = np.random.randn(4)
        >>> tc_np = TensorPack([arr1, arr2])
        >>> tc_np
        TensorPack(num_tensors=2, tensors=[tensor([[...]]), tensor([...])])

    Nested lists::

        >>> tc_nested = TensorPack([[1, 2, 3], [4, 5, 6]])
        >>> tc_nested
        TensorPack(num_tensors=1, tensors=[tensor([[1, 2, 3],
            [4, 5, 6]])])

    **Notes**:
    
    - The container provides methods similar to a list (e.g., `len`, indexing, iteration).
    - It also provides `.to()`, `.cpu()`, `.cuda()`, and `.numpy()` for convenience.
    """

    def __init__(self, data=None, device=None):
        """
        Initialize the TensorPack with the given data.

        Parameters:
            data (Any): The input data for initializing the container.
            dtype (torch.dtype, optional): The dtype for casting the Tensors.
            device (torch.device, optional): The device for the Tensors.
        """
        self.device = device

        if data is None:
            # No data => empty container
            self.tensors = []

        elif isinstance(data, (list, tuple, TensorPack)):
            self.tensors = self.recursive_convert(data)

        else:
            raise ValueError("data must be a (nested) list of Tensors, np.ndarrays, or None")

    def recursive_convert(self, data):
        if isinstance(data, (list, tuple, TensorPack)):
            return [self.recursive_convert(x) for x in data]
        else:
            return self.convert_to_tensor(data)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        return_item = self.tensors
        if isinstance(index, tuple):
            for idx in index:
                return_item = return_item[idx]
        else:
            return_item = return_item[index]
        if isinstance(return_item, list):
            return TensorPack(self.tensors[index])
        else:
            return return_item

    def __setitem__(self, index, value):
        # If value is a NumPy array, convert it first
        if isinstance(value, np.ndarray):
            value = torch.tensor(value)
        if value is not None and not isinstance(value, torch.Tensor):
            raise ValueError("Value must be a Tensor or None")
        
        if isinstance(index, int):
            # Handle integer indexing
            if isinstance(self.tensors[index], list) and not isinstance(value, list):
                raise ValueError("Cannot set a tensor/NumPy/None to a vector (list) position")
            if isinstance(value, list):
                raise ValueError("Cannot set a list to a single tensor position")
            if isinstance(value, TensorPack):
                raise ValueError("Cannot set a TensorPack to a single tensor position")
            if isinstance(value, np.ndarray) or isinstance(value, torch.Tensor):
                value = self.convert_to_tensor(value)
            self.tensors[index] = value
        else:
            raise ValueError("Index must be an integer")

    def __iter__(self):
        return iter(self.tensors)

    def __repr__(self):
        """
        Print the type of the entries like a torch tensor

        Examples:
            >>> import nn4n
            >>> print(nn4n.tp.TensorPack([torch.randn(2, 2), torch.randn(5)]))
            TensorPack([torch.Size([2, 2]), torch.Size([5])])
        """
        def format_tensor(tensor):
            if tensor is None:
                return "None"
            elif isinstance(tensor, torch.Tensor):
                shape_str = str(list(tensor.shape))
                if len(shape_str) > 20:
                    return f"{shape_str[:20]}..."
                else:
                    print_str = f"{shape_str}"
                    return print_str # + " " * (8 - len(print_str))
            else:
                raise ValueError("Unexpected type in TensorPack")

        def recursive_format(tensors, item_index=0, depth=0):
            if depth < 2 and isinstance(tensors, list):
                if len(tensors) <= 6:
                    return_str = f"[{', '.join([recursive_format(t, i, depth + 1) for i, t in enumerate(tensors)])}]"
                else:
                    if depth == 0:
                        first_half_tensor_types = f"[{', '.join([recursive_format(t, i, depth + 1) for i, t in enumerate(tensors[:3])])}"
                        last_half_tensor_types = f"{', '.join([recursive_format(t, i, depth + 1) for i, t in enumerate(tensors[-3:])])}]"
                        return_str = f"{first_half_tensor_types}, \n" + " " * 12 + "..., \n" + " " * 12 + f"{last_half_tensor_types}"
                    else:
                        tensor_types = [recursive_format(t, i, depth + 1) for i, t in enumerate(tensors[:3])] \
                            + ["  ...   "] + [recursive_format(t, i, depth + 1) for i, t in enumerate(tensors[-3:])]
                        return_str = f"[{', '.join(tensor_types)}]"
                if item_index == 0:
                    return return_str
                else:
                    return "\n" + " " * 12 + f"{return_str}"
            else:
                return format_tensor(tensors)

        return f"TP({recursive_format(self.tensors)})"

    def __str__(self):
        return self.__repr__()

    def to_list(self, keep_none=False):
        """
        Convert the TensorPack to a list of tensors.
        """
        if keep_none:
            return self.tensors
        else:
            t_list = [t for t in self.tensors if t is not None]
            return None if len(t_list) == 0 else t_list

    @property
    def shape(self):
        """
        Recursively get the size unless it's a 1-d array, in which case return (size,).
        """
        def recursive_shape(tensors):
            if isinstance(tensors, list):
                if len(tensors) == 0:
                    return (0,)
                shapes = [recursive_shape(t) for t in tensors]
                if all(s == shapes[0] for s in shapes):
                    return (len(tensors),) + shapes[0]
                else:
                    return (len(tensors),)
            elif isinstance(tensors, torch.Tensor) or tensors is None:
                return ()
            else:
                return ()

        return recursive_shape(self.tensors)
        
    def convert_to_tensor(self, x):
        if x is None:
            return None
        elif isinstance(x, np.ndarray):
            device = self.device if self.device is not None else 'cpu'
            tensor = torch.tensor(x, device=device)
            # Update self.device if it was None
            if self.device is None:
                self.device = tensor.device
            return tensor
        elif isinstance(x, torch.Tensor):
            if self.device is None:
                self.device = x.device
            assert self.device == x.device, \
                f"Expecting tensor to be on the same device as the TensorPack, but got input on {x.device} and TensorPack on {self.device}"
            return x
        elif isinstance(x, (list, tuple)):
            device = self.device if self.device is not None else 'cpu'
            tensor = torch.tensor(x, device=device)
            if self.device is None:
                self.device = tensor.device
            return tensor
        else:
            raise ValueError("Elements must be Tensors, np.ndarrays, or None")

    def to(self, device):
        """
        Move or cast all Tensors to the specified device/dtype.
        Similar to calling .to(...) on a single tensor.
        """
        assert device is not None, "Device must be specified"
        for i, t in enumerate(self.tensors):
            if t is not None:
                self.tensors[i] = t.to(device=device)
        return self

    def cpu(self):
        """Move all Tensors to CPU."""
        return self.to(device='cpu')

    def cuda(self, device=None):
        """
        Move all Tensors to a CUDA device.
        If device is None, defaults to 'cuda'.
        """
        if device is None:
            device = 'cuda'
        return self.to(device=device)

    def numpy(self):
        """
        Return a list of NumPy arrays from the stored Tensors.
        If any tensor requires_grad, it is detached first.
        None entries remain None.
        """
        arrays = []
        for t in self.tensors:
            if t is None:
                arrays.append(None)
            else:
                arrays.append(t.detach().cpu().numpy())
        return arrays

def empty_tp(shape):
    """
    Create an empty TensorPack with the specified shape.
    The TensorPack will all be None.
    
    Parameters:
        shape (tuple): The shape of the TensorPack.

    Examples:
        >>> import nn4n
        >>> nn4n.tp.empty((2, 2))
        TensorPack([[None, None], [None, None]])
    """
    def create_n_dim_list(shape):
        """
        Create an n-dimensional list with the specified shape, with all values being None.

        Parameters:
            shape (tuple): The shape of the n-dimensional list.

        Returns:
            list: An n-dimensional list with the specified shape, filled with None.
        """
        if isinstance(shape, int):
            return [None for _ in range(shape)]
        elif isinstance(shape, tuple):
            return [create_n_dim_list(shape[1:]) for _ in range(shape[0])]
        else:
            raise ValueError("Shape must be an int or tuple")

    return TensorPack(create_n_dim_list(shape))
