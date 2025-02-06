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

    def __init__(self, data=None, dtype=None, device=None):
        """
        Initialize the TensorPack with the given data.

        Parameters:
            data (Any): The input data for initializing the container.
            dtype (torch.dtype, optional): The dtype for casting the Tensors.
            device (torch.device, optional): The device for the Tensors.
        """
        if data is None:
            # No data => empty container
            self.tensors = []

        if isinstance(data, (list, tuple)):
            # Recursively convert nested lists to Tensors
            def recursive_convert(data):
                if isinstance(data, (list, tuple)):
                    return [recursive_convert(x) for x in data]
                else:
                    return TensorPack.convert_to_tensor(data, dtype=dtype, device=device)

            self.tensors = recursive_convert(data)
        else:
            raise ValueError("data must be a (nested) list of Tensors, np.ndarrays, or None")

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
        
        if isinstance(index, tuple):
            # Handle tuple indexing
            target = self.tensors
            for idx in index[:-1]:
                target = target[idx]
            if isinstance(target[index[-1]], list) and not isinstance(value, list):
                raise ValueError("Cannot set a tensor/NumPy/None to a vector (list) position")
            target[index[-1]] = value
        else:
            # Handle integer indexing
            if isinstance(self.tensors[index], list) and not isinstance(value, list):
                raise ValueError("Cannot set a tensor/NumPy/None to a vector (list) position")
            self.tensors[index] = value

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
                if len(shape_str) > 8:
                    return f"({shape_str[:8]}...)"
                else:
                    print_str = f"({shape_str})"
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

        return f"TensorPack({recursive_format(self.tensors)})"

    def __str__(self):
        return self.__repr__()

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
    
    @staticmethod
    def convert_to_tensor(x, dtype=None, device=None):
        if isinstance(x, np.ndarray):
            return torch.tensor(x, dtype=dtype, device=device)
        elif isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype) if (device or dtype) else x
        elif isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=dtype, device=device)
        elif x is None:
            return None
        else:
            raise ValueError("Elements must be Tensors, np.ndarrays, or None")

    def to(self, device=None, dtype=None):
        """
        Move or cast all Tensors to the specified device/dtype.
        Similar to calling .to(...) on a single tensor.
        """
        for i, t in enumerate(self.tensors):
            if t is not None:
                self.tensors[i] = t.to(device=device, dtype=dtype)
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

def empty(shape):
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
        if len(shape) == 0:
            return None
        return [create_n_dim_list(shape[1:]) for _ in range(shape[0])]

    return TensorPack(create_n_dim_list(shape))
