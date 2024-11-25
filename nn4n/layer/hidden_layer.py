import torch
import torch.nn as nn
from .linear_layer import LinearLayer


class HiddenLayer(nn.Module):
    def __init__(
        self,
        linear_layer: LinearLayer,
        activation: nn.Module,
        input_layer: LinearLayer = None,
        alpha: float = 0.1,
        learn_alpha: bool = False,
        preact_noise: float = 0,
        postact_noise: float = 0,
    ):
        """
        Hidden layer of the network. The layer is initialized by passing specs in layer_struct.

        Parameters:
            - linear_layer: linear layer. I.e., W_rc in the RNN
            - activation: activation function
            - input_layer: input projection layer. I.e., W_in in the RNN, this is
                optional. If not provided, the input will directly be added to the
                hidden state, which is equivalent to setting W_in to identity matrix.
            - alpha: alpha value for the hidden layer
            - learn_alpha: whether to learn alpha
            - preact_noise: noise added to pre-activation
            - postact_noise: noise added to post-activation
        """
        super().__init__()
        assert linear_layer.input_dim == linear_layer.output_dim, \
            "Input and output dimensions of the linear layer must be the same"
        self.linear_layer = linear_layer
        self.input_layer = input_layer
        self.activation = activation
        self.learn_alpha = learn_alpha
        self.preact_noise = preact_noise
        self.postact_noise = postact_noise
        self.alpha = (
            torch.nn.Parameter(
                torch.full((self.linear_layer.input_dim,),
                alpha
            ), requires_grad=True)
            if learn_alpha
            else alpha
        )

    @property
    def input_dim(self) -> int:
        return self.linear_layer.input_dim
    
    @property
    def output_dim(self) -> int:
        return self.linear_layer.output_dim

    @property
    def hidden_size(self) -> int:
        return self.output_dim

    @staticmethod
    def _generate_noise(shape: torch.Size, noise: float) -> torch.Tensor:
        return torch.randn(shape) * noise

    def to(self, device):
        """Move the network to the device (cpu/gpu)"""
        super().to(device)
        self.linear_layer.to(device)
        self.alpha = self.alpha.to(device)
        return self

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
