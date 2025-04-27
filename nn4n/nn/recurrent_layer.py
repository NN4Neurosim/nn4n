import torch
from .linear_layer import LinearLayer
from .leaky_linear_layer import LeakyLinearLayer

from typing import Optional

class RecurrentLayer(torch.nn.Module):
    def __init__(
        self,
        leaky_layer: Optional[LeakyLinearLayer] = None,
        projection_layer: Optional[LinearLayer] = None,
        device: str = "cpu"
    ):
        """
        Recurrent layer of the RNN. It primarily serves to group the recurrent layer with its
        projection layer if there's any.

        Parameters:
            leaky_layer: leaky linear layer
            projection_layer: projection layer
        """
        super().__init__()

        self.leaky_layer = leaky_layer
        self.projection_layer = projection_layer
        self.device = torch.device(device)

    @property
    def input_dim(self) -> int:
        return self.leaky_layer.input_dim if self.projection_layer is None else self.projection_layer.input_dim

    @property
    def output_dim(self) -> int:
        return self.leaky_layer.output_dim

    @property
    def hidden_size(self) -> int:
        return self.leaky_layer.input_dim
    
    @property
    def size(self) -> int:
        return (self.hidden_size, self.hidden_size)
    
    @property
    def postact_noise(self) -> float:
        return self.leaky_layer.postact_noise
    
    @property
    def preact_noise(self) -> float:
        return self.leaky_layer.preact_noise

    def freeze(self):
        """
        Freeze the layer
        """
        self.leaky_layer.freeze()
        if self.projection_layer is not None:
            self.projection_layer.freeze()

    def unfreeze(self):
        """
        Unfreeze the layer
        """
        self.leaky_layer.unfreeze()
        if self.projection_layer is not None:
            self.projection_layer.unfreeze()

    def clear_parameters(self):
        """
        Clear the parameters of the layer
        """
        self.leaky_layer.clear_parameters()
        if self.projection_layer is not None:
            self.projection_layer.clear_parameters()
    
    def set_noise(self, postact_noise: float = None, preact_noise: float = None):
        self.leaky_layer.set_noise(postact_noise, preact_noise)

    def forward(self, fr: torch.Tensor, v: torch.Tensor, u: torch.Tensor, u_aux: torch.Tensor = None):
        """
        Forwardly update network

        Parameters:
            fr: hidden state (post-activation), shape: (batch_size, hidden_size)
            v: hidden state (pre-activation), shape: (batch_size, hidden_size)
            u: input, shape: (batch_size, input_size)
            u_aux: auxiliary input to be added after projection, shape: (batch_size, hidden_size)

        Returns:
            fr_next: hidden state (post-activation), shape: (batch_size, hidden_size)
            v_next: hidden state (pre-activation), shape: (batch_size, hidden_size)
        """
        if self.projection_layer is not None:
            u = self.projection_layer(u)
        u = 0 if u is None else u
        u = u + u_aux if u_aux is not None else u
        return self.leaky_layer(fr, v, u)
