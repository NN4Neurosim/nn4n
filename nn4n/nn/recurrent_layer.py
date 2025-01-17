import torch
from nn4n.nn import LinearLayer, LeakyLinearLayer


class RecurrentLayer(torch.nn.Module):
    def __init__(
        self,
        leaky_layer: LeakyLinearLayer,
        projection_layer: LinearLayer = None,
        device: str = "cpu"
    ):
        """
        Recurrent layer of the RNN. It primarily serves to group the recurrent layer with its
        projection layer if there's any.

        Parameters:
            - leaky_layer: leaky linear layer
            - projection_layer: projection layer
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
    def size(self) -> int:
        return self.leaky_layer.input_dim

    def to(self, device: torch.device):
        """Move the network to the device (cpu/gpu)"""
        super().to(device)
        self.device = device
        self.leaky_layer.to(device)
        if self.projection_layer is not None:
            self.projection_layer.to(device)
        return self

    def forward(self, fr: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forwardly update network

        Parameters:
            - fr: hidden state (post-activation), shape: (batch_size, hidden_size)
            - v: hidden state (pre-activation), shape: (batch_size, hidden_size)
            - u: input, shape: (batch_size, input_size)

        Returns:
            - fr_next: hidden state (post-activation), shape: (batch_size, hidden_size)
            - v_next: hidden state (pre-activation), shape: (batch_size, hidden_size)
        """
        u = self.projection_layer(u) if self.projection_layer is not None else u
        return self.leaky_layer(fr, v, u)
