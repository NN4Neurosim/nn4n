import torch

from nn4n.model import BaseNN
from nn4n.layer import LinearLayer
from nn4n.utils import get_activation


class MLP(BaseNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        self.input_dim = kwargs.pop("input_dim")
        self.output_dim = kwargs.pop("output_dim")
        self.hidden_size = kwargs.pop("hidden_size")
        self.learnable = kwargs.pop("learnable", [True, True])
        self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'uniform'])
        self.layer_biases = kwargs.pop("layer_biases", [True, True])
        self.inhibit = kwargs.pop("inhibit", 0)
        # dynamics parameters
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)
        self.layer_masks = kwargs.pop("layer_masks", [None, None])

        if len(self.layer_distributions) == 3:
            self.layer_distributions.pop(1)
        if len(self.layer_biases) == 3:
            self.layer_biases.pop(1)

        self.act = kwargs.pop("act", "relu")
        self.activation = get_activation(self.act)

        self.input_layer = LinearLayer(
            input_dim=self.input_dim,
            output_dim=self.hidden_size,
            dist=self.layer_distributions[0],
            use_bias=self.layer_biases[0],
            mask=self.layer_masks[0],
            use_dale=False,
            new_synapses=True if self.layer_masks[0] is None else False,
            allow_negative=True,
            learnable=self.learnable[0],
        )

        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            output_dim=self.output_dim,
            dist=self.layer_distributions[1],
            use_bias=self.layer_biases[1],
            mask=self.layer_masks[0],
            use_dale=False,
            new_synapses=True if self.layer_masks[1] is None else False,
            allow_negative=True,
            learnable=self.learnable[1],
        )

    def forward(self, x):
        """
        @param x: size=(batch_size, input_dim)
        """
        x = self.input_layer(x)
        if self.preact_noise > 0:
            x = x + torch.randn_like(x) * self.preact_noise
        # Apply Lateral Inhibition (pre-activation)
        if self.inhibit > 0:
            inhibition = self.inhibit * (x.sum(dim=-1, keepdim=True) - x)
            x = x - inhibition
        s = self.activation(x)
        if self.postact_noise > 0:
            s = s + torch.randn_like(s) * self.postact_noise
        x = self.readout_layer(s)
        return x, s

    def print_layers(self):
        """
        Print the layer information
        """
        self.input_layer.print_layer()
        self.readout_layer.print_layer()