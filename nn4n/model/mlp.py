import torch

from nn4n.model import BaseNN
from nn4n.layer import LinearLayer
from nn4n.utils import get_activation


class MLP(BaseNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'uniform'])
        self.layer_biases = kwargs.pop("layer_biases", [True, True])
        # dynamics parameters
        self.preact_noise = kwargs.pop("preact_noise", 0)
        self.postact_noise = kwargs.pop("postact_noise", 0)

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
            mask=None,
            use_dale=False,
            new_synapses=True,
            allow_negative=True,
        )

        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            output_dim=self.output_dim,
            dist=self.layer_distributions[1],
            use_bias=self.layer_biases[1],
            mask=None,
            use_dale=False,
            new_synapses=True,
            allow_negative=True,
        )

    def forward(self, x):
        """
        @param x: size=(batch_size, input_dim)
        """
        x = self.input_layer(x)
        if self.preact_noise > 0:
            x = x + torch.randn_like(x) * self.preact_noise
        s = self.activation(x)
        if self.postact_noise > 0:
            s = s + torch.randn_like(s) * self.postact_noise
        x = self.readout_layer(s)
        return x, s
