import torch

from nn4n.model import BaseNN
from nn4n.layer import LinearLayer
from nn4n.utils import get_activation


class MLP(BaseNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _initialize(self, **kwargs):
        super()._initialize(**kwargs)
        self.input_dim = kwargs.pop("input_dim", 1)
        self.output_dim = kwargs.pop("output_dim", 1)
        self.hidden_size = kwargs.pop("hidden_size", 100)
        self.learnable = kwargs.pop("learnable", [True, True])
        self.layer_distributions = kwargs.pop("layer_distributions", ['uniform', 'uniform'])
        self.layer_biases = kwargs.pop("layer_biases", [True, True])
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
            positivity_constraints=False,
            sparsity_constraints=True if self.layer_masks[0] is None else False,
            learnable=self.learnable[0],
        )

        self.readout_layer = LinearLayer(
            input_dim=self.hidden_size,
            output_dim=self.output_dim,
            dist=self.layer_distributions[1],
            use_bias=self.layer_biases[1],
            mask=self.layer_masks[0],
            positivity_constraints=False,
            sparsity_constraints=True if self.layer_masks[1] is None else False,
            learnable=self.learnable[1],
        )

    def forward(self, input):
        """
        @param x: size=(batch_size, input_dim)
        """
        v = self.input_layer(input)
        if self.preact_noise > 0:
            v = v + torch.randn_like(v) * self.preact_noise
        fr = self.activation(v)
        if self.postact_noise > 0:
            fr = fr + torch.randn_like(fr) * self.postact_noise
        v = self.readout_layer(fr)
        return v, fr

    def print_layers(self):
        """
        Print the layer information
        """
        self.input_layer.print_layers()
        self.readout_layer.print_layers()