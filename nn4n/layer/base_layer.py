import torch
import torch.nn as nn
import numpy as np
import nn4n.utils as utils


class BaseLayer(nn.Module):
    """
    nn4n Layer class
    """

    def __init__(self):
        super().__init__()

    def get_specs(self):
        pass

    def print_layer(self):
        """
        Print the specs of the layer
        """
        utils.print_dict(f"{self.__class__.__name__} layer", self.get_specs())
