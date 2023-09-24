import torch
import torch.nn as nn

class BaseNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        assert type(self) != BaseNN, "BaseNN cannot be run directly"
        self.kwargs_checkpoint = kwargs.copy()
        self._initialize(**kwargs)
    
    def _initialize(self, **kwargs):
        """ initialize the model """
        pass

    def save(self, path):
        """ save model and kwargs to the same file """
        assert type(path) == str, "path must be a string"
        assert path[-4:] == ".pth", "path must end with .pth"
        torch.save({
            "model_state_dict": self.state_dict(),
            "kwargs": self.kwargs_checkpoint
        }, path)

    def load(self, path):
        """ load model and kwargs from the same file """
        assert type(path) == str, "path must be a string"
        assert path[-4:] == ".pth" or path[-3:] == ".pt", "path must end with .pth or .pt"
        checkpoint = torch.load(path)
        self.kwargs_checkpoint = checkpoint["kwargs"]
        self._initialize(**self.kwargs_checkpoint)
        self.load_state_dict(checkpoint["model_state_dict"])

    def print_layers(self):
        """ print layer information """
        pass
