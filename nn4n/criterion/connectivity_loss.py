import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNConnectivityLoss(nn.Module):
    def __init__(self, layer, metric='fro', **kwargs):
        super().__init__()
        assert metric in ['l1', 'fro'], "metric must be either l1 or l2"
        self.metric = metric
        self.layer = layer

    def forward(self, model, **kwargs):
        if self.layer == 'all':
            weights = [
                model.recurrent_layer.input_layer.weight,
                model.recurrent_layer.hidden_layer.weight,
                model.readout_layer.weight
            ]

            loss = torch.sum(torch.stack([self._compute_norm(weight) for weight in weights]))
            return loss
        elif self.layer == 'input':
            return self._compute_norm(model.recurrent_layer.input_layer.weight)
        elif self.layer == 'hidden':
            return self._compute_norm(model.recurrent_layer.hidden_layer.weight)
        elif self.layer == 'readout':
            return self._compute_norm(model.readout_layer.weight)
        else:
            raise ValueError(f"Invalid layer '{self.layer}'. Available layers are: 'all', 'input', 'hidden', 'readout'")

    def _compute_norm(self, weight):
        if self.metric == 'l1':
            return torch.norm(weight, p=1)
        else:
            return torch.norm(weight, p='fro')
