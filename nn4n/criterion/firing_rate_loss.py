import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomLoss(nn.Module):
    def __init__(self, batch_first=True):
        super().__init__()
        self.batch_first = batch_first
    
    def forward(self, **kwargs):
        pass


class FiringRateLoss(CustomLoss):
    def __init__(self, metric='l2', **kwargs):
        super().__init__(**kwargs)
        assert metric in ['l1', 'l2'], "metric must be either l1 or l2"
        self.metric = metric

    def forward(self, states, **kwargs):
        # Calculate the mean firing rate across specified dimensions
        mean_fr = torch.mean(states, dim=(0, 1))
        
        # Replace custom norm calculation with PyTorch's built-in norm
        if self.metric == 'l1':
            return F.l1_loss(mean_fr, torch.zeros_like(mean_fr), reduction='mean')
        else:
            return F.mse_loss(mean_fr, torch.zeros_like(mean_fr), reduction='mean')


class FiringRateDistLoss(CustomLoss):
    def __init__(self, metric='sd', **kwargs):
        super().__init__(**kwargs)
        valid_metrics = ['sd', 'cv', 'mean_ad', 'max_ad']
        assert metric in valid_metrics, (
            "metric must be chosen from 'sd' (standard deviation), "
            "'cv' (coefficient of variation), 'mean_ad' (mean abs deviation), "
            "or 'max_ad' (max abs deviation)."
        )
        self.metric = metric

    def forward(self, states, **kwargs):
        mean_fr = torch.mean(states, dim=(0, 1))

        # Standard deviation
        if self.metric == 'sd':
            return torch.std(mean_fr)

        # Coefficient of variation
        elif self.metric == 'cv':
            return torch.std(mean_fr) / torch.mean(mean_fr)

        # Mean absolute deviation
        elif self.metric == 'mean_ad':
            avg_mean_fr = torch.mean(mean_fr)
            # Use F.l1_loss for mean absolute deviation
            return F.l1_loss(mean_fr, avg_mean_fr.expand_as(mean_fr), reduction='mean')

        # Maximum absolute deviation
        elif self.metric == 'max_ad':
            avg_mean_fr = torch.mean(mean_fr)
            return torch.max(torch.abs(mean_fr - avg_mean_fr))


class StatePredictionLoss(CustomLoss):
    def __init__(self, tau=1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def forward(self, states, **kwargs):
        if not self.batch_first:
            states = states.transpose(0, 1)

        # Ensure the sequence is long enough for the prediction window
        assert states.shape[1] > self.tau, "The sequence length is shorter than the prediction window."
        
        # Use MSE loss instead of manual difference calculation
        return F.mse_loss(states[:-self.tau], states[self.tau:], reduction='mean')
