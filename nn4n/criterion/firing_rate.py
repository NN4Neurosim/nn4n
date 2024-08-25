import torch
import torch.nn as nn


class L1FiringRateLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states):
        mean_fr = torch.mean(states, dim=(0, 1))
        return torch.norm(mean_fr, p=1)**2/mean_fr.numel()


class L2FiringRateLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, states):
        mean_fr = torch.mean(states, dim=(0, 1))
        return torch.norm(mean_fr, p=2)**2/mean_fr.numel()
