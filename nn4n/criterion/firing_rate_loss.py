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


class EntropyLoss(nn.Module):
    def __init__(self, reg=1e1, **kwargs):
        super().__init__(**kwargs)
        self.reg = reg

    def forward(self, states):
        # states shape: (batch_size, time_steps, num_neurons)
        batch_size, time_steps, num_neurons = states.shape

        # Normalize the states to create a probability distribution
        # Add a small epsilon to avoid log(0)
        eps = 1e-8
        prob_states = states / (states.sum(dim=-1, keepdim=True) + eps)

        # Compute the entropy of the neuron activations
        entropy_loss = -torch.sum(prob_states * torch.log(prob_states + eps), dim=-1)
        
        # Take the mean entropy over batches and time steps
        mean_entropy = torch.mean(entropy_loss)

        # Add regularization term (optional, same as before)
        reg_loss = torch.mean(torch.norm(states, dim=-1) ** 2)
        total_loss = mean_entropy + self.reg * reg_loss

        return total_loss


class PopulationKL(nn.Module):
    def __init__(self, symmetric=True, reg=1e-3, reduction='mean'):
        super().__init__()
        self.symmetric = symmetric
        self.reg = reg
        self.reduction = reduction

    def forward(self, states_0, states_1):
        # Compute the mean and variance across batches and time steps
        mean_0 = torch.mean(states_0, dim=(0, 1), keepdim=True)  # Shape: (1, 1, n_neurons)
        mean_1 = torch.mean(states_1, dim=(0, 1), keepdim=True)  # Shape: (1, 1, n_neurons)
        var_0 = torch.var(states_0, dim=(0, 1), unbiased=False, keepdim=True)  # Shape: (1, 1, n_neurons)
        var_1 = torch.var(states_1, dim=(0, 1), unbiased=False, keepdim=True)  # Shape: (1, 1, n_neurons)

        # Compute the KL divergence between the two populations (per neuron)
        kl_div = 0.5 * (torch.log(var_1 / var_0) + (var_0 + (mean_0 - mean_1) ** 2) / var_1 - 1)  # Shape: (1, 1, n_neurons)

        # Symmetric KL divergence: average the KL(P || Q) and KL(Q || P)
        if self.symmetric:
            reverse_kl_div = 0.5 * (torch.log(var_0 / var_1) + (var_1 + (mean_1 - mean_0) ** 2) / var_0 - 1)  # Shape: (1, 1, n_neurons)
            kl_div = 0.5 * (kl_div + reverse_kl_div)  # Shape: (1, 1, n_neurons)

        # Apply reduction based on the reduction method
        if self.reduction == 'mean':
            kl_loss = torch.mean(kl_div)  # Scalar value
        elif self.reduction == 'sum':
            kl_loss = torch.sum(kl_div)  # Scalar value
        elif self.reduction == 'none':
            kl_loss = kl_div  # Shape: (1, 1, n_neurons)
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")

        # Regularization: L2 norm of the states across the neurons
        reg_loss = torch.mean(torch.norm(states_0, dim=-1) ** 2) + torch.mean(torch.norm(states_1, dim=-1) ** 2)

        # Combine the KL divergence with the regularization term
        if self.reduction == 'none':
            # If no reduction, add regularization element-wise
            total_loss = kl_loss + self.reg * (torch.norm(states_0, dim=-1) ** 2 + torch.norm(states_1, dim=-1) ** 2)
        else:
            total_loss = kl_loss + self.reg * reg_loss

        return total_loss
