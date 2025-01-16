import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, apply_softmax=True):
        """
        Cross-Entropy Loss for continuous probability distributions.

        Args:
            apply_softmax (bool): Whether to apply softmax to predictions.
        """
        super().__init__()
        self.apply_softmax = apply_softmax

    def forward(self, pred, target):
        """
        Compute the cross-entropy loss.

        Args:
            pred (torch.Tensor): Predicted logits or probabilities of shape [batch_size, num_classes].
            target (torch.Tensor): Ground truth probabilities of shape [batch_size, num_classes].

        Returns:
            torch.Tensor: Scalar cross-entropy loss.
        """
        if self.apply_softmax:
            pred = F.softmax(pred, dim=-1)
        
        # Clamp predictions to avoid log(0)
        pred = torch.clamp(pred, min=1e-12)
        
        # Compute cross-entropy loss
        loss = -(target * torch.log(pred)).sum(dim=-1).mean()
        return loss
