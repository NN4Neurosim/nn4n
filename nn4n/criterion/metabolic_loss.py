import torch
from .song_yang import SongYangLoss

class MetabolicLoss(SongYangLoss):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.lambda_met = kwargs.get("lambda_met", 1)

    def forward(self, pred, label, states):
        """
        Compute the loss
        @param pred: size=(-1, 2), predicted labels
        @param label: size=(-1, 2), labels
        @param dur: duration of the trial
        """
        loss = super().forward(pred, label)
        loss_met = torch.square(states).mean()
        
        return loss + self.lambda_met*loss_met
