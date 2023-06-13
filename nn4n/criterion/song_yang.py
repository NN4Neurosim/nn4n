import torch
import torch.nn as nn

class SongYangLoss(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.lambda_in = kwargs.get("lambda_in", 1)
        self.lambda_rec = kwargs.get("lambda_rec", 1)
        self.lambda_out = kwargs.get("lambda_out", 1)
        self.dt = kwargs.get("dt", 1)

    def forward(self, pred, label):
        """
        Compute the loss
        @param pred: size=(-1, 2), predicted labels
        @param label: size=(-1, 2), labels
        @param dur: duration of the trial
        """
        n_in = self.model.recurrent.input_layer.weight.shape[0]
        n_out = self.model.readout_layer.weight.shape[0]
        n_size = self.model.recurrent.hidden_layer.weight.shape[0]
        mse = torch.square(pred-label).mean()
        loss_in = torch.norm(self.model.recurrent.input_layer.weight, p='fro')/(n_in*n_size)**2
        loss_rec = torch.norm(self.model.recurrent.hidden_layer.weight, p='fro')/(n_size*n_size)**2
        loss_out = torch.norm(self.model.readout_layer.weight, p='fro')/(n_out*n_size)**2
        
        return mse + self.lambda_in*loss_in + self.lambda_rec*loss_rec + self.lambda_out*loss_out