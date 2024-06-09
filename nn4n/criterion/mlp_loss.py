import torch
import torch.nn as nn


class MLPLoss(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self._init_losses(**kwargs)

    def _init_losses(self, **kwargs):
        """
        Initialize the loss functions
        """
        self.lambda_mse = kwargs.get("lambda_mse", 1)

        # the number of loss functions
        lambda_list, loss_list = [], []

        # init lambdas
        # TODO: weight constraints
        lambda_list.append(kwargs.get("lambda_fr", 0))
        lambda_list.append(kwargs.get("lambda_fr_sd", 0))
        # init loss functions
        loss_list.append(self._loss_fr)
        loss_list.append(self._loss_fr_sd)

        # save to self
        self.lambda_list = torch.tensor(lambda_list)
        self.loss_list = loss_list

    def _loss_fr(self, states, **kwargs):
        """ Compute the loss for firing rate """
        # return torch.sqrt(torch.square(states)).mean()
        loss = []
        for s in states:
            l = torch.pow(torch.mean(s, dim=(0, 1)), 2).mean()
            loss.append(l)
        return torch.stack(loss).mean()

    def _loss_fr_sd(self, states, **kwargs):
        """ Compute the loss for firing rate for each neuron in terms of SD """
        # return torch.sqrt(torch.square(states)).mean(dim=(0)).std()
        return torch.pow(torch.mean(states, dim=(0, 1)), 2).std()

    def forward(self, pred, label, **kwargs):
        """
        Compute the loss
        @param pred: size=(-1, batch_size, 2), predicted labels
        @param label: size=(-1, batch_size, 2), labels
        @param dur: duration of the trial
        """
        loss = [self.lambda_mse * torch.square(pred-label).mean()]
        for i in range(len(self.loss_list)):
            if self.lambda_list[i] == 0:
                continue
            else:
                loss.append(self.lambda_list[i]*self.loss_list[i](**kwargs))
        loss = torch.stack(loss)
        return loss.sum(), loss

    def to(self, device):
        """ Move to device """
        super().to(device)
        self.lambda_list = self.lambda_list.to(device)