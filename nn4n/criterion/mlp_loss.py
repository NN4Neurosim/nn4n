import torch
import torch.nn as nn


class MLPLoss(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.norm_loss = kwargs.get("norm_loss", False)
        self._init_losses(**kwargs)

    def _init_losses(self, **kwargs):
        """
        Initialize the loss functions
        """
        # the number of loss functions
        lambda_list, loss_list = [], []

        # init lambdas
        lambda_list.append(kwargs.get("lambda_mse", 1))
        lambda_list.append(kwargs.get("lambda_in", 0))
        lambda_list.append(kwargs.get("lambda_out", 0))
        lambda_list.append(kwargs.get("lambda_fr", 0))
        lambda_list.append(kwargs.get("lambda_fr_sd", 0))
        # init loss functions
        loss_list.append(self._loss_mse)
        loss_list.append(self._loss_in)
        loss_list.append(self._loss_out)
        loss_list.append(self._loss_fr)
        loss_list.append(self._loss_fr_sd)

        # save to self
        self.lambda_list = torch.tensor(lambda_list)
        self.loss_list = loss_list

        # if norm_loss, keep track of the last 50 losses
        if self.norm_loss:
            self.losses = []
            self.norm = 1

        # init constants
        n_in = self.model.input_layer.weight.shape[1]
        n_size = self.model.input_layer.weight.shape[0]
        n_out = self.model.readout_layer.weight.shape[0]
        self.n_in_dividend = n_in*n_size
        self.n_out_dividend = n_out*n_size

    def _loss_mse(self, pred, label, **kwargs):
        """ Compute the MSE loss """
        return torch.square(pred-label).mean()

    def _loss_in(self, **kwargs):
        """ Compute the loss for InputLayer """
        return torch.norm(self.model.input_layer.weight, p='fro')**2/self.n_in_dividend

    def _loss_out(self, **kwargs):
        """ Compute the loss for ReadoutLayer """
        return torch.norm(self.model.readout_layer.weight, p='fro')**2/self.n_out_dividend

    def _loss_fr(self, states, **kwargs):
        """ Compute the loss for firing rate """
        # return torch.sqrt(torch.square(states)).mean()
        return torch.pow(torch.mean(states, dim=(0, 1)), 2).mean()

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
        loss=[]
        for i in range(len(self.loss_list)):
            if self.lambda_list[i] == 0:
                loss.append(torch.tensor(0.0).to(pred.device))
            else:
                loss.append(self.loss_list[i](pred=pred, label=label, **kwargs))
        loss = torch.stack(loss)

        # if self.norm_loss:
        #     self.losses.append(loss.detach().clone())
        #     if len(self.losses) % 20 == 0:
        #         # if len(self.losses) > 100:
        #         #     self.losses = self.losses[-100:]
        #         self.norm = torch.stack(self.losses).mean(dim=0)
        #     # if len(self.losses) == 11:
        #     #     # self.losses = self.losses[10:]
        #     #     self.norm = torch.stack(self.losses).mean(dim=0)
        #     loss = (loss-self.norm)/(self.norm.mi)
        
        loss = loss * self.lambda_list
        return loss.sum(), loss

    def to(self, device):
        """ Move to device """
        super().to(device)
        self.lambda_list = self.lambda_list.to(device)