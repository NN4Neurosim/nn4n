import torch
import torch.nn as nn

from nn4n.model import CTRNN

class RNNLoss(nn.Module):
    """
    Loss function for RNN

    Inputs:
        - model: the model, must be an nn4n.model.CTRNN

    Keyword Arguments:
        - lambda_mse: coefficient for the MSE loss, default: 1
        - lambda_in: coefficient for the input layer loss, default: 0
        - lambda_hid: coefficient for the hidden layer loss, default: 0
        - lambda_out: coefficient for the readout layer loss, default: 0
        - lambda_fr: coefficient for the overall firing rate loss, default: 0
        - lambda_fr_sd: coefficient for the standard deviation of firing rate 
            loss (to evenly distribute firing rate across neurons), default: 0
        - lambda_fr_cv: coefficient for the coefficient of variation of firing
            rate loss (to evenly distribute firing rate across neurons), default: 0
    
    Note that these firing rate does not automatically normalized to a similar magnitude
    """
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        if type(self.model) != CTRNN:
            raise TypeError("model must be CTRNN")
        self._init_losses(**kwargs)

    def _init_losses(self, **kwargs):
        """
        Initialize the loss functions
        """
        self.lambda_mse = kwargs.get("lambda_mse", 1)

        # the number of loss functions
        lambda_list, loss_list = [], []

        # init lambdas
        lambda_list.append(kwargs.get("lambda_in", 0))
        lambda_list.append(kwargs.get("lambda_hid", 0))
        lambda_list.append(kwargs.get("lambda_out", 0))
        lambda_list.append(kwargs.get("lambda_fr", 0))
        lambda_list.append(kwargs.get("lambda_fr_sd", 0))
        lambda_list.append(kwargs.get("lambda_fr_cv", 0))
        # init loss functions
        loss_list.append(self._loss_in)
        loss_list.append(self._loss_hid)
        loss_list.append(self._loss_out)
        loss_list.append(self._loss_fr)
        loss_list.append(self._loss_fr_sd)
        loss_list.append(self._loss_fr_cv)

        # save to self
        self.lambda_list = lambda_list
        self.loss_list = loss_list

        # init constants
        n_in = self.model.recurrent_layer.input_layer.weight.shape[1]
        n_size = self.model.recurrent_layer.hidden_layer.weight.shape[0]
        n_out = self.model.readout_layer.weight.shape[0]
        self.n_in_dividend = n_in*n_size
        self.n_hid_dividend = n_size*n_size
        self.n_out_dividend = n_out*n_size

    def _loss_in(self, **kwargs):
        """ Compute the loss for InputLayer """
        return torch.norm(self.model.recurrent_layer.input_layer.weight, p='fro')**2/self.n_in_dividend

    def _loss_hid(self, **kwargs):
        """ Compute the loss for RecurrentLayer """
        return torch.norm(self.model.recurrent_layer.hidden_layer.weight, p='fro')**2/self.n_hid_dividend

    def _loss_out(self, **kwargs):
        """ Compute the loss for ReadoutLayer """
        return torch.norm(self.model.readout_layer.weight, p='fro')**2/self.n_out_dividend

    def _loss_fr(self, states, **kwargs):
        """ Compute the loss for firing rate """
        return torch.pow(torch.mean(states, dim=(0, 1)), 2).mean()

    def _loss_fr_sd(self, states, **kwargs):
        """ Compute the loss for firing rate for each neuron in terms of SD """
        return torch.pow(torch.mean(states, dim=(0, 1)), 2).std()

    def _loss_fr_cv(self, states, **kwargs):
        """ Compute the loss for firing rate for each neuron in terms of coefficient of variation """
        avg_fr = torch.sqrt(torch.square(states)).mean(dim=0)
        return avg_fr.std()/avg_fr.mean()

    def forward(self, pred, label, **kwargs):
        """
        Compute the loss

        Inputs:
            - pred: size=(-1, batch_size, 2), predicted labels
            - label: size=(-1, batch_size, 2), labels
        
        where -1 is the sequence length
        """
        loss = [self.lambda_mse * torch.square(pred-label).mean()]
        for i in range(len(self.loss_list)):
            if self.lambda_list[i] == 0:
                continue
            else:
                loss.append(self.lambda_list[i]*self.loss_list[i](**kwargs))
        loss = torch.stack(loss)
        return loss.sum(), loss
