import torch
import torch.nn as nn

class RNNLoss(nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self._init_losses(**kwargs)
        self.dt = kwargs.get("dt", 10)
    
    def _init_losses(self, **kwargs):
        """
        Initialize the loss functions
        """
        # the number of loss functions
        n_losses = 5

        # init lambdas
        lambda_list = [0] * n_losses
        lambda_list[0] = kwargs.get("lambda_in", 0)
        lambda_list[1] = kwargs.get("lambda_rec", 0)
        lambda_list[2] = kwargs.get("lambda_out", 0)
        lambda_list[3] = kwargs.get("lambda_met", 0)
        lambda_list[4] = kwargs.get("lambda_fr", 0)
        self.lambda_list = lambda_list

        # init loss functions
        loss_list = [None] * n_losses
        loss_list[0] = self._loss_in
        loss_list[1] = self._loss_rec
        loss_list[2] = self._loss_out
        loss_list[3] = self._loss_met
        loss_list[4] = self._loss_fr
        self.loss_list = loss_list

        # init constants
        n_in = self.model.recurrent.input_layer.weight.shape[0]
        n_size = self.model.recurrent.hidden_layer.weight.shape[0]
        n_out = self.model.readout_layer.weight.shape[0]
        self.n_in_dividend = n_in*n_size
        self.n_rec_dividend = n_size*n_size
        self.n_out_dividend = n_out*n_size


    def _loss_in(self, **kwargs):
        """
        Compute the loss for input layer
        """
        return torch.norm(self.model.recurrent.input_layer.weight, p='fro')**2/self.n_in_dividend
    

    def _loss_rec(self, **kwargs):
        """
        Compute the loss for recurrent layer
        """
        return torch.norm(self.model.recurrent.hidden_layer.weight, p='fro')**2/self.n_rec_dividend


    def _loss_out(self, **kwargs):
        """
        Compute the loss for readout layer
        """
        return torch.norm(self.model.readout_layer.weight, p='fro')**2/self.n_out_dividend
    

    def _loss_fr(self, states, **kwargs):
        """
        Compute the loss for firing rate
        """
        return torch.square(states).mean()


    def _loss_met(self, states, **kwargs):
        """
        Compute the loss for metabolic states
        """
        # return torch.square(states).mean()
        l1_loss_per_timestep = torch.norm(states, p=1, dim=1)
        return l1_loss_per_timestep.mean()


    def forward(self, pred, label, **kwargs):
        """
        Compute the loss
        @param pred: size=(-1, 2), predicted labels
        @param label: size=(-1, 2), labels
        @param dur: duration of the trial
        """
        loss = torch.square(pred-label).mean()
        print('mse loss', loss)
        for i in range(len(self.loss_list)):
            if self.lambda_list[i] == 0:
                continue
            else:
                loss_ = self.lambda_list[i]*self.loss_list[i](**kwargs)
                print('other loss', loss_)
                loss += loss_
        return loss
        
