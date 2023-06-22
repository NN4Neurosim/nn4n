import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseTrainer(object):
    def __init__(self, model, optimizer, loss_fn, device, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = kwargs.get('device', torch.device('cpu'))
        self.validate = kwargs.get('validate', False)

    def train(self, train_loader, val_loader, epochs, log_interval):
        for epoch in range(1, epochs + 1):
            self._train_epoch(train_loader, epoch, log_interval)
            if self.validate: self._val_epoch(val_loader, epoch)

    def _train_epoch(self, train_loader, epoch, log_interval):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                output = output.view(-1, 2)
                loss = self.loss_fn(output, target)
                loss.backward

# def train(n_train, task, model, criterion, optimizer, verbose=False):
#     rt_loss, loss_log, norm_log = [], [], []
#     for i in range(n_train):
#         # Generate input/label tensors
#         input, label = task.to_signal(i)

#         # model update
#         optimizer.zero_grad()
#         output, _ = model(input)
#         output = output.view(-1, 2)

#         loss = criterion(output, label)
#         loss.backward()
#         norm_log.append(clip_norm(model))
#         optimizer.step()
#         rt_loss.append(loss.item())

#         if (i+1) % 100 == 0 and i > 0:
#             model.save(os.path.join(CHECKPOINT_PATH, MODEL_TYPE, f"model_{i+1}.pth"))

#         if verbose:
#             if i % 10 == 0:
#                 clear_output(wait=True)

#                 mean_loss = sum(rt_loss)/len(rt_loss)
#                 loss_log.append(mean_loss)

#                 # plot
#                 fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
#                 ax[0].set_title(f"running time loss: {mean_loss}")
#                 ax[0].set_ylabel("training loss")
#                 ax[0].plot(range(0, i+1, 10), loss_log)
#                 ax[0].set_ylim([0, 0.5])
#                 ax[1].set_xlabel("num trials")
#                 ax[1].set_ylabel("gradient norm")
#                 ax[1].plot(norm_log)
#                 ax[1].set_ylim([0, 30])
#                 plt.show()