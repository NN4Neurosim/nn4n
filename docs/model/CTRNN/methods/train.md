# nn4n.model.CTRNN.train()

[Back to CTRNN](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/index.md) </br>

## Introduction
Set the network to training mode, training will be performed and constraints will be enforced. Also, during training, the recurrent noises (preact_noise and postact_noise) won't be added.

## Usage
```python
import torch
from nn4n.model import CTRNN

inputs = torch.rand(100, 1)
targets = torch.rand(100, 1)

ctrnn = CTRNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

for _ in range(100):
    ctrnn.train()
    outputs, _ = ctrnn(inputs)
    loss = torch.nn.MSELoss()(outputs, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

see also: [eval()](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/eval.md)
