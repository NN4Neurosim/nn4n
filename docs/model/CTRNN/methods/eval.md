# nn4n.model.CTRNN.eval()

[Back to CTRNN](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/index.md) </br>

## Introduction
Set the network to training mode, training will be performed and constraints will be enforced. Also, during training, the recurrent noises (preact_noise and postact_noise) won't be added.

## Usage

```python
import torch
from nn4n.model import CTRNN

ctrnn = CTRNN()
inputs = torch.rand(100, 1)

ctrnn = CTRNN()

for _ in range(100):
    ctrnn.eval()
    outputs, _ = ctrnn(inputs)
```

see also: [train()](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/train.md)
