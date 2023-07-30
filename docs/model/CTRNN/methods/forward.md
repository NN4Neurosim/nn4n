# nn4n.model.CTRNN.forward()

[Back to CTRNN](https://github.com/zhaozewang/NN4Neurosci/docs/model/CTRNN/index.md) </br>

## Introduction
The forward pass of the CTRNN model. As CTRNN is a child class of `torch.nn.Module`, it can be called either using `ctrnn.forward()` or `ctrnn()`, where `ctrnn` is a instance of `CTRNN`.


## Parameters
> `x` (torch.Tensor): required.
>> The input tensor to the CTRNN model. The shape of the input tensor should be `(seq_len, batch_size, input_dim)`.

## Returns
> `outputs` (torch.Tensor):
>> The output tensor of the CTRNN model. The shape of the output tensor is `(seq_len, batch_size, output_dim)`. Note: when batch_size is 1, the shape of the output tensor is `(seq_len, 1, output_dim)`.

> `hidden_states` (torch.Tensor):
>> The hidden state tensor of the CTRNN model. The shape of the hidden state tensor is `(seq_len, batch_size, hidden_dim)`. Note: when batch_size is 1, the shape of the hidden state tensor is `(seq_len, 1, hidden_dim)`.

## Usage
```python
import torch
from nn4n.model import CTRNN

ctrnn = CTRNN()
outputs, hidden_states = ctrnn(torch.rand(100, 1))

# or

outputs, hidden_states = ctrnn.forward(torch.rand(100, 1))
```
