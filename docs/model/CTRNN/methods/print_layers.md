# nn4n.model.CTRNN.print_layers()

[Back to CTRNN](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/index.md) </br>

## Introduction
Print out all layer details of the CTRNN.

## Usage
```python
from nn4n.model import CTRNN

ctrnn = CTRNN()
ctrnn.print_layers()
```
Output
```
Linear Layer: 
   | input_dim:  1
   | output_dim: 100
   | dist:       uniform
   | bias:       True
   | shape:      torch.Size([100, 1])
   | weight_min: -0.9917911291122437
   | weight_max: 0.9899972677230835
   | bias_min:   0.0
   | bias_max:   0.0
   | sparsity:   1
```

<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/model/ctrnn_input.png" width="500"></br>
<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/model/ctrnn_input_dist.png" width="500"></br>

```
Recurrence: 
   | hidden_min:    0.0
   | hidden_max:    0.0
   | hidden_mean:   0.0
   | preact_noise:  0
   | postact_noise: 0
   | activation:    relu
   | alpha:         0.1

Hidden Layer: 
   | self_connections: False
   | input/output_dim: 100
   | distribution:     normal
   | bias:             True
   | dale:             False
   | shape:            torch.Size([100, 100])
   | weight_min:       -0.3491513133049011
   | weight_max:       0.3499620854854584
   | weight_mean:      -0.0004077071789652109
   | bias_min:         0.0
   | bias_max:         0.0
   | sparsity:         0.9900000095367432
   | scaling:          1.0
```

<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/model/ctrnn_hidden.png" width="500"></br>
<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/model/ctrnn_hidden_dist.png" width="500"></br>
<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs//images/model/ctrnn_hidden_eig.png" width="300"></br>

```
Linear Layer: 
   | input_dim:  100
   | output_dim: 1
   | dist:       uniform
   | bias:       True
   | shape:      torch.Size([1, 100])
   | weight_min: -0.09865634888410568
   | weight_max: 0.0993180200457573
   | bias_min:   0.0
   | bias_max:   0.0
   | sparsity:   1
```

<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/model/ctrnn_readout.png" width="500"></br>
<img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/model/ctrnn_readout_dist.png" width="500"></br>
