# Continuous-Time RNN
[Back to Home](https://github.com/zhaozewang/NN4Neurosci/blob/main/README.md)

## Table of Contents
- [Introduction](#introduction)
- [Recurrent Layer](#recurrent-layer)
    - [Mathematical Formulation](#mathematical-formulation)
    - [RNN dynamics](#rnn-dynamics)
- [Excitatory-Inhibitory Constrained Continuous-Time RNN](#excitatory-inhibitory-constrained-continuous-time-rnn)
- [Model Structure](#model-structure)
- [Parameters](#parameters)
    - [Structure parameters](#structure-parameters)
    - [Training parameters](#training-parameters)
    - [Constraint parameters](#constraint-parameters)
- [Parameter Specifications](#parameter-specifications)
    - [Pre-activation noise and post-activation noise](#pre-activation-noise-and-post-activation-noise)
    - [Constraints and masks](#constraints-and-masks)
        - [Dale's law](#dales-law)
        - [New synapse](#new-synapse)
        - [Self connections](#self-connections)
- [Methods](#methods)

## Introduction
This is an implementation of the standard Continuous-Time RNN. CTRNN is in the standard 3-layer RNN structure as depicted below:

<p align="center"><img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/basics/RNN_structure.png" width="400"></p>

## Recurrent Layer
### Mathematical Formulation
The firing rate of a single neuron is described by the following equation:

```math
fr_j^t = f(\alpha \sum_{i=0}^{N} w_{ij}fr_{i}^{t-1} + (1-\alpha) v^{t-1})
```

Where $fr_j^t$ is the firing rate of neuron $j$ at time $t$, and $f$ is the activation function. $\alpha$ is the decay constant. $w_{ij}$ is the weight from neuron $i$ to neuron $j$. $v^{t-1}$ is the action potential of neuron $j$ at time $t-1$. $N$ is the number of neurons connected to neuron $j$. </br>

We can generalize the above equation to the entire network:

```math
\vec{fr^t} = f(\alpha W^T \vec{fr^{t-1}} + (1-\alpha) v_j^{t-1})
```

### RNN dynamics
The dynamics of the HiddenLayer is guarded by the standard CTRNN equation:
```math
\tau \frac{d v^t}{dt} = -v^t + W_{hid}^T f(v^t) + W_{in}^T u^t + b_{hid} + \epsilon_t
```

Essentially, this is a generalization of the equations in the above section, with external input $u^t$ and noise added. If we re-write the equation in discrete time, we can get an equation looks more similar to the one we have before:

```math
d v^t = \frac{dt}{\tau} (-v^t + W_{hid}^T f(v^t) + W_{in}^T u^t + b_{hid} + \epsilon_{t, preact})
```

Let $`\space \alpha = \frac{dt}{\tau}`$,

```math
v^{t+1} = v^t + d v^t = v^t + \alpha (-v^t + W_{hid}^T f(v^t) + W_{in}^T u^t + b_{hid} + \epsilon_{t, preact})
```

```math
v^{t+1} = (1-\alpha) v^t + \alpha( W_{hid}^T f(v^t) + W_{in}^T u^t + b_{hid} + \epsilon_{t, preact})
```

```math
fr^{t+1} = f((1-\alpha) v^t + \alpha( W_{hid}^T f(v^t) + W_{in}^T u^t + b_{hid} + \epsilon_{t, preact})) + \epsilon_{t, postact}
```

## Excitatory-Inhibitory constrained continuous-time RNN
The implementation of CTRNN also supports Excitatory-Inhibitory constrained continuous-time RNN (EIRNN). EIRNN is proposed by H. Francis Song, Guangyu R. Yang, and Xiao-Jing Wang in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792)

The original [code](https://github.com/frsong/pycog) is implemented in [Theano](https://pypi.org/project/Theano/) and may be deprecated due to the unsupported Python version. Theano is no longer maintained after Jul 2020. In this repo, the PyTorch version of EIRNN is implemented. It is implicitly included in the CTRNN class and can be enabled by setting `positivity_constraints` to `True` and use appropriate masks.

A visual illustration of the EIRNN is shown below.

<p align="center"><img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/basics/EIRNN_structure.png" width="400"></p>


The yellow nodes denote nodes in the input layer. The middle circle denotes the hidden layer. There are blue nodes and red nodes, representing inhibitory neurons and excitatory neurons, respectively. The depicted network has an E/I ratio of 4/1. The purple nodes are ReadoutLayer neurons. The network structure is as follows:

## Model Structure
```
├── CTRNN
│   ├── RecurrentLayer
│   │   ├── InputLayer (class LinearLayer)
│   │   ├── HiddenLayer
│   ├── Readout_areas (class LinearLayer)
```
At the begining of each trial, the hidden states are set to zero. The RecurrentLayer, which contains the InputLayer and the HiddenLayer, is updated T/dt times during each trial. Finally, T/dt of hidden states are mapped out by the ReadoutLayer.<br>
For more details, refer to [Song et al. 2016](https://doi.org/10.1371/journal.pcbi.1004792).

## Parameters
### Structure parameters
These parameters primarily determine the structure of the network. It is recommended to check these parameters before initializing the network.
| Parameter                | Default       | Type                                | Description                                |	
|:-------------------------|:-------------:|:-----------------------------------:|:-------------------------------------------|
| input_dim                | 1              | `int`                               | Input dimension                  |
| output_dim               | 1              | `int`                               | Output dimension                  |
| hidden_size              | 100            | `int`                               | Number of hidden nodes                      |
| scaling                  | 1.0           | `float`                             | Scaling factor for the hidden weights, it will scale the hidden weight by $`\frac{scaling}{\sqrt{N\_{hid}}}`$. Won't be affected when the HiddenLayer distribution is `uniform`.   |
| self_connections         | False         | `boolean`                           | Whether a neuron can connect to itself          |
| activation               | 'relu'        | 'relu'/'tanh'/'sigmoid'/'retanh'    | Activation function                   |
| layer_distributions      | ['uniform', 'normal', 'uniform']      | `string`/`list`            | Layer distributions. Either `string` or a `list` of three elements. The `string` or `list` element must be either 'uniform', 'normal', or 'zero'. If the given value is a `string`, it will set all three layers to the given distribution. If the provided value is a `list` of three elements, from the first to the last, corresponding to the distribution of the InputLayer, HiddenLayer, and ReadoutLayer, respectively.       |
| layer_biases             | [True, True, True] | `boolean` or `list`  | Whether to use bias in each layer. Either a `boolean` or a `list` of three `boolean`s. If the given value is a list, from the first element to the last element, corresponding to the InputLayer, HiddenLayer, and ReadoutLayer, respectively. |


### Training parameters
These parameters primarily determine the training process of the network. The `tau` and `dt` parameters are used to discretize the continuous-time dynamics. It is **highly recommended** to check these parameters before training. They have a significant impact on the training result.
| Parameter                | Default       | Type                                | Description                                |	
|:-------------------------|:-------------:|:-----------------------------------:|:-------------------------------------------|
| tau                      | 100           | `int`                               | Time constant                              |
| dt                       | 10            | `int`                               | Constant that used to discretize time      |
| preact_noise             | 0             | `float`                             | Whether to add zero-mean Gaussian preactivation noise during training. The noise is added before the activation function is applied. See difference between `preact_noise` and `postact_noise` [here](#preact_noise-and-postact_noise). |
| postact_noise            | 0             | `float`                             | Whether to add zero-mean Gaussian postactivation noise during training. The noise is added after the activation function is applied. See difference between `preact_noise` and `postact_noise` [here](#preact_noise-and-postact_noise). |
| init_state               | 'zero'        | 'zero', 'keep', 'learn'             | Method to initialize the hidden states. 'zero' will set the hidden states to zero at the beginning of each trial. 'keep' will keep the hidden states at the end of the previous trial. 'learn' will learn the initial hidden states. **Note:** 'keep' hasn't been tested yet. |

### Constraint parameters
These parameters primarily determine the constraints of the network. By default, the network is initialized using the most lenient constraints, i.e., no constraints being enforced.
| Parameter                | Default       | Type                       | Description                                |	
|:-------------------------|:-------------:|:--------------------------:|:-------------------------------------------|
| positivity_constraints    | False         | `boolean`/`list`           | Whether to enforce Dale's law. Either a `boolean` or a `list` of three `boolean`s. If the given value is a list, from the first element to the last element, corresponds to the InputLayer, HiddenLayer, and ReadoutLayer, respectively. |
| sparsity_constraints  | True         | `boolean`/`list`            | Whether a neuron can grow new connections. See [constraints and masks](#constraints-and-masks). If it's a list, it must have precisely three elements. Note: this must be checked even if your mask is sparse, otherwise the new connection will still be generated                  |
| layer_masks              | `None` or `list` | `list` of `np.ndarray`               | Layer masks if `sparsity_constraints/positivity_constraints is set to true. From the first to the last, the list elements correspond to the mask for Input-Hidden, Hidden-Hidden, and Hidden-Readout weights, respectively. Each mask must have the same dimension as the corresponding weight matrix. See [constraints and masks](#constraints-and-masks) for details.                   |


## Parameter Specifications
### Pre-activation noise and post-activation noise
When no noise is added, the network dynamic update as follows:
```math
v_{t+1} = (1 - \alpha) v_t + \alpha (W_{hid}^T f(v_t) + W_{in}^T u_t + b_{hid})
```
Where $v$ is the pre-activation state, i.e., neurons' action potential. $`f(\cdot)`$ is the activation function, and $u_t$ is the input.
#### Pre-activation noise
If the noise is added before the activation function is applied, then $v_{t+2}$ will be:
```math
v_{t+2, preact} = (1 - \alpha) (v_{t+1} + \epsilon_t) + \alpha (W_{hid}^T f(v_{t+1} + \epsilon_t) + W_{in}^T u_{t+1} + b_{hid})
```
then add noise for $t+2$
#### Post-activation noise
If the noise is added after the activation function is applied, the $v_{t+2}$ will be:
```math
v_{t+2, postact} = (1 - \alpha) v_{t+1} + \alpha (W_{hid}^T (f(v_{t+1}) + \epsilon_t) + W_{in}^T u_{t+1} + b_{hid})
```
then add noise for $t+2$

### Constraints and masks
Constraints are enforced before each forward pass
#### Dale's law:
Masks (input, hidden, and output) cannot be `None` if `positivity_constraints` is `True`.<br>
Only entry signs matter for the enforcement of Dale's law. All edges from the same neuron must be all excitatory or all inhibitory. This is enforced across training using the `relu()` and `-relu()` functions.<br>
When `positivity_constraints` is set to true, it will automatically balance the excitatory/inhibitory such that all synaptic strengths add up to zero.
#### New synapse:
`sparsity_constraints` defines whether a neuron can 'grow' new connections.<br>
If plasticity is set to False, neurons cannot 'grow' new connections. A mask must be provided if `sparsity_constraints` is set to False.<br>
Only zeros entries matter. All entries that correspond to a zero value in the mask will remain zero across all time.
#### Self connections:
Whether a neuron can connect to itself. This feature is enforced along with the `sparsity_constraints` mask. If mask is not specified but `self_connections` is set, a mask that only has zero entires on the diagonal will be generated automatically.

## Methods
| Method                                        | Description                                |
|:----------------------------------------------|:-------------------------------------------|
| [`forward()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/forward.md)           | Forward pass                               |
| [`save()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/save.md)                 | Save the network to a given path.          |
| [`load()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/load.md)                 | Load the network from a given path.        |
| [`print_layers()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/print_layers.md) | Print the network architecture and layer-by-layer specifications |
| [`train()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/train.md)               | Set the network to training mode, training will be performed and constraints will be enforced. Also, during training, the recurrent noises (preact_noise and postact_noise) won't be added.  |
| [`eval()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/model/CTRNN/methods/eval.md)                 | Set the network to evaluation mode, no training will be performed and no constraints will be enforced |

## Todos
- [x] Test different activation functions
- [x] Bias when using Dale's law?
- [ ] If the masks are not set, there need default values.
- [x] Potentially user can choose to enforce `sparsity_constraints` or not for a specific layer
- [x] Re-write Dale's law such that it can still work when `sparsity_constraints` is not enforced.
- [x] Can InputLayer and ReadoutLayer weights be negative when Dale's law is enforced?
- [x] Check if bias does not change when use_bias = False
- [x] Merge hidden_bias, input_bias, readout_bias to a single parameter
- [x] Merge hidden_dist, input_dist, readout_dist to a single parameter
- [x] Check different exc_pct
- [x] Optimize constraints parameters
