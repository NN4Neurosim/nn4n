# RNN structure(s)
[Back to Home](https://github.com/zhaozewang/NN4Neurosci/blob/main/README.md)
## Table of Contents
- [Introduction](#introduction)
- [Structures](#structures)
    - [BaseStruct](#basestruct)
        - [Parameters](#basestruct-parameters)
        - [Methods](#basestruct-methods)
    - [Multi Area](#multiarea)
        - [Parameters](#multiarea-parameters)
        - [Forward Backward Specifications](#forward-backward-specifications)
    - [Multi Area EI](#multiareaei)
        - [Parameters](#multiareaei-parameters)
        - [Inter-Area Connections Under EI Constraints](#inter-area-connections-under-ei-constraints)
    - [Random Input](#randominput)
        - [Parameters](#randominput-parameters)

## Introduction
This module defines structures for any RNN in the standard 3-layer architectures (as shown below). The structures of the hidden layer in this project are defined using masks. Therefore, classes in this module will generate input_mask, hidden_mask, and readout_mask that are used in the `model` module<br>

<p align="center"><img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/basics/RNN_structure.png" width="400"></p>

Where yellow nodes are in the InputLayer, green nodes are in the HiddenLayer, and purple nodes are in the ReadoutLayer.

## Structures
### BaseStruct
Base class for all structures. It defines the basic structure of a RNN. It serves as a boilerplate for other structures. It is not meant to be used directly.

#### BaseStruct Parameters
| Parameter            | Default       | Required      | Type             | Description                                |
|:---------------------|:-------------:|:-------------:|:----------------:|:-------------------------------------------|
| input_dim            | `None`        | True          |`int`              | Input dimension. Used to generate the input layer mask. |
| hidden_size          | `None`        | True          | `int`              | HiddenLayer size. Used to generate the hidden layer mask. |
| output_dim           | `None`        | True          | `int`              | Output dimension. Used to generate the readout layer mask. |

#### BaseStruct Methods
Methods that are shared by all structures. <br>
| Method                                               | Description                                         |
|:-----------------------------------------------------|:----------------------------------------------------|
| [`get_input_idx()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_input_idx.md)      | Get indices of neurons that receive input.          |
| [`get_readout_idx()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_readout_idx.md)  | Get indices of neurons that readout from.           |
| [`get_non_input_idx()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_non_input_idx.md) | Get indices of neurons that don't receive input. |
| [`visualize()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/visualize.md)              | Visualize the generated masks.                      |
| [`masks()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/masks.md)                      | Return a list of np.ndarray masks. It will be of length 3, where the first element is the input mask, the second element is the hidden mask, and the third element is the readout mask. For those structures that do not have specification for a certain mask, it will be an all-one matrix. |
| [`get_areas()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_areas.md)              | Get a list of areas names.                 | 
| [`get_area_idx()`](https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/structure/methods/get_area_idx.md)        | Get indices of neurons in a specific area. The parameter `area` could be either a string from the `get_areas()` or a index of the area. |


### MultiArea
See [Examples](https://github.com/zhaozewang/NN4Neurosci/blob/main/examples/MultiArea.ipynb) <br>
This will generate a multi-area RNN without E/I constraints. Therefore, by default, the input/hidden/readout masks are binary masks. Use cautious when the `use_dale` parameter of CTRNN is set to `True`, because it will make all neurons to be excitatory.
**NOTE:** This also implicitly covers single area case. If `n_area` is set to 1. All other parameters that conflict this setting will be ignored.
#### MultiArea Parameters
| Parameter            | Default       | Required      | Type                      | Description                                |
|:---------------------|:-------------:|:-------------:|:-------------------------:|:-------------------------------------------|
| n_areas              | 2             | False         |`int` or `list`            | Number of areas.<br>- If `n_areas` is an integer, `n_areas` must be a divisor of `hidden_size`. It will divide the HiddenLayer into three equal size regions.<br>- If `n_areas` is a list, it must sums up to `hidden_size`, where each element in the list denote the number of neurons in that area.   |
| area_connectivities  | [0.1, 0.1]    | False         |`list` or `np.ndarray`     | Area-to-area connection connectivity. Entries must between `[0,1]`<br>- If its a list of two elements, the first element is the forward connectivity, and the second is the backward connectivity. The within-area connectivity will be 1.<br>- If its a list of three elements, the last element will be the within-area connectivity.<br>- If `area_connectivities` is an `np.ndarray`, it must be of shape (`n_areas`, `n_areas`). See [forward/backward specifications](#forward-backward-specifications)|
| input_areas          | `None`        | False         |`list` or `None`          | Areas that receive input. If set to `None`, all neurons will receive inputs. If set to a `list`, list elements should be the index of the areas that receive input. Set it to a list of one element if only one area receives input. | 
| readout_areas        | `None`        | False         |`list` or `None`         | Areas that readout from. If set to `None`, all neurons will readout from. If set to a `list`, list elements should be the index of the areas that readout from. Set it to a list of one element if only one area readout from. |


| Attributes               | Type                       | Description                                |	
|:-------------------------|:--------------------------:|:-------------------------------------------|
| n_areas                  | `int`                      | Number of areas                            |
| node_assignment          | `list`                     | Nodes area assignment                      |
| hidden_size              | `int`                      | Number of nodes in the HiddenLayer         |
| input_dim                | `int`                      | Input dimension                            |
| output_dim               | `int`                      | Output dimension                           |
| area_connectivities      | `np.ndarray`               | Area-to-area connectivity matrix. If it is a list in params, it will be transformed into a numpy matrix after initialization                   |

#### Forward Backward Specifications
RNNs can be implemented in various ways, in this library,
$$s W^T + b$$
is used in the HiddenLayer forward pass, where $W$ is the connectivity matrix of the HiddenLayer and $s$ is the current HiddenLayer state.<br>
$W$ may not matter if your connectivity matrix is symmetric. But if it's not, you might want to pay attention to the forward connections and backward connections. In the figure below, three networks (`n_areas` = 2, 3, 4) and their corresponding forward/backward connection matrix are provided. The blue regions are intra-area connectivity, the green regions are forward connections, and the red regions are backward connections.

<p align="center"><img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/basics/Multi_Area.png" width="700"></p>

<!-- ![area_connectivities](../img/Multi_Area_Transpose.png) -->


### MultiAreaEI
[Examples](https://github.com/zhaozewang/NN4Neurosci/blob/main/examples/MultiArea.ipynb) <br>
This class is a child class of `MultiArea`. It will generate a multi-area RNN with E/I constraints. Therefore, by default, the input/hidden/readout masks are signed masks. Use cautious as it will change the sign of the weights. 
#### MultiAreaEI Parameters
| Parameter                     | Default                 | Type                       | Description                                |
|:------------------------------|:-----------------------:|:--------------------------:|:-------------------------------------------|
| ext_pct                       | 0.8                     | `float`                    | Percentage of excitatory neurons              |
| inter_area_connections        |[True, True, True, True] | `list` (of booleans)       | Allows for what type of inter-area connections. `inter_area_connections` must be a `boolean` list of 4 elements, denoting whether 'exc-exc', 'exc-inh', 'inh-exc', and 'inh-inh' connections are allowed between areas. see [inter-area connections under EI constraints](#inter-area-connections-under-ei-constraints). |
| inh_readout                   | True                     | `boolean`                 | Whether to readout inhibitory neurons              |


#### Inter-Area Connections Under EI Constraints
Depending on the specific problem you are investigating on, it is possible that you want to eliminate inhibitory connections between areas. Or, you might not want excitatory neurons to connect to inhibitory neurons in other areas. See figure below for different cases of inter-area connections under EI constraints.

<p align="center"><img src="https://github.com/zhaozewang/NN4Neurosci/blob/main/docs/images/basics/Multi_Area_EI.png" width="550"></p>

To specify what kinds of inter-area connections you want to keep, simple pass a 4-element boolean list to `inter_area_connections`. The 4 elements denotes whether to keep inter-area 'exc-exc', 'exc-inh', 'inh-exc', and 'inh-inh' connections.

### RandomInput
Randomly inject input to the network. Neurons' dynamic receiving input will be heavily driven by the inputting signal. Injecting signal to only part of the neuron will result in more versatile and hierarchical dynamics. See [A Versatile Hub Model For Efficient Information Propagation And Feature Selection](https://arxiv.org/abs/2307.02398) <br>

#### RandomInput Parameters
| Parameter                     | Default                 | Type                       | Description                                |
|:------------------------------|:-----------------------:|:--------------------------:|:-------------------------------------------|
| input_spar                    | 1                       | `float`                    | Input sparsity. Percentage of neurons that receive input.             |
| readout_spars                 | 1                       | `float`                    | Readout sparsity. Percentage of neurons that readout from.              |
| hidden_spar                   | 1                       | `float`                    | Hidden sparsity. Percentage of edges that are non-zero.                |
| overlap                       | `True`                  | `boolean`                  | Whether to allow overlap between input and readout neurons.            |