# Excitatory-Inhibitory Continuous-Time RNN
## Introduction
This is the implimentation of E/I constraint RNN (EIRNN) originally proposed by H. Francis Song, Guangyu R. Yang, and Xiao-Jing Wang in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792)

The original [code](https://github.com/frsong/pycog) is implemented in [Theano](https://pypi.org/project/Theano/) and is deprecated due to the unsupported Python version. Theano is no longer maintained after Jul 2020. Hence, we re-implemented the method proposed by Song et al. using PyTorch.

EIRNN is a derivative of continuous-time RNN (CTRNN), therefore CTRNN is implicitly included in this module.

## Model Structure
A visual illustration of the EIRNN is shown below.

![EIRNN Structure](../img/EIRNN_structure.png)

The yellow nodes denotes nodes in the input layer. The middle circle denotes the hidden layer. There are blue nodes and red nodes, representing inhibitory neurons and ecitatory neurons, respectively. The depicted network has E/I ratio of 4/1. The purple nodes are readout layer neurons. The network structure is as follows:
```
├── EIRNN
│   ├── RecurrentLayer
│   │   ├── InputLayer (class LinearLayer)
│   │   ├── HiddenLayer
│   ├── OutputLayer (class LinearLayer)
```
At the begining of each trial, the hidden states are set to zero. The recurrent layer, which contains the InputLayer and the HiddenLayer, is updated T/dt times during each trial. Finally, T/dt of hidden states are mapped out by the OutputLayer.<br>
For more details, refer to [Song et al. 2016](https://doi.org/10.1371/journal.pcbi.1004792).

## Parameters
| Parameter                | Default       | Type                       | Description                                |	
|:-------------------------|:-------------:|:--------------------------:|:-------------------------------------------|
| tau                      | 1             | `float`                    | Time constant                              |
| dt                       | 1             | `float`                    | Constant that used to discretize time      |
| use_dale                 | False         | `boolean`                  | Enfore Dale's law or not. Dale's law will only be enforced on the HiddenLayer and the OutputLayer                                                            |
| plasticity               | False         | `boolean`                  | Enforce plasticity or not. That is, whether a neuron can grow new connections. See [Constraints and masks](#constraints-and-masks).                             |
| input_size               | 1             | `int`                      | Input dimension                            |
| input_dist               | 'uniform'     | 'uniform'/'normal'         | InputLayer Distribution                    |
| input_bias               | False         | `Boolean`                  | Use bias or not for InputLayer             |
| input_mask               | `None`        | `np.ndarray`               | InputLayer mask for plasticity/dale's law. Must has the same dimension as the input weight. See [Constraints and masks](#constraints-and-masks).               |
| recurrent_noise          | 0.05          | `float`                    | Zero-mean Gaussian recurrent noise         |
| self_connections         | False         | `boolean`                  | Whether a neuron can connect to itself     |
| activation               | 'relu'        | 'relu'/'tanh'/'sigmoid'    | Activation function                        |
| spec_rad                 | 1             | `float`                    | HiddenLayer spectral radius                |
| hidden_size              | 100           | `int`                      | Number of hidden nodes                     |
| hidden_dist              | 'normal'      | 'uniform'/'normal'         | HiddenLayer Distribution                   |
| hidden_bias              | False         | `boolean`                  | Use bias or not for HiddenLayer            |
| hidden_mask              | `None`        | `np.ndarray`               | HiddenLayer mask for plasticity/dale's law. Must has the same dimension as the hidden weight. See [Constraints and masks](#constraints-and-masks).              |
| output_size              | 1             | `int`                      | Output dimension                           |
| output_dist              | 'uniform'     | 'uniform'/'normal'         | OutputLayer Distribution                   |
| output_bias              | False         | `boolean`                  | Use bias or not for OutputLayer            |
| output_mask              | `None`        | `np.ndarray`               | OutputLayer mask for plasticity/dale's law. Must has the same dimension as the output weight. See [Constraints and masks](#constraints-and-masks).              |


### Constraints and masks
Constraints are enforced before each forward pass
#### Dale's Law:
Masks (input, hidden, and output) cannot be `None` if `use_dale` is `True`.<br>
Only entries signs matter for the enforcement of Dale's law. All edges from the same neuron must be all excitatory or inhibitory. This is enforced across training using the `relu()` and `-relu()` functions.
#### Plasticity:
Plasticity defines whether a neuron can 'grow' new connections.<br>
Only zeros entries matter if plasticity is enforced. All zero entires in `input_mask` will be removed.<br>
Currently the plasticity is enforced globally. That is, if `plasticity` is set to be True, it will be enforced on all three layers. We will consider adding separate plasticity for different layers in the future.
#### Self-Connections:
Whether a neuron can connect to itself. This is enforced along with the plasticity mask. If mask is undefined, it will automatically generate a mask that has zero entires only on the diagonal.


## Todos
- [ ] Load in connectivity matrices
- [ ] Test different activation functions
- [ ] Bias when using dale's law?
- [ ] If the masks are not set, there need default values.
- [ ] Potentially user can choose to enfore plasticity or not for a specific layer
- [x] Re-write Dale's law such that it can still work when plasticity is not enforced.
- [ ] Can InputLayer and OutputLayer weights be negative when Dale's law enforced?