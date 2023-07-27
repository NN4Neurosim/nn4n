# Neural Networks for Neuroscience
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/nn4n.svg)](https://badge.fury.io/py/nn4n)
<br>
Some of the most commonly used the NN architectures in neuroscience research are included in this project to ease the implementation process.

## Table of contents
- [Install](#install)
    - [Install From GitHub](#install-from-github)
- [Model](#model)
    - [CTRNN (Continuous-Time RNN)](#ctrnn)
- [Structure](#structure)
    - [Multi-Area](#multi-area)
    - [Multi-Area with E/I constraints](#multi-area-with-ei-constraints)
    - [Random Input](#random-input)


## Install
### Install from GitHub
```
git clone https://github.com/zhaozewang/NN4Neurosci.git
```
#### Install using command line
```
cd NN4Neurosci/
python setup.py install
```
#### Install using pip
```
cd NN4Neurosci/
pip install .
```


## Model
### CTRNN
The implementation of standard continuour-time RNN (CTRNN). This implementation also supports adding Excitator-Inhibitory contraints proposed in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792) by Song et al. 2016.

- [Documentation](./docs/CTRNN.md)
- [Examples](./examples/CTRNN.ipynb)

## Structure
The detailed structure (e.g. whether its modular or hierarchical etc.) of any standard 3-layer RNN (as shown below) can be specified using masks in our `model` module implementation. Easy implementations of a few RNN structures is included in the `structure` module.
- [Documentation](./docs/structure.md)

<p align="center"><img src="./img/RNN_structure.png" width="400"></p>

### Multi-Area
The HiddenLayer of a RNN is often defined using a connectivity matrix, depicting a somewhat 'random' connectivity between neurons. The connectivity matrix is often designed to imitate the connectivity of a certain brain area or a few brain areas. When modeling a single brain area, the connectivity matrix is often a fully connected matrix. </br>
However, to model multiple brain areas, it would be more reasonable to use a connectivity matrix with multiple areas. In each areas is densely connected within itself and sparsely connected between areas. The `MultiArea` class in the `structure` module is designed to implement such a connectivity matrix. </br>

- [Examples](./examples/MultiArea.ipynb)

### Multi-Area with E/I constraints
On top of modeling brain with multi-area hidden layer, another critical constraint would be the Dale's law, as proposed in the paper [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792) by Song et al. 2016. The `MultiAreaEI` class in the `structure` module is designed to implement such a connectivity matrix. </br>
This class allows for a much easier implementation of the E/I constraints particularly when there are multiple areas in the hidden layer. It provides flexible control over the excitatory-excitatory, excitatory-inhibitory, inhibitory-excitatory, and inhibitory-inhibitory connections on top of the basic `MultiArea` features. </br>

- [Examples](./examples/MultiArea.ipynb)

### Random Input
Neurons's dynamic receiving input will be heavily driven by the inputting signal. Injecting signal to only part of the neuron will result in more versatile and hierarchical dynamics. See [A Versatile Hub Model For Efficient Information Propagation And Feature Selection](https://arxiv.org/abs/2307.02398) <br>

- Example to be added

## Criterion
### RNNLoss
The loss function is modularized. Each criterion is designed in the format of $`\lambda_L L(\cdot)`$. By default, all $`\lambda_{L}`$ are set to 0 and won't be added to loss (nor the auto-grad tree). By changing the corresponding $`\lambda_{L}`$ to non-zero positive values, the corresponding loss function will be added to the total loss. The total loss is the sum of all loss functions with non-zero $`\lambda_{L}`$.
- [Documentation](./docs/criterion.md)

## Others
For similar projects: 
- [nn-brain](https://github.com/gyyang/nn-brain)

## Acknowledgements
Immense thanks to Christopher J. Cueva for his mentorship in developing this project. This project can't be done without his invaluable help.