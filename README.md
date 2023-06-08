# Neural Networks for Neuroscience
Neural network has been increasingly used as a means to study both computational and theoretical neuroscience in recent years. We include some of the most commonly used the NN architectures in the research of neuroscience to ease the implementation process.

## Table of contents
- [Install](#install)
    - [Install From GitHub](#install-from-github)
- [Models](#models)
    - [CTRNN (Continuous-Time RNN)](#CTRNN)
- [Structures](#structures)
    - [Mult-Area RNN w/o EI Constraints](#singlemulti-area-rnn-wo-ei-constraints)


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


## Models
### CTRNN
We implemented the standard continuour-time RNN (CTRNN). This implementation also supports adding Excitator-Inhibitory contraints proposed in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792) by Song et al. 2016.

- [Docs](./docs/CTRNN.md)
- [Examples](./examples/CTRNN.ipynb)

## Structures
The detailed structure (e.g. whether its modular or hierarchical etc.) of any standard 3-layer RNN (as shown below) can be specified using masks. We provided easy implementations of a few RNN structures, that will generate HiddenLayer masks and its corresponding InputLayer/OutputLayer masks based on a few parameters.

<p align="center"><img src="./img/RNN_structure.png" width="400"></p>

### Single/Multi-Area RNN w/o EI Constraints
The HiddenLayer of a RNN could be a whole module or could be splitted into few modules. The implementation of both Single-Area RNN and Multi-Area RNN can be easily achieved using the [MultiArea](./nn4n/structures/multi_area.py) class. An Multi-Area RNN that supports E/I constraints is also included in [MultiAreaEI](./nn4n/structures/multi_area_ei.py) class.

- [Docs](./docs/structures.md/)
- [Examples](./examples/MultiArea.ipynb)


## Criterion
Under development

## Trainer
Under development


## Others
For similar projects: 
- [nn-brain](https://github.com/gyyang/nn-brain)
