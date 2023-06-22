# Neural Networks for Neuroscience
Some of the most commonly used the NN architectures in neuroscience research are included in this project to ease the implementation process.

## Table of contents
- [Install](#install)
    - [Install From GitHub](#install-from-github)
- [Model](#model)
    - [CTRNN (Continuous-Time RNN)](#CTRNN)
- [Structure](#structure)
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


## Model
### CTRNN
The implementation of standard continuour-time RNN (CTRNN). This implementation also supports adding Excitator-Inhibitory contraints proposed in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792) by Song et al. 2016.

- [Docs](./docs/CTRNN.md)
- [Examples](./examples/CTRNN.ipynb)

## Structure
The detailed structure (e.g. whether its modular or hierarchical etc.) of any standard 3-layer RNN (as shown below) can be specified using masks in our `model` module implementation. Easy implementations of a few RNN structures is included in the `structure` module.

<p align="center"><img src="./img/RNN_structure.png" width="400"></p>

### Single/Multi-Area RNN w/o EI Constraints
The HiddenLayer of a RNN could be a whole module or could be splitted into few modules. The implementation of both Single-Area RNN and Multi-Area RNN can be easily achieved using the [MultiArea](./nn4n/structure/multi_area.py) class. An Multi-Area RNN that supports E/I constraints is also included in [MultiAreaEI](./nn4n/structure/multi_area_ei.py) class.

- [Docs](./docs/structure.md)
- [Examples](./examples/MultiArea.ipynb)


## Criterion
### Song-Yang Loss function
Loss function proposed in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792) by Song et al. 2016.


## Others
For similar projects: 
- [nn-brain](https://github.com/gyyang/nn-brain)
