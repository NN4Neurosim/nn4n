# NN4N: Neural Networks for Neurosimulations

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/nn4n.svg)](https://badge.fury.io/py/nn4n)
[![Downloads](https://static.pepy.tech/badge/nn4n)](https://pepy.tech/project/nn4n)
[![Downloads](https://static.pepy.tech/badge/nn4n/month)](https://pepy.tech/project/nn4n)<br>

<p align="center">
<img src="https://github.com/NN4Neurosim/nn4n/blob/main/docs/images/attractor.png" width="300">
</p>

## [Documentation](https://nn4n.org/)
- [Installation](https://nn4n.org/install/installation/)
- [Quickstart](https://nn4n.org/install/quickstart/)

## About

RNNs are a powerful tool for modeling the dynamics of neural systems. They’ve been applied to a range of cognitive tasks, including working memory, decision-making, and motor control. Despite the strengths, standard RNNs have limitations when it comes to modeling the brain. They often lack biological realism, such as the inclusion of excitatory and inhibitory neurons. Their layer-wise design tends to focus on feedforward connections, leaving out the (potentially important) feedback connections that real neural systems have. On top of that, neurons within each layer are typically homogeneous, which doesn’t reflect the diversity seen in actual brain networks.

This project aims to address these issues by improving the biological plausibility of RNN models. It moves away from layer-based structures and adds controls over connectivity and neuron properties, making the models more flexible and representative of real neural dynamics. It’s also designed to be practical for computational neuroscience, allowing for concise definitions of complex recurrent structures and easy access to hidden layer activity. The goal is to create a tool that is both biologically realistic and straightforward to use for exploring neural systems.

<div style="margin-top: 40px;"></div>

## Network Structures
#### Vanilla CTRNN
A simplistic Vanilla CTRNN contains three layers: an input layer, a hidden layer, and a readout layer, as depicted below.

<p align="center">
  <img src="https://github.com/NN4Neurosim/nn4n/blob/main/docs/images/RNN_structure.png" width="400">
</p>

The yellow nodes represent neurons that project input signals to the hidden layer, the green neurons are in the hidden layer, and the purple nodes represent neurons that read out from the hidden layer neurons. Both input and readout neurons are 'imagined' to be there. I.e., they only project or receive signals and, therefore, do not have activations and internal states.

#### Excitatory-Inhibitory Constrained CTRNN
The implementation of CTRNN also supports Excitatory-Inhibitory constrained continuous-time RNN (EIRNN) similar to what was proposed by H. Francis Song, Guangyu R. Yang, and Xiao-Jing Wang in [Training Excitatory-Inhibitory Recurrent Neural Networks for Cognitive Tasks: A Simple and Flexible Framework](https://doi.org/10.1371/journal.pcbi.1004792)

A visual illustration of the EIRNN is shown below.

<p align="center">
  <img src="https://github.com/NN4Neurosim/nn4n/blob/main/docs/images/EIRNN_structure.png" width="400">
</p>

The yellow nodes denote nodes in the input layer. The middle circle denotes the hidden layer. There are blue nodes and red nodes, representing inhibitory neurons and excitatory neurons, respectively. The depicted network has an E/I ratio of 4/1. The purple nodes are ReadoutLayer neurons.

#### Multi-Area CTRNN
The RNN could also contain multiple areas. Denote the neurons in the hidden layer as \( \mathcal{N} = \{ n_1, n_2, \ldots, n_{N_{hid}} \} \). The neurons within it may be partitioned into multiple areas, \( \mathcal{A} = \{A_1, A_2, \ldots, A_{N_{area}}\} \). The areas are disjoint and their union is the set of all neurons in the hidden layer, i.e., \( \mathcal{N} = \bigcup_{i=1}^{N_{area}} A_i \). Neurons within the same area may be more densely connected and even receive different inputs.

A visual illustration of the Multi-Area CTRNN:

<p align="center">
  <img src="https://github.com/NN4Neurosim/nn4n/blob/main/docs/images/multi_area_structure.png" width="400">
</p>
<div style="margin-top: 40px;"></div>

## Papers Using NN4Neurosim
<h5>
  Time Makes Space: Emergence of Place Fields in Networks Encoding Temporally Continuous Sensory Experiences
  <span style="font-weight: normal;">in</span> 
  <em>NeurIPS 2024</em>
  [<a href="https://zhaozewang.github.io/projects/time_makes_space/">Project</a>]
  [<a href="https://openreview.net/pdf?id=ioe66JeCMF">PDF</a>]
</h5>

<div style="margin-top: 20px;"></div>

```bibtex
@inproceedings{
  wang2024time, 
  title={Time Makes Space: Emergence of Place Fields in Networks Encoding Temporally Continuous Sensory Experiences}, 
  author={Zhaoze Wang and Ronald W. Di. Tullio and Spencer Rooke and Vijay Balasubramanian}, 
  booktitle={Proceedings of the 2024 Conference on Neural Information Processing Systems (NeurIPS)}, 
  year={2024}
} 
```

<div style="margin-top: 30px;"></div>


## Acknowledgement
I would like to thank [Dr. Christopher J. Cueva](https://www.metaconscious.org/author/chris-cueva/) for his mentorship in the original implementation of this project.

## License
This project is licensed under the terms of the MIT license.
