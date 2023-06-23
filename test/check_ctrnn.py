"""
A test script for the CTRNN class.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from nn4n.model import CTRNN

if __name__ == '__main__':
    params = {
        "input_dim": 2,
        "output_dim": 2,
        "hidden_size": 10,
        "dt": 10,
        "tau": 100,
        "ei_balance": "neuron"
    }

    # TODO: add test script