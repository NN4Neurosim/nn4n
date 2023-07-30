"""
Test file for the base vanilla RNN.
Item to be tested:
    - CTRNN construction
"""
from structure_checker_base import check_model_params
from nn4n.model import CTRNN

# Constants
MODEL_BASE_PM = {
    "input_dim": 1,
    "hidden_size": 100,
    "output_dim": 1,
    "tau": 100,
    "scaling": 1.0,
    "dt": 10,
    "self_connections": False,
    "use_dale": False,
    "layer_distributions": ["uniform", "normal", "uniform"],
    "layer_biases": [True, True, True],
    "self-connections": False,
    "activation": "relu",
    "allow_negative": [True, True, True],
}


def test_vanilla_rnn():
    model = CTRNN()
    check_model_params(model, MODEL_BASE_PM)


if __name__ == "__main__":
    test_vanilla_rnn()
