import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from test_utils import get_weight_info, check_activation


# Checker functions
def check_model_params(model, pm):
    assert model.recurrent_layer.input_layer.input_dim == pm["input_dim"], \
        AssertionError(f'Input dimension should be {pm["input_dim"]}')
    assert model.recurrent_layer.hidden_layer.hidden_size == pm["hidden_size"], \
        AssertionError(f'Hidden size should be {pm["hidden_size"]}')
    assert model.readout_layer.output_dim == pm["output_dim"], \
        AssertionError(f'Output dimension should be {pm["output_dim"]}')
    assert model.recurrent_layer.alpha == pm["dt"] / pm["tau"], \
        AssertionError(f'Alpha should be {pm["dt"] / pm["tau"]}')
    assert model.recurrent_layer.hidden_layer.scaling == pm["scaling"], \
        AssertionError(f'Scaling should be {pm["scaling"]}')
    assert model.recurrent_layer.hidden_layer.self_connections == pm["self_connections"], \
        AssertionError(f'Self connections should be {pm["self_connections"]}')
    assert model.recurrent_layer.hidden_layer.positivity_constraints == pm["positivity_constraints"], \
        AssertionError(f'Positivity contraint should be {pm["positivity_constraints"]}')

    input_info = get_weight_info(model.recurrent_layer.input_layer.weight)
    hidden_info = get_weight_info(model.recurrent_layer.hidden_layer.weight)
    readout_info = get_weight_info(model.readout_layer.weight)
    assert input_info["dist"] == pm["layer_distributions"][0], \
        AssertionError(f'Input layer weight should be {pm["layer_distributions"][0]}')
    assert hidden_info["dist"] == pm["layer_distributions"][1], \
        AssertionError(f'Hidden layer weight should be {pm["layer_distributions"][1]}')
    assert readout_info["dist"] == pm["layer_distributions"][2], \
        AssertionError(f'Output layer weight should be {pm["layer_distributions"][2]}')

    assert model.recurrent_layer.input_layer.use_bias == pm["layer_biases"][0], \
        AssertionError(f'Input layer bias should be {pm["layer_biases"][0]}')
    assert model.recurrent_layer.hidden_layer.use_bias == pm["layer_biases"][1], \
        AssertionError(f'Hidden layer bias should be {pm["layer_biases"][1]}')
    assert model.readout_layer.use_bias == pm["layer_biases"][2], \
        AssertionError(f'Output layer bias should be {pm["layer_biases"][2]}')

    if not pm["self_connections"]:
        # only check if self connections are not allowed
        assert hidden_info["self_connections"] == pm["self_connections"], \
            AssertionError(f'Hidden layer should have self connections {pm["self_connections"]}')

    assert check_activation(pm["activation"], model.recurrent_layer.activation), \
        AssertionError(f'Hidden layer activation should be {pm["activation"]}, output value error')
