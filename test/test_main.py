# import torch
# import numpy as np
# from nn4n.model import CTRNN
# from nn4n.structure import MultiArea


# # Constants ===================================================================
# SPARSE = 1
# MODEL_PARAMS = {
#     # size
#     "input_dim": 1,
#     "hidden_size": 200,
#     "output_dim": 1,

#     # hyperparameters
#     "tau": 100,
#     "scaling": 1.0,
#     "dt": 10,
#     "activation": "relu",
#     "preact_noise": 0,
#     "postact_noise": 0,
#     "self_connections": False,

#     # bias and distribution
#     "layer_biases": [False, False, False],
#     "layer_distributions": ['uniform', 'normal', 'uniform'],
# }
# STRUCT_PARAMS = {
#     'n_areas': 2,
#     'hidden_size': 200,
#     'input_dim': 1,
#     'output_dim': 1,
#     'inh_readout': False,
#     'input_areas': [0],
#     'readout_areas': [1],
# }


# # Checker functions ============================================================
# def test_models(logger):
#     check_vanilla_rnn(logger)
#     check_forward_pass(logger)
#     check_struct(logger)
#     check_fwd_bwd(logger)


# def check_vanilla_rnn(logger):
#     # check base CTRNN construction
#     logger.info('Testing CTRNN construction...')
#     try:
#         CTRNN()
#     except Exception as e:
#         logger.error('Error in constructing vanilla CTRNN without parameters.')
#         raise e


# def check_vanilla_rnn_with_parameters(logger):
#     # check CTRNN construction with parameters
#     logger.info('Testing CTRNN construction with parameters...')
#     try:
#         rnn = CTRNN(**MODEL_PARAMS)

#         _input_layer, _hidden_layer, _readout_layer = rnn.recurrent_layer.input_layer, rnn.recurrent_layer.hidden_layer, rnn.readout_layer
#         assert torch.all(_input_layer.bias == 0), AssertionError('Input layer bias should be all zeros.')
#         assert torch.all(_hidden_layer.bias == 0), AssertionError('Hidden layer bias should be all zeros.')
#         assert torch.all(_readout_layer.bias == 0), AssertionError('Readout layer bias should be all zeros.')
#         # assert all bias does not require grad
#         assert not _input_layer.bias.requires_grad, AssertionError('Input layer bias should not require grad.')
#         assert not _hidden_layer.bias.requires_grad, AssertionError('Hidden layer bias should not require grad.')
#         assert not _readout_layer.bias.requires_grad, AssertionError('Readout layer bias should not require grad.')

#         assert _input_layer.weight.shape == (200, 1), AssertionError('Input layer weight shape should be (200, 1).')
#         assert _hidden_layer.weight.shape == (200, 200), AssertionError('Hidden layer weight shape should be (200, 200).')
#         assert _readout_layer.weight.shape == (1, 200), AssertionError('Readout layer weight shape should be (1, 200).')

#         assert torch.all(torch.diag(_hidden_layer.weight) == 0), AssertionError('Hidden layer weight matrix should have zeros on the diagonal.')
#     except Exception as e:
#         logger.error('Error in constructing CTRNN with parameters.')
#         raise e


# def check_forward_pass(logger):
#     # check forward pass
#     logger.info('Testing forward pass...')
#     try:
#         in_ = torch.randn(1, 1)
#         rnn = CTRNN(**MODEL_PARAMS)

#         # check forward pass value
#         a = MODEL_PARAMS['dt'] / MODEL_PARAMS['tau']
#         pre_hidden = in_ @ rnn.recurrent_layer.input_layer.weight.T + rnn.recurrent_layer.input_layer.bias
#         post_activation = rnn.recurrent_layer.activation(a*pre_hidden).detach().numpy()

#         _, hidden_states = rnn.forward(in_)
#         hidden_states = hidden_states.detach().numpy()

#         assert hidden_states.shape == (1, 200), AssertionError('Hidden states should be of shape (1, 200).')
#         assert np.allclose(post_activation, hidden_states, atol=1e-3), AssertionError('Hidden states should be equal to post activation.')
#     except Exception as e:
#         logger.error('Error in performing forward pass.')
#         raise e


# def check_struct(logger):
#     logger.info('Testing MultiArea construction...')
#     try:
#         area_connectivities = np.array([
#             [1, 0],
#             [0, 1]
#         ])
#         _struct_pm = STRUCT_PARAMS.copy()
#         _struct_pm['area_connectivities'] = area_connectivities
#         struct = MultiArea(**_struct_pm)

#         input_mask, hidden_mask, readout_mask = struct.masks()
#         assert input_mask.shape == (200, 1), AssertionError('Input mask should be of shape (200, 1).')
#         assert hidden_mask.shape == (200, 200), AssertionError('Hidden mask should be of shape (200, 200).')
#         assert readout_mask.shape == (1, 200), AssertionError('Readout mask should be of shape (1, 200).')

#         # check input/readout masks
#         assert np.all(input_mask[:100, :] == 1), AssertionError('Input mask for area 1 should be all ones.')
#         assert np.all(input_mask[100:, :] == 0), AssertionError('Input mask for area 2 should be all zeros.')
#         assert np.all(readout_mask[:, :100] == 0), AssertionError('Readout mask for area 1 should be all zeros.')
#         assert np.all(readout_mask[:, 100:] == 1), AssertionError('Readout mask for area 2 should be all ones.')

#         # check hidden mask
#         assert np.all(hidden_mask[:100, :100] == 1), AssertionError('Hidden mask for area 1 should be all ones.')
#         assert np.all(hidden_mask[100:, 100:] == 1), AssertionError('Hidden mask for area 2 should be all ones.')
#     except Exception as e:
#         logger.error('Error in constructing MultiArea.')
#         raise e


# def check_fwd_bwd(logger):
#     # check multi-area forward connections
#     logger.info('Testing forward connections...')

#     # # no area-area connections
#     # hidden_states = connectivity_check_helper(np.array([[1, 0], [0, 1]]))
#     # assert torch.all(hidden_states[0,100:] == 0), AssertionError('Hidden states for area 2 should be all zeros.')

#     # area 1 -> area 2
#     # input to area 1
#     hidden_states = connectivity_check_helper(np.array(([[1, 0], [SPARSE, 1]])))
#     assert torch.any(hidden_states[1, 100:] != 0), AssertionError('Hidden states for area 2 should not be all zeros.')
#     # input to area 2
#     hidden_states = connectivity_check_helper(np.array(([[1, 0], [SPARSE, 1]])), in_a=[1], out_a=[0])
#     assert torch.all(hidden_states[1, :100] == 0), AssertionError('Hidden states for area 1 should be all zeros.')

#     # area 2 -> area 1
#     # input to area 1
#     hidden_states = connectivity_check_helper(np.array([[1, SPARSE], [0, 1]]))
#     assert torch.all(hidden_states[1, 100:] == 0), AssertionError('Hidden states for area 2 should be all zeros.')
#     # input to area 2
#     hidden_states = connectivity_check_helper(np.array([[1, SPARSE], [0, 1]]), in_a=[1], out_a=[0])
#     assert torch.any(hidden_states[1, :100] != 0), AssertionError('Hidden states for area 1 should not be all zeros.')


# def connectivity_check_helper(conn, in_a=[0], out_a=[1]):
#     # construct structure parameters
#     _struct_pm = STRUCT_PARAMS.copy()
#     _struct_pm['area_connectivities'] = conn
#     _struct_pm['input_areas'] = in_a
#     _struct_pm['readout_areas'] = out_a
#     struct = MultiArea(**_struct_pm)
#     # construct model parameters
#     _model_pm = MODEL_PARAMS.copy()
#     _model_pm['new_synapses'] = False
#     _model_pm['layer_masks'] = struct.masks()
#     # intialize model
#     rnn = CTRNN(**_model_pm)
#     in_ = torch.randn(2, 1)
#     _, hidden_states = rnn.forward(in_)

#     return hidden_states
