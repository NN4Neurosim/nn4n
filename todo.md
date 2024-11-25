# TODOs:
- [ ] The examples need to be updated. Especially on the main branch.
- [ ] Resolve the transpose issue in the model module and the mask module.
- [x] Make the model use `batch_first` by default. All `batch_first` parameters are removed, let user set it in their own usage.
- [x] Refactor the RNNLoss part, let it take a dictionary instead of many separate `lambda_*` parameters. --> added the `CompositeLoss` instead.
- [x] Adjusted the network to batch_first by default to follow PyTorch standard.
- [x] Varying `alpha`. Alpha is now learnable
- [x] Make `alpha` can be defined with a vector.
- [ ] Need to adjust implementation for `apply_plasticity` as it won't support SSL framework.
- [ ] Reconsider the `apply_plasticity`, Adam will not work with it.
- [ ] Change output to readout. (Forgot what I meant by this).
- [ ] Need some property functions to get the model parameters in a simpler way.
- [ ] Need some kind of function or managing class to perform the forward pass. Like `model({"area_1": input_1, "area_2": input_2})`.
- [ ] Remove the deprecation warnings.
- [ ] The `get_area` related of functions in the mask module are a bit un-intuitive. Might be good to rename them.
- [x] ~~Consider restructure.~~
- [x] Add `print_specs`, already have that as `print_layer`.
- [x] Better to construct network layer by layer.
- [ ] Make `area_manager` an instance of the model?
- [x] Remove `relaxed_states` in the RNN
- [x] Remove the init_state.
- [x] Make the layers can be initialized with either a dictionary or directly with init.
- [ ] For some reason, all the masks are set with a transpose. For instance: `self.ei_mask = layer_struct['ei_mask'].T`. Not a big issue, but good to check or just simplify. (duplicated item)
- [ ] Remove the `plasticity` scales, only allow 1 or 0 as for reason suggested in one of the previous todo item.
- [x] Make the `weights` and `biases` to be singular.
- [ ] Change the `weights` and `biases` in layers to `weight_dist` and `bias_dist`.
- [ ] Don't really remember what does `LinearLayer.auto_rescale()` do. Need to check.
- [ ] The `hidden_layer` is the same as the linear layer, consider to remove.
- [ ] Make the `dt` and `tau` to be a single property. Already did this but the `ctrnn` module need to automatically parse `dt` and `tau` to `alpha` for compatibility.
- [ ] Put back init_state.


## RNN Composer Module
- [ ] Think about how store the entire model. Currently thinking about making a checkpoint function for each layer.
