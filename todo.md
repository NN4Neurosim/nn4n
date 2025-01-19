## Resolved Items
- [x] Resolve the transpose issue in the model module and the mask module. --> So both masks and weights are now consistent, will transpose them per call.
- [x] Make the model use `batch_first` by default. All `batch_first` parameters are removed, let user set it in their own usage.
- [x] Refactor the RNNLoss part, let it take a dictionary instead of many separate `lambda_*` parameters. --> added the `CompositeLoss` instead.
- [x] Adjusted the network to batch_first by default to follow PyTorch standard.
- [x] Varying `alpha`. Alpha is now learnable.
- [x] Make `alpha` can be defined with a vector.
- [x] ~~eed to adjust implementation for `apply_plasticity` as it won't support SSL framework.~~ --> Removed plasticity for now, see below.
- [x] ~~Reconsider the `apply_plasticity`, Adam will not work with it.~~ --> Removed plasticity for now, see below.
- [x] Add `print_specs`, already have that as `print_layer`.
- [x] Better to construct network layer by layer.
- [x] Remove `relaxed_states` in the RNN.
- [x] Make the layers can be initialized with either a dictionary or directly with init.
- [x] For some reason, all the masks are set with a transpose. For instance: `self.ei_mask = layer_struct['ei_mask'].T`. Not a big issue, but good to check or just simplify.
- [x] ~~Remove the `plasticity` scales, only allow 1 or 0 as for reason suggested in one of the previous todo item.~~ --> Removed plasticity for now, see below.
- [x] Make the `weights` and `biases` to be singular.
- [x] Clean up the masks code and make them register as a buffer for simplicity.
- [x] Put back init_state.
- [x] Constraints (sparsity, positivity, etc.) are now enforced after each forward pass, registered as a buffer and also registered `enforce_constraints()` with `register_forward_pre_hook`.
- [x] Think about how to store the entire model. Currently thinking about making a checkpoint function for each layer.
- [x] The `hidden_layer` is the same as the linear layer, consider to remove.

---

## Unresolved Items
- [ ] The examples need to be updated. Especially on the main branch.
- [ ] Change output to readout. (Forgot what I meant by this).
- [ ] Need some property functions to get the model parameters in a simpler way.
- [ ] Need some kind of function or managing class to perform the forward pass. Like `model({"area_1": input_1, "area_2": input_2})`.
- [ ] The `get_area` related functions in the mask module are a bit un-intuitive. Might be good to rename them.
- [ ] Make `area_manager` an instance of the model?
- [ ] Change the `weights` and `biases` in layers to `weight_dist` and `bias_dist`. --> Prehaps pass initializers instead?
- [ ] Don't really remember what does `LinearLayer.auto_rescale()` do. Need to check.
- [ ] See if the model.print_layer() can directly be defined into `__repr__`.
- [ ] Removed plasticity for now, need to figure out a better way to do it as the optimizer won't work with it.
- [ ] Though the mask and weights shapes are consistent, it might be good to check them back by plotting.
