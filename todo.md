# TODOs:
- [ ] The examples need to be updated. Especially on the main branch.
- [ ] Resolve the transpose issue in the model module and the mask module.
- [ ] Make the model use `batch_first` by default.
- [x] Refactor the RNNLoss part, let it take a dictionary instead of many separate `lambda_*` parameters. --> added the `CompositeLoss` instead.
- [x] Added batch_first parameter. Adjusted to batch_first by default to follow PyTorch standard.
- [ ] Varying alpha
- [ ] Need to adjust implementation for `apply_plasticity` as it won't support SSL framework.
- [ ] Change output to readout.
- [ ] Some quick methods to access firing rates of different values