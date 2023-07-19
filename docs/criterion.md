# Criterion
## RNNLoss
The loss function is modularized. Each criterion is designed in the format of $\lambda_L L(\cdot)$. By default, all $\lambda_{*}$ are set to 0 and won't be added to loss (nor the auto-grad tree). By changing the corresponding $\lambda_{*}$ to non-zero positive values, the corresponding loss function will be added to the total loss. The total loss is the sum of all loss functions with non-zero $\lambda_{*}$.

## Loss Functions
### MSE
#### Definition
Mean Squared Error (MSE) loss function. The MSE loss function is defined as:
$$
L_{MSE} = \frac{\lambda_{mse}}{N} \sum_{i=1}^{N} \left( \hat{y}_i - y_i \right)^2
$$
where $N$ is the number of samples, $\hat{y}_i$ is the predicted value, and $y_i$ is the ground truth value.<br>
#### Lambda key
`'lambda_mse'`<br>

### InputLayer Sparsity
#### Definition
InputLayer Sparsity loss function. The InputLayer Sparsity loss function is defined as:
$$
L_{in} = \frac{\lambda_{in}}{N_{in} N_{hid}} \sum_{i,j} w_{ij}^2 = \frac{1}{N_{in} N_{hid}} ||W_{in}||_F^2
$$
where $N_{in}$ is the number of input neurons, $N_{hid}$ is the size of the input, and $w_{ij}$ is the weight from the $i$-th input neuron to the $j$-th hidden neuron.<br>
#### Lambda key
`'lambda_in'`<br>

### HiddenLayer Sparsity
#### Definition
HiddenLayer Sparsity loss function. The HiddenLayer Sparsity loss function is defined as:
$$
L_{hid} = \frac{\lambda_{hid}}{N_{hid} N_{hid}} \sum_{i,j} w_{ij}^2 = \frac{1}{N_{hid} N_{hid}} ||W_{hid}||_F^2
$$
where $N_{hid}$ is the number of hidden neurons, $N_{hid}$ is the size of the input (of the HiddenLayer), and $w_{ij}$ is the weight from the $i$-th hidden neuron to the $j$-th output neuron.<br>
#### Lambda key
`'lambda_hid'`<br>

### ReadoutLayer Sparsity
#### Definition
ReadoutLayer Sparsity loss function. The ReadoutLayer Sparsity loss function is defined as:
$$
L_{out} = \frac{\lambda_{out}}{N_{out} N_{hid}} \sum_{i,j} w_{ij}^2 = \frac{1}{N_{out} N_{hid}} ||W_{out}||_F^2
$$
where $N_{out}$ is the number of output neurons, $N_{hid}$ is the size of the input (of the ReadoutLayer), and $w_{ij}$ is the weight from the $i$-th output neuron to the $j$-th output neuron.<br>
#### Lambda key
`'lambda_out'`<br>

### Firing Rate Regularization
#### Definition
Firing Rate Regularization loss function. The Firing Rate Regularization loss function is defined as:
$$
L_{fr} = \frac{\lambda_{fr}}{N_{batch} T N_{hid}} \sum_{b,t,n=0}^{N_{batch}, T, N_{hid}} \left( \hat{y}_{btn} - y_{btn} \right)^2
$$
where $N_{batch}$ is the batch size, $T$ is the number of time steps, $N_{out}$ is the number of output neurons, $\hat{y}_{ijt}$ is the predicted firing rate of the $j$-th output neuron at time $t$ in the $i$-th batch, and $y_{ijt}$ is the ground truth firing rate of the $j$-th output neuron at time $t$ in the $i$-th batch.<br>
#### Lambda key
`'lambda_fr'`<br>

### Firing Rate Regularization (Neuron standard deviation)
#### Definition
Regularize the firing rate of a single neuron such that all neurons will fire at approximately the same rate. The Firing Rate Regularization loss function is defined as:
$$
L_{fr\_std} = \lambda_{fr\_std} \sqrt{\frac{1}{N_{batch} T} \sum_{b,t=0}^{N_{batch}, T} \left( \left( \hat{y}_{bt} - y_{bt} \right)^2 - \mu \right)^2}
$$
$$
\mu = \frac{1}{N_{batch} T} \sum_{b,t=0}^{N_{batch}, T} \left( \hat{y}_{bt} - y_{bt} \right)^2
$$
where $N_{batch}$ is the batch size, $T$ is the number of time steps, $\hat{y}_{bt}$ is the predicted firing rate of the neuron at time $t$ in the $i$-th batch, and $y_{bt}$ is the ground truth firing rate of the neuron at time $t$ in the $i$-th batch.<br>
#### Lambda key
`'lambda_fr_std'`<br>

### Firing Rate Regularization (Single Neuron coefficient of variation)
#### Definition
Regularize the firing rate of a single neuron such that all neurons will fire at approximately the same rate. The Firing Rate Regularization loss function is defined as:
$$
L_{fr\_cv} = \lambda_{fr\_cv} \frac{\sigma}{\mu}
$$
where $\sigma$ is the standard deviation of the firing rate of the neuron, and $\mu$ is the mean of the firing rate of the neuron.<br>