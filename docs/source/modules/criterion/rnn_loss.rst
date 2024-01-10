RNNLoss()
=========

The loss function is modularized. Each criterion is designed in the format of 
:math:`\lambda_L L(\cdot)`. By default, all :math:`\lambda_{L}` are set to 0 
and won't be added to loss (nor the auto-grad tree). By changing the corresponding 
:math:`\lambda_{L}` to non-zero positive values, the corresponding loss function 
will be added to the total loss. The total loss is the sum of all loss functions 
with non-zero :math:`\lambda_{L}`.


Contents
--------

.. contents::
   :local:
   :depth: 3


Class Methods
^^^^^^^^^^^^^

.. automodule:: nn4n.criterion
   :members:

Keyword Arguments
^^^^^^^^^^^^^^^^^

`'lambda_mse'`
   description: 
      Coefficient for the Mean Squared Error (MSE) loss function. The MSE loss function is defined as:

      .. math::

         L_{MSE} = \frac{\lambda_{mse}}{N_{batch} T N_{out}} \sum_{b,t,n=0}^{N_{batch}, T, N_{out}} \left( \hat{y}_{btn} - y_{btn} \right)^2

      where :math:`N` is the number of samples, :math:`\hat{y}_{btn}` is the predicted value, and :math:`y_{btn}` is the ground truth value.

   default: 1

   type: ``float``

`'lambda_in'`
   description: 
      Coefficient for the InputLayer sparsity loss function. This argument controls the sparsity of the InputLayer
      by constraining InputLayer Frobenius norm. The InputLayer sparsity loss function is defined as:

      .. math::

         L_{in} = \frac{\lambda_{in}}{N_{in} N_{hid}} \sum_{i,j} w_{ij}^2 = \frac{\lambda_{in}}{N_{in} N_{hid}} ||W_{in}||_F^2

      where :math:`N_{in}` is the number of input neurons, :math:`N_{hid}` is the size of the input, and :math:`w_{ij}` is the weight from the :math:`i`-th input neuron to the :math:`j`-th hidden neuron.

   default: 0

   type: ``float``

`'lambda_hid'`
   description: 
      Coefficient for the HiddenLayer sparsity loss function. This argument controls the sparsity of the HiddenLayer
      by constraining HiddenLayer Frobenius norm. The HiddenLayer sparsity loss function is defined as:

      .. math::

         L_{hid} = \frac{\lambda_{hid}}{N_{hid} N_{out}} \sum_{i,j} w_{ij}^2 = \frac{\lambda_{hid}}{N_{hid} N_{out}} ||W_{hid}||_F^2

      where :math:`N_{hid}` is the number of hidden neurons, :math:`N_{out}` is the size of the output, and :math:`w_{ij}` is the weight from the :math:`i`-th hidden neuron to the :math:`j`-th output neuron.
   
   default: 0

   type: ``float``

`lambda_out`
   description: 
      Coefficient for the ReadoutLayer sparsity loss function. This argument controls the sparsity of the ReadoutLayer
      by constraining ReadoutLayer Frobenius norm. The ReadoutLayer sparsity loss function is defined as:

      .. math::

         L_{out} = \frac{\lambda_{out}}{N_{out}} \sum_{i} w_{i}^2 = \frac{\lambda_{out}}{N_{out}} ||W_{out}||_F^2

      where :math:`N_{out}` is the number of output neurons, and :math:`w_{i}` is the weight from the :math:`i`-th output neuron to the output.
   
   default: 0

   type: ``float``

`'lambda_fr'`
   description: 
      Coefficient constraining the overall firing rate of HiddenLayer neurons. 
      The Firing rate loss function is defined as:

      .. math::

         L_{fr} = \frac{\lambda_{fr}}{N_{batch} T N_{hid}} \sum_{b,t,n=0}^{N_{batch}, T, N_{hid}} fr_{btn}^2

      where :math:`N_{batch}` is the batch size, :math:`T` is the number of time steps, :math:`N_{out}` is the number of output neurons, :math:`fr_{btn}` is the firing rate of neuron :math:`n` at time :math:`t` in the :math:`i`-th batch.

   default: 0

   type: ``float``

`'lambda_fr_sd'`
   description: 
      Coefficient constraining the standard deviation of the firing rate of HiddenLayer neurons. 
      This argument serves to distribute the firing rate more evenly across neurons.
      The Firing rate SD loss function is defined as:

      .. math::

         L_{fr\_sd} = \lambda_{fr\_sd} \sqrt{\frac{1}{N_{hid}} \sum_{n=0}^{N_{hid}} \left(\sum_{b,t=0}^{N_{batch}, T}\frac{fr_{bt}}{N_{batch} T} - \mu \right)^2}

      .. math::

         \mu = \frac{\lambda_{fr}}{N_{batch} T N_{hid}} \sum_{b,t,n=0}^{N_{batch}, T, N_{hid}} fr_{btn}

      where :math:`N_{batch}` is the batch size, :math:`T` is the number of time steps, :math:`\hat{y}_{bt}` is the predicted firing rate of the neuron at time :math:`t` in the :math:`i`-th batch, and :math:`f_{bt}` is the ground truth firing rate of the neuron at time :math:`t` in the :math:`i`-th batch.

   default: 0

   type: ``float``

`'lambda_fr_cv'`
   description: 
      Coefficient constraining the coefficient of variation of the firing rate of HiddenLayer neurons. 
      This argument also serves to distribute the firing rate more evenly across neurons, but with the 
      Coefficient of Variation (CV) instead of the standard deviation. This is less sensitive to the
      absolute firing rate of the neurons. The Firing rate CV loss function is defined as:

      .. math::

         L_{fr\_cv} = \lambda_{fr\_cv} \frac{\sigma}{\mu}

      where :math:`\sigma` is the standard deviation of the firing rate of the neuron, and :math:`\mu` is the network mean firing rate.

   default: 0

   type: ``float``
