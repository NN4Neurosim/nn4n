.. NN4Neurosci documentation installation guide

Get Started
============

Setup Environment (Optional)
----------------------------
It is recommended to install NN4Neurosci in a virtual environment.
To set up a virtual environment, follow the instructions at `https://conda.io/projects/conda/en/latest/user-guide/install/index.html <https://conda.io/projects/conda/en/latest/user-guide/install/index.html>`_.

1.  Create a virtual environment using conda

        .. code-block:: bash

            conda create -n nn4n python=3.8

2.  Activate the conda environment:
    
        .. code-block:: bash
    
            conda activate nn4n

Installation
------------

1. Using pip

    .. code-block:: bash

        pip install nn4n

2. From source

    Clone the repository from github

    .. code-block:: bash

        git clone --depth 1 https://github.com/zhaozewang/NN4Neurosci

    Install the package

    .. code-block:: bash

        cd NN4Neurosci
        pip install -e .

Tutorial: build a simple network with NN4Neurosci
-------------------------------------------------

Build a simple network
^^^^^^^^^^^^^^^^^^^^^^

    .. code-block:: python

        from nn4n.model import CTRNN
        import numpy as np

        # define the network with default parameters
        model = CTRNN()

        # print the network structure
        model.print_layers()

        # plot the network structure
        model.plot_layers()

Outputs:
^^^^^^^^
    .. code-block:: bash

        Linear Layer: 
        | input_dim:        1
        | output_dim:       100
        | weight_learnable: True
        | weight_min:       -0.9671670198440552
        | weight_max:       0.9995396137237549
        | bias_learnable:   False
        | bias_min:         0.0
        | bias_max:         0.0
        | sparsity:         1.0

        Recurrence: 
        | init_hidden_min: 0.0
        | init_hidden_max: 0.0
        | preact_noise:    0
        | postact_noise:   0
        | activation:      relu
        | alpha:           0.1

        Hidden Layer: 
        | input_dim:        100
        | output_dim:       100
        | weight_learnable: True
        | weight_min:       -0.09999121725559235
        | weight_max:       0.09996973723173141
        | bias_learnable:   False
        | bias_min:         0.0
        | bias_max:         0.0
        | sparsity:         1

        Linear Layer: 
        | input_dim:        100
        | output_dim:       1
        | weight_learnable: True
        | weight_min:       -0.09951122850179672
        | weight_max:       0.09540671855211258
        | bias_learnable:   False
        | bias_min:         0.0
        | bias_max:         0.0
        | sparsity:         1.0

    .. image:: _static/get_started/input_mat.png
    .. image:: _static/get_started/input_dist.png
    .. image:: _static/get_started/hidden_mat.png
    .. image:: _static/get_started/hidden_dist.png
    .. image:: _static/get_started/readout_mat.png
    .. image:: _static/get_started/readout_dist.png
