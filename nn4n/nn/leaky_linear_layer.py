import torch


class LeakyLinearLayer(torch.nn.Module):
    def __init__(
        self,
        linear_layer: torch.nn.Module,
        activation: torch.nn.Module,
        alpha: float = 0.1,
        learn_alpha: bool = False,
        preact_noise: float = 0,
        postact_noise: float = 0,
    ):
        """
        Leaky linear layer, which usually is used in the hidden layer of the RNN

        Parameters:
            - linear_layer: linear layer. I.e., W_rc in the RNN
            - activation: activation function
            - alpha: alpha value for the hidden layer
            - learn_alpha: whether to learn alpha
            - preact_noise: noise added to pre-activation
            - postact_noise: noise added to post-activation
        """
        super().__init__()
        assert linear_layer.input_dim == linear_layer.output_dim, \
            "Input and output dimensions of the linear layer must be the same"
        self.linear_layer = linear_layer
        self.activation = activation
        self.learn_alpha = learn_alpha
        self.preact_noise = preact_noise
        self.postact_noise = postact_noise
        self.alpha = (
            torch.nn.Parameter(
                torch.full((self.size,), alpha
            ), requires_grad=True if learn_alpha else False)
        )

    @property
    def input_dim(self) -> int:
        return self.linear_layer.input_dim
    
    @property
    def output_dim(self) -> int:
        return self.linear_layer.output_dim

    @property
    def size(self) -> int:
        return self.linear_layer.input_dim

    @property
    def hidden_size(self) -> int:
        return self.linear_layer.input_dim

    @staticmethod
    def _generate_noise(shape: torch.Size, noise: float, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device) * noise
    
    def clear_parameters(self):
        """
        Clear the parameters of the layer
        """
        self.linear_layer.clear_parameters()
        del self.alpha, self.learn_alpha, self.preact_noise, self.postact_noise

    def freeze(self):
        """
        Freeze the layer
        """
        self.linear_layer.freeze()
        self.alpha.requires_grad = False

    def unfreeze(self):
        """
        Unfreeze the layer
        """
        self.linear_layer.unfreeze()

    def set_noise(self, postact_noise: float = None, preact_noise: float = None):
        if postact_noise is not None:
            self.postact_noise = postact_noise
        if preact_noise is not None:
            self.preact_noise = preact_noise

    # FORWARD
    # =================================================================================
    def forward(self, fr: torch.Tensor, v: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        Forwardly update network

        Parameters:
            - fr: hidden state (post-activation), shape: (batch_size, hidden_size)
            - v: hidden state (pre-activation), shape: (batch_size, hidden_size)
            - u: input (raw/projected input), shape: (batch_size, input_size)

        Returns:
            - fr_n: hidden state (post-activation), shape: (batch_size, hidden_size)
            - v_n: hidden state (pre-activation), shape: (batch_size, hidden_size)
        """
        # Main update step
        v_n = (1 - self.alpha) * v + self.alpha * (self.linear_layer(fr) + u)

        # Preactivation noise
        if self.preact_noise > 0 and self.training:
            _preact_noise = self._generate_noise(v_n.size(), self.preact_noise, device=v_n.device)
            v_n = v_n + _preact_noise

        # Activation
        fr_n = self.activation(v_n)

        # Postactivation noise
        if self.postact_noise > 0 and self.training:
            _postact_noise = self._generate_noise(fr_n.size(), self.postact_noise, device=fr_n.device)
            fr_n = fr_n + _postact_noise
        
        return fr_n, v_n

    # HELPER FUNCTIONS
    # ======================================================================================
    def plot_layer(self, **kwargs):
        """
        Plot the layer
        """
        self.linear_layer.plot_layer(**kwargs)

    def _get_specs(self):
        """
        Get specs of the layer
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_size": self.hidden_size,
            "alpha": self.alpha,
            "learn_alpha": self.learn_alpha,
            "preact_noise": self.preact_noise,
            "postact_noise": self.postact_noise,
        }
