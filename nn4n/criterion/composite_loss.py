import torch
from .firing_rate_loss import *
from .connectivity_loss import *
from .prediction_loss import *


class CompositeLoss(torch.nn.Module):
    def __init__(self, loss_cfg):
        """
        Initializes the CompositeLoss module.

        Args:
            loss_cfg: A dictionary where the keys are unique identifiers for each loss (e.g., 'loss_fr_1') and the values are
                       dictionaries specifying the loss type, params, and lambda weight. Example:
                       {
                           'loss_fr_1': {'type': 'fr', 'params': {'metric': 'l2'}, 'lambda': 1.0},
                           'loss_mse': {'type': 'mse_loss', 'lambda': 1.0}
                       }
        """
        super().__init__()
        self.loss_components = {}

        # Mapping of loss types to their respective classes or instances
        loss_types = {
            "fr": FiringRateLoss,
            "fr_dist": FiringRateDistLoss,
            "rnn_conn": RNNConnectivityLoss,
            "state_pred": StatePredictionLoss,
            "entropy": EntropyLoss,
            "mse": torch.nn.MSELoss,
            "cross_entropy": CrossEntropyLoss,
            "hebbian": HebbianLoss,
        }

        # Iterate over the loss_cfg to instantiate and store losses
        for loss_name, loss_spec in loss_cfg.items():
            loss_type = loss_spec["type"]
            loss_params = loss_spec.get("params", {})
            loss_lambda = loss_spec.get("lambda", 1.0)

            # Instantiate the loss function
            if loss_type in loss_types:
                loss_class = loss_types[loss_type]
                if loss_type in ["mse"]:
                    # If torch built-in loss, don't pass the params
                    loss_instance = loss_class()
                else:
                    # Other losses might need params
                    loss_instance = loss_class(**loss_params)

                # Store the loss instance and its weight in a dictionary
                self.loss_components[loss_name] = (loss_instance, loss_lambda)
            else:
                raise ValueError(
                    f"Invalid loss type '{loss_type}'. Available types are: {list(loss_types.keys())}"
                )

    def forward(self, loss_input_dict):
        """
        Forward pass that computes the total weighted loss.

        Args:
            loss_input_dict: A dictionary where keys correspond to the initialized loss identifiers (e.g., 'loss_fr_1'),
                             and the values are dictionaries containing parameters to pass to the corresponding loss
                             function during the forward pass (e.g., {'states': <tensor>}).
        """
        total_loss = 0
        loss_dict = {}
        for loss_name, (loss_fn, loss_weight) in self.loss_components.items():
            # Retrieve the corresponding input for this loss from the input dictionary
            if loss_name in loss_input_dict:
                loss_inputs = loss_input_dict[loss_name]
                if isinstance(loss_fn, torch.nn.MSELoss) or isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss_value = loss_fn(loss_inputs["pred"], loss_inputs["target"])
                else:
                    loss_value = loss_fn(**loss_inputs)
                loss_dict[loss_name] = loss_weight * loss_value
                total_loss += loss_dict[loss_name]
            else:
                raise KeyError(f"Loss input for '{loss_name}' not provided in forward.")

        return total_loss, loss_dict
