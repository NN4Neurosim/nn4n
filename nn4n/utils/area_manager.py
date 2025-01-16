import numpy as np


def check_init(func):
    def wrapper(self, *args, **kwargs):
        if self._area_indices is None:
            raise ValueError("Area indices are not initialized")
        return func(self, *args, **kwargs)

    return wrapper


class AreaManager:
    def __init__(self, area_indices=None):
        """
        Initialize the AreaManager

        Inputs:
            - area_indices: a list of indices (array) denoting a neuron's area assignment
        """
        if area_indices is not None:
            self.set_area_indices(area_indices)
        else:
            self._area_indices = None  # Ensure it's None initially

    def set_area_indices(self, area_indices):
        """
        Set the area indices

        Inputs:
            - area_indices: a list of indices (array) denoting a neuron's area assignment
        """
        self._n_areas = len(area_indices)
        self._area_indices = area_indices

    @property
    def n_areas(self):
        return self._n_areas

    @property
    def ai(self):
        return self._area_indices

    @check_init
    def split_states(self, states):
        """
        Parse the states of a complete RNN into a list of states of different areas

        Inputs:
            - states: network states of shape (batch_size, seq_len, hidden_size)
        Returns:
            - list of states of different areas
        """
        area_states = [states[:, :, idx] for idx in self._area_indices]
        return area_states

    @check_init
    def get_area_states(self, states, area_idx):
        """
        Get the states of a specific area

        Inputs:
            - states: network states of shape (batch_size, seq_len, hidden_size)
            - area_idx: index of the area
        Returns:
            - states of the specific area
        """
        return states[:, :, self._area_indices[area_idx]]

    @check_init
    def random_indices(self, area_idx, n, replace=False):
        """
        Randomly pick n indices from an area

        Inputs:
            - n: number of indices to pick
            - area_idx: index of the area
            - replace: whether to sample with replacement (default: False)
        Returns:
            - indices of the neurons
        """
        n_neurons = len(self._area_indices[area_idx])
        if not replace and n > n_neurons:
            return self._area_indices[area_idx]
        return np.random.choice(self._area_indices[area_idx], n, replace=replace)
