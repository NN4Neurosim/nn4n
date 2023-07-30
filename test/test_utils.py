import torch
import numpy as np
from scipy.stats import shapiro, chisquare


def to_numpy(w):
    if isinstance(w, np.ndarray):
        return w
    elif isinstance(w, torch.Tensor):
        return w.detach().cpu().numpy()
    else:
        raise ValueError("w must be either a torch.Tensor or a numpy.ndarray")


def self_connections(w):
    """ Check if w has self connections """
    return np.any(np.diag(w) != 0)


def determine_distribution(w):
    """
    Determine the distribution of w
    NOTE: assumed dist cannot be both uniform and normal
    """
    # randomly sample 5000 weights if w is too large
    w = w.flatten()
    if len(w) > 5000:
        w = np.random.choice(w, 5000)

    _, n = shapiro(w)
    counts, _ = np.histogram(w, bins=50)
    _, u = chisquare(counts)

    if n > u:
        return "normal"
    else:
        return "uniform"


def check_activation(act, act_func):
    print(act)
    pos = torch.tensor([1, 2, 3, 4, 5])
    neg = torch.tensor([-1, -2, -3, -4, -5])
    if act == 'relu':
        zero = torch.tensor([0, 0, 0, 0, 0])
        return torch.all(act_func(pos) == pos) and torch.all(act_func(neg) == zero)
    elif act == 'tanh':
        pos_ans = torch.tensor([0.7616, 0.9640, 0.9951, 0.9993, 0.9999])
        neg_ans = torch.tensor([-0.7616, -0.9640, -0.9951, -0.9993, -0.9999])
        return torch.allclose(act_func(pos), pos_ans, atol=1e-4) and torch.allclose(act_func(neg), neg_ans, atol=1e-4)
    elif act == 'sigmoid':
        pos_ans = torch.tensor([0.7311, 0.8808, 0.9526, 0.9820, 0.9933])
        neg_ans = torch.tensor([0.2689, 0.1192, 0.0474, 0.0180, 0.0067])
        return torch.allclose(act_func(pos), pos_ans, atol=1e-4) and torch.allclose(act_func(neg), neg_ans, atol=1e-4)
    else:
        return False


def get_weight_info(w):
    """ Check the distribution of w and get some statistics """
    w = to_numpy(w)
    specs = {
        "mean": np.mean(w),
        "std": np.std(w),
        "min": np.min(w),
        "max": np.max(w),
        "dist": determine_distribution(w),
        "self_connections": self_connections(w),
    }
    if np.any(np.isnan(w)):
        specs["contains_nan"] = "nan"

    if np.any(np.isinf(w)):
        specs["contains_inf"] = "inf"

    return specs
