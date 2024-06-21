import torch
import numpy as np
import matplotlib.pyplot as plt


def print_dict(title, params):
    """
    Print a dictionary in key-value pairs with a nice format

    Inputs:
        - title: title of the dictionary
        - params: dictionary to be printed
    """
    print(f"{title}: ")
    maxlen = max([len(s) for s in params.keys()])
    for k in params.keys():
        print(3*' ' + '| {}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', params[k]))
    print()


def get_activation(act):
    """
    Return the activation function given the name

    Inputs:
        - act: name of the activation function (supported values: "relu", "tanh", "sigmoid", "retanh")
    Returns:
        - activation function
    """
    if act == "relu":
        return torch.relu
    elif act == "tanh":
        return torch.tanh
    elif act == "sigmoid":
        return torch.sigmoid
    elif act == "retanh":
        return lambda x: torch.maximum(torch.tanh(x), torch.tensor(0))
    else:
        raise NotImplementedError


def plot_connectivity_matrix_dist(w, title, colorbar=True, ignore_zeros=False):
    """
    Plot the distribution of a connectivity matrix

    Inputs:
        - w: connectivity matrix, must be a numpy array or a torch tensor
        - title: title of the plot
        - colorbar: whether to show the colorbar (default: True)
        - ignore_zeros: whether to ignore zeros in the distribution (needed for sparse matrices) (default: False)
    """
    if type(w) == torch.Tensor:
        w = w.detach().numpy()
    
    r = np.max(np.abs(w))

    img_width, hist_height = 6, 2
    hw_ratio = w.shape[0] / w.shape[1]
    plt_height = img_width * hw_ratio + hist_height

    fig, ax = plt.subplots(figsize=(img_width, plt_height))
    ax.imshow(-w, cmap='bwr', vmin=-r, vmax=r)
    ax.set_title(f'{title}' if not ignore_zeros else f'{title} (nonzero)')
    if colorbar:
        fig.colorbar(ax.imshow(-w, cmap='bwr', vmin=-r, vmax=r), ax=ax)
    if w.shape[1] < 5:
        ax.set_xticks([])
    if w.shape[0] < 5:
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(img_width, hist_height))
    ax.set_title(f'{title} distribution')
    if ignore_zeros:
        mean_nonzero = np.mean(np.abs(w)[np.abs(w) != 0])
        ax.hist(w[np.abs(w) > mean_nonzero*0.001].flatten(), bins=100)
    else:
        ax.hist(w.flatten(), bins=100)
    plt.tight_layout()
    plt.show()


def plot_connectivity_matrix(w, title, colorbar=True):
    """
    Plot a connectivity matrix

    Inputs:
        - w: connectivity matrix, must be a numpy array or a torch tensor
        - title: title of the plot
        - colorbar: whether to show the colorbar (default: True)
    """
    if type(w) == torch.Tensor:
        w = w.detach().numpy()

    r = np.max(np.abs(w))

    fig, ax = plt.subplots(figsize=(6, 6*w.shape[0]/w.shape[1]))

    ax.imshow(-w, cmap='bwr', vmin=-r, vmax=r)
    ax.set_title(title)
    if colorbar:
        fig.colorbar(ax.imshow(w, cmap='bwr', vmin=-r, vmax=r), ax=ax)
    # plt.tight_layout()
    plt.show()


def plot_eigenvalues(w, title):
    """
    Plot the eigenvalues of a connectivity matrix in the complex plane

    Inputs:
        - w: connectivity matrix, must be a numpy array or a torch tensor
        - title: title of the plot
    """
    if type(w) == torch.Tensor:
        w = w.detach().numpy()

    eigvals = np.linalg.eigvals(w)
    plt.figure(figsize=(4, 4))
    plt.plot(eigvals.real, eigvals.imag, 'o')
    max_eig = np.round(np.max(np.abs(eigvals)), 1)
    plt.xticks(np.arange(-max_eig, max_eig, 0.5), rotation=45)
    plt.yticks(np.arange(-max_eig, max_eig, 0.5))
    plt.xlabel(r'$\lambda_{real}$')
    plt.ylabel(r'$\lambda_{imag}$')
    plt.title(title)
    plt.show()
