import torch
import numpy as np
import matplotlib.pyplot as plt


def print_dict(title, params):
    print(f"{title}: ")
    maxlen = max([len(s) for s in params.keys()])
    for k in params.keys():
        print(3*' ' + '| {}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', params[k]))
    print()


def get_activation(act):
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
    r = np.max(np.abs(w))

    img_width, hist_height = 6, 2
    hw_ratio = w.shape[0] / w.shape[1]
    plt_height = img_width * hw_ratio + hist_height

    fig, ax = plt.subplots(figsize=(img_width, plt_height))
    ax.imshow(-w, cmap='bwr', vmin=-r, vmax=r)
    ax.set_title(f'{title}')
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
        ax.hist(w[np.abs(w) < np.mean(np.abs(w))*0.1].flatten(), bins=50)
    else:
        ax.hist(w.flatten(), bins=50)
    plt.tight_layout()
    plt.show()


def plot_connectivity_matrix(w, title, colorbar=True):
    r = np.max(np.abs(w))

    fig, ax = plt.subplots(figsize=(6, 6*w.shape[0]/w.shape[1]))

    ax.imshow(-w, cmap='bwr', vmin=-r, vmax=r)
    ax.set_title(title)
    if colorbar:
        fig.colorbar(ax.imshow(w, cmap='bwr', vmin=-r, vmax=r), ax=ax)
    # plt.tight_layout()
    plt.show()


def plot_eigenvalues(w, title):
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
