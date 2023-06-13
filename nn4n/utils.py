import numpy as np
import matplotlib.pyplot as plt

def print_dict(title, params):
    print(f"{title}: ")
    maxlen = max([len(s) for s in params.keys()])
    for k in params.keys():
        print(3*' ' + '| {}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', params[k]))
    print()

def plot_connectivity_matrix_dist(w, title, colorbar=True, ignore_zeros=False):
    r = np.max(np.abs(w))
    
    fig, axs = plt.subplots(2, 1, figsize=(6, 6 * w.shape[0] / w.shape[1] + 3), gridspec_kw={'height_ratios': [w.shape[0] / w.shape[1], 0.2]})

    axs[0].imshow(-w, cmap='bwr', vmin=-r, vmax=r)
    axs[0].set_title(title)
    if colorbar: fig.colorbar(axs[0].imshow(-w, cmap='bwr', vmin=-r, vmax=r), ax=axs[0])
    
    if ignore_zeros:
        axs[1].hist(w[w != 0].flatten(), bins=50)
    else:
        axs[1].hist(w.flatten(), bins=50)
    axs[1].set_title(f"{title} Distribution")
    # plt.tight_layout()
    plt.show()


def plot_connectivity_matrix(w, title, colorbar=True):
    r = np.max(np.abs(w))

    fig, ax = plt.subplots(figsize=(6, 6*w.shape[0]/w.shape[1]))

    ax.imshow(-w, cmap='bwr', vmin=-r, vmax=r)
    ax.set_title(title)
    if colorbar: fig.colorbar(ax.imshow(w, cmap='bwr', vmin=-r, vmax=r), ax=ax)
    # plt.tight_layout()
    plt.show()