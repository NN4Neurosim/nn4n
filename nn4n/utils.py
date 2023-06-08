import numpy as np
import matplotlib.pyplot as plt

def print_dict(title, params):
    print(f"{title}: ")
    maxlen = max([len(s) for s in params.keys()])
    for k in params.keys():
        print(3*' ' + '| {}:{}{}'.format(k, (maxlen - len(k) + 1)*' ', params[k]))
    print()

def plot_connectivity_matrix(w, title, colorbar=True):
    r = np.max(np.abs(w))
    
    # plot connectivity matrix and weight distribution in separate subplots vertically
    # compute the height of the figure based on the aspect ratio of the weight matrix
    # make sure the connectivity matrix has the same width as the weight distribution
    fig, axs = plt.subplots(2, 1, figsize=(6, 6 * w.shape[0] / w.shape[1] + 3), gridspec_kw={'height_ratios': [w.shape[0] / w.shape[1], 0.2]})

    axs[0].imshow(w, cmap='bwr', vmin=-r, vmax=r)
    axs[0].set_title(title)
    if colorbar: fig.colorbar(axs[0].imshow(w, cmap='bwr', vmin=-r, vmax=r), ax=axs[0])
    axs[1].hist(w.flatten(), bins=50)
    axs[1].set_title(f"{title} Distribution")
    plt.tight_layout()
    plt.show()
