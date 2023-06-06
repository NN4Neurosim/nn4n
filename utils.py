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
    plt.figure(figsize=(5, 5))

    plt.imshow(w, cmap='bwr', vmin=-r, vmax=r)
    plt.title(title)
    if colorbar: plt.colorbar()
    plt.tight_layout()
    plt.show()