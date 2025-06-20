import matplotlib.pyplot as plt
import numpy as np


def plot_classification(X, y):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=160)
    if np.ndim(y) > 1:
        y = y[:, 0] + 2 * y[:, 1] + 4 * y[:, 2]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, alpha=0.1)
    legend1 = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()
