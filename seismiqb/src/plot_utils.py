""" Utility functions for plotting. """
#pylint: disable=expression-not-assigned
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from ..batchflow import Pipeline



def plot_loss(graph_lists, labels=None, ylabel='Loss', figsize=(8, 5), title=None, savefig=False, show_plot=True):
    """ Plot losses.

    Parameters
    ----------
    graph_lists : sequence of arrays
        Arrays to plot.
    labels : sequence of str
        Labels for different graphs.
    ylabel : str
        y-axis label.
    figsize : tuple of int
        Size of the resulting figure.
    title : str
        Title of the resulting figure.
    savefig : bool or str
        If str, then path for image saving.
        If False, then image is not saved.
    show_plot: bool
        Whether to show image in output stream.
    """
    if not isinstance(graph_lists[0], (tuple, list)):
        graph_lists = [graph_lists]

    labels = labels or 'loss'
    labels = labels if isinstance(labels, (tuple, list)) else [labels]

    plt.figure(figsize=figsize)
    for arr, label in zip(graph_lists, labels):
        plt.plot(arr, label=label)
    plt.xlabel('Iterations', fontdict={'fontsize': 15})
    plt.ylabel(ylabel, fontdict={'fontsize': 15})
    plt.grid(True)
    if title:
        plt.title(title, fontdict={'fontsize': 15})
    plt.legend()

    if savefig:
        plt.savefig(savefig, bbox_inches='tight', pad_inches=0)
    plt.show() if show_plot else plt.close()
