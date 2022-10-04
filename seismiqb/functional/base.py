""" Useful functions. """
import numpy as np


def make_gaussian_kernel(kernel_size=3, sigma=1.):
    """ Create Gaussian kernel with given parameters: kernel size and std. """
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    x_points, y_points = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (x_points**2 + y_points**2) / sigma**2)
    gaussian_kernel = (kernel / np.sum(kernel).astype(np.float32))
    return gaussian_kernel
