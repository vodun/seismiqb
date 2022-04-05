""" Utility functions for tests."""
import numpy as np

def generate_synthetic(shape=(500, 500, 300), i_scale=5, i_frequency=0.02, x_scale=5, x_frequency=0.05):
    """ Create synthetic data cube and horizon for tests."""
    i_shape, x_shape, depth = shape
    synthetic = np.empty(shape, dtype=np.float32)
    matrix = np.empty(shape[:2], dtype=np.int32)

    arange = np.arange(depth, dtype=np.float32)
    trace = np.sin(0.1 * arange)

    for i in range(i_shape):
        for j in range(x_shape):

            offset = i_scale*np.cos(i_frequency*i) + x_scale*np.sin(x_frequency*j)
            offset = int(offset)

            synthetic[i, j, :] = np.roll(trace, offset)
            matrix[i, j] = offset

    return synthetic, matrix + depth // 2
