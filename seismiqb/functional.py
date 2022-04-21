""" Contains various functions for mathematical/geological transforms. """
from math import ceil
from warnings import warn

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False
import bottleneck
import numexpr
from numba import njit, prange
from scipy.ndimage import find_objects
from skimage.measure import label

from .utils import Accumulator



# Device management
def to_device(array, device='cpu'):
    """ Transfer array to chosen GPU, if possible.
    If `cupy` is not installed, does nothing.

    Parameters
    ----------
    device : str or int
        Device specificator. Can be either string (`cpu`, `gpu:4`) or integer (`4`).
    """
    if isinstance(device, str) and ':' in device:
        device = int(device.split(':')[1])
    if device in ['cuda', 'gpu']:
        device = 0

    if isinstance(device, int):
        if CUPY_AVAILABLE:
            with cp.cuda.Device(device):
                array = cp.asarray(array)
        else:
            warn('Performance Warning: computing metrics on CPU as `cupy` is not available', RuntimeWarning)
    return array

def from_device(array):
    """ Move the data from GPU, if needed.
    If `cupy` is not installed or supplied array already resides on CPU, does nothing.
    """
    if CUPY_AVAILABLE and hasattr(array, 'device'):
        array = cp.asnumpy(array)
    return array



# Functions to compute various distances between two atleast 2d arrays
def correlation(array1, array2, std1, std2, **kwargs):
    """ Compute correlation. """
    _ = kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    if xp is np:
        covariation = bottleneck.nanmean(numexpr.evaluate('array1 * array2'), axis=-1)
        result = numexpr.evaluate('covariation / (std1 * std2)')
    else:
        covariation = (array1 * array2).mean(axis=-1)
        result = covariation / (std1 * std2)
    return result


def crosscorrelation(array1, array2, std1, std2, **kwargs):
    """ Compute crosscorrelation. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    window = array1.shape[-1]
    pad_width = [(0, 0)] * (array2.ndim - 1) + [(window//2, window - window//2)]
    padded = xp.pad(array2, pad_width=tuple(pad_width))

    accumulator = Accumulator('argmax')
    for i in range(window):
        corrs = (array1 * padded[..., i:i+window]).sum(axis=-1)
        accumulator.update(corrs)
    return accumulator.get(final=True).astype(float) - window//2


def btch(array1, array2, std1, std2, **kwargs):
    """ Compute Bhattacharyya distance. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    return xp.sqrt(array1 * array2).sum(axis=-1)


def kl(array1, array2, std1, std2, **kwargs):
    """ Compute Kullback-Leibler divergence. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    return 1 - (array2 * xp.log2(array2 / array1)).sum(axis=-1)


def js(array1, array2, std1, std2, **kwargs):
    """ Compute Janson-Shannon divergence. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np

    average = (array1 + array2) / 2
    log_average = xp.log2(average)
    div1 = (array1 * (xp.log2(array1) - log_average)).sum(axis=-1)
    div2 = (array2 * (xp.log2(array2) - log_average)).sum(axis=-1)
    return 1 - (div1 + div2) / 2


def hellinger(array1, array2, std1, std2, **kwargs):
    """ Compute Hellinger distance. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np

    div = xp.sqrt(((xp.sqrt(array1) - xp.sqrt(array2)) ** 2).sum(axis=-1)) / xp.sqrt(2)
    return 1 - div


def tv(array1, array2, std1, std2, **kwargs):
    """ Compute total variation distance. """
    _ = std1, std2, kwargs
    xp = cp.get_array_module(array1) if CUPY_AVAILABLE else np
    return 1 - xp.abs(array2 - array1).sum(axis=-1) / 2


# Helper functions
def hilbert(array, axis=-1):
    """ Compute the analytic signal, using the Hilbert transform. """
    xp = cp.get_array_module(array) if CUPY_AVAILABLE else np
    N = array.shape[axis]
    fft = xp.fft.fft(array, n=N, axis=axis)

    h = xp.zeros(N)
    if N % 2 == 0:
        h[0] = h[N // 2] = 1
        h[1:N // 2] = 2
    else:
        h[0] = 1
        h[1:(N + 1) // 2] = 2

    if array.ndim > 1:
        ind = [xp.newaxis] * array.ndim
        ind[axis] = slice(None)
        h = h[tuple(ind)]

    result = xp.fft.ifft(fft * h, axis=axis)
    return result

def instantaneous_phase(array, continuous=False, axis=-1):
    """ Compute instantaneous phase. """
    xp = cp.get_array_module(array) if CUPY_AVAILABLE else np
    array = hilbert(array, axis=axis)
    phase = xp.angle(array) % (2 * xp.pi) - xp.pi
    if continuous:
        phase = xp.abs(phase)
    return phase

def make_gaussian_kernel(kernel_size=3, sigma=1.):
    """ Create Gaussian kernel with given parameters: kernel size and std. """
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    x_points, y_points = np.meshgrid(ax, ax)
    kernel = np.exp(-0.5 * (np.square(x_points) + np.square(y_points)) / sigma**2)
    gaussian_kernel = (kernel / np.sum(kernel).astype(np.float32))
    return gaussian_kernel

def digitize(matrix, quantiles):
    """ Convert continuous metric into binarized version with thresholds defined by `quantiles`. """
    bins = np.nanquantile(matrix, np.sort(quantiles)[::-1])

    if len(bins) > 1:
        digitized = np.digitize(matrix, [*bins, np.nan]).astype(float)
        digitized[digitized > 0] -= 1
    else:
        digitized = np.zeros_like(matrix, dtype=np.float64)
        digitized[matrix <= bins[0]] = 1.0

    digitized[np.isnan(matrix)] = np.nan
    return digitized

def extend_grid(grid, idx_1, idx_2, max_idx_2, transposed, extension, max_frequency, cut_lines_by_grid):
    """ Extend lines on grid depend on extension value.

    Parameters:
    ----------
    grid : ndarray
        A 2D array with grid matrix.
    idx_1, idx_2 : ndarray
        Arrays of quality grid points indices by axes.
    max_idx_2 : int
        Max possible coordinate of grid point for second array.
    transposed : bool
        Whether the idx_1, idx_2 are transposed in relation to the coordinate axes.
    extension : int
        Amount of traces to extend near the trace on quality map.
    max_frequency : int
        Grid frequency for the simplest level of hardness in `quality_map`.
    cut_lines_by_grid : bool
        Whether to cut lines by sparse grid.
    """
    # Get grid borders if needed
    if cut_lines_by_grid:
        down_grid = (idx_2 // max_frequency) * max_frequency
        up_grid = np.clip(down_grid + max_frequency, 0, max_idx_2)

    for shift in range(-extension, extension + 1):
        indices_1 = idx_1
        indices_2 = idx_2 + shift

        # Filter point out of the field
        valid_points_indices = np.argwhere((indices_2 > 0) & (indices_2 < max_idx_2))

        if cut_lines_by_grid:
            # Filter points out of the grid unit
            valid_points_indices_ = np.argwhere((down_grid < indices_2) & (indices_2 < up_grid))
            valid_points_indices = np.intersect1d(valid_points_indices, valid_points_indices_)

        indices_2 = indices_2[valid_points_indices]
        indices_1 = indices_1[valid_points_indices]

        if not transposed:
            grid[indices_1, indices_2] = 1
        else:
            grid[indices_2, indices_1] = 1

    return grid

def gridify(matrix, frequencies, iline=True, xline=True, extension='cell', filter_outliers=0):
    """ Convert digitized map into grid with various frequencies corresponding to different bins.

    Parameters
    ----------
    frequencies : sequence of numbers
        Grid frequencies for individual levels of hardness in `quality_map`.
    iline, xline : bool
        Whether to make lines in grid to account for `ilines`/`xlines`.
    extension : 'full', 'cell', False or int
        Number of traces to grid lines extension.
        If 'full', then extends quality grid base points to field borders.
        If 'cell', then extends quality grid base points to sparse grid cells borders.
        If False, then make no extension.
        If int, then extends quality grid base points to +-extension//2 neighboring points.
    filter_outliers : int
        A degree of quality map thinning.
        `filter_outliers` more than zero cuts areas that contain too small connectivity regions.
        Notice that the method cut the squared area with these regions. It is made for more thinning.
    """
    # Preprocess a matrix: drop small complex regions from a quality map to make grid thinned
    if filter_outliers > 0:
        # Get connectivity objects
        labeled = label(matrix > 0)
        objects = find_objects(labeled)

        # Find and vanish regions that contains too small connectivity objects
        for object_slice in objects:
            obj = matrix[object_slice]
            obj_points = obj.sum()

            if obj_points < filter_outliers:
                matrix[object_slice] = 0

    values = np.unique(matrix[~np.isnan(matrix)])

    if len(values) != len(frequencies):
        min_freq = min(frequencies)
        max_freq = max(frequencies)
        multiplier = np.power(max_freq/min_freq, 1/(len(values) - 1))
        frequencies = [np.rint(max_freq / (multiplier ** i))
                       for i, _ in enumerate(values)]
    else:
        frequencies = np.sort(frequencies)[::-1]

    # Parse extension value
    if isinstance(extension, int):
        extension_value = extension // 2
    elif extension == 'cell':
        extension_value = frequencies[0]
    elif extension is False:
        extension_value = 0

    cut_lines_by_grid = extension == 'cell'

    grid = np.zeros_like(matrix)

    for value, freq in zip(values, frequencies):
        idx_1, idx_2 = np.nonzero(matrix == value)

        if iline:
            mask = (idx_1 % freq == 0)
            if (extension == 'full') or (freq == frequencies[0]):
                grid[idx_1[mask], :] = 1
            else:
                grid = extend_grid(grid=grid, idx_1=idx_1[mask], idx_2=idx_2[mask], max_idx_2=matrix.shape[1]-1,
                                   transposed=False, extension=extension_value, max_frequency=frequencies[0],
                                   cut_lines_by_grid=cut_lines_by_grid)

        if xline:
            mask = (idx_2 % freq == 0)
            if (extension == 'full') or (freq == frequencies[0]):
                grid[:, idx_2[mask]] = 1
            else:
                grid = extend_grid(grid=grid, idx_1=idx_2[mask], idx_2=idx_1[mask], max_idx_2=matrix.shape[0]-1,
                                   transposed=True, extension=extension_value, max_frequency=frequencies[0],
                                   cut_lines_by_grid=cut_lines_by_grid)

    grid[np.isnan(matrix)] = np.nan
    return grid


@njit(parallel=True)
def perturb(data, perturbations, window):
    """ Take a subset of size `window` from each trace, with the center being shifted by `perturbations`. """
    #pylint: disable=not-an-iterable
    i_range, x_range = data.shape[:2]
    output = np.full((i_range, x_range, window), 0.0)
    central = ceil(data.shape[-1] / 2)
    low = window // 2
    high = max(window - low, 0)

    for il in prange(i_range):
        for xl in range(x_range):
            start = central + perturbations[il, xl]
            output[il, xl, :] = data[il, xl, start-low:start+high]
    return output


@njit
def histo_reduce(data, bins):
    """ Convert each entry in data to histograms according to `bins`. """
    #pylint: disable=not-an-iterable
    i_range, x_range = data.shape[:2]

    hist_matrix = np.full((i_range, x_range, len(bins) - 1), np.nan)
    for il in prange(i_range):
        for xl in prange(x_range):
            hist_matrix[il, xl] = np.histogram(data[il, xl], bins=bins)[0]
    return hist_matrix
