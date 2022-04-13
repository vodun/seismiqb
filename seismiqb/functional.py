""" Contains various functions for mathematical/geological transforms. """
from math import isnan, ceil
from functools import wraps
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
    kernel = np.exp(-0.5 * (np.square(x_points) + np.square(y_points)) / np.square(sigma))
    gaussian_kernel = (kernel / np.sum(kernel).astype(np.float32))
    return gaussian_kernel

def process_fill_values(function):
    """ Decorator which applies a special treatment to the missing points marked with `fill_value`.

    Under the hood, this decorator converts missing values to `np.nan` and after applying the decorated function
    makes the reverse transformation.

    Note, that the decorator expects that the decorated function does not change the `matrix` inplace.
    """
    @wraps(function)
    def wrapper(matrix, fill_value=None, **kwargs):
        # Convert all fill values to nans
        if not isinstance(matrix, np.float32):
            matrix = matrix.astype(np.float32)

        if fill_value is not None:
            matrix[matrix == fill_value] = np.nan

        # Apply function
        result = function(matrix=matrix, **kwargs)

        # Convert nans back to the `fill_value`
        if fill_value is not None:
            result[np.isnan(result)] = fill_value
        return result
    return wrapper

def process_missings(function):
    """ Decorator which applies a special treatment to the missing points in a `matrix` marked with `np.nan`.

    Under the hood, this decorator preserve missing values from changing if needed.

    Note, that the decorator expects that the decorated function does not change the `matrix` inplace.
    Note, that if you want to operate with fill values as with nans, you need primarily to call
    the :func:`process_fill_values` decorator.
    """
    @wraps(function)
    def wrapper(matrix, preserve_missings=True, **kwargs):
        # Apply function
        result = function(matrix=matrix, preserve_missings=preserve_missings, **kwargs)

        # Remove all the unwanted values
        if preserve_missings:
            result[np.isnan(matrix)] = np.nan
        return result
    return wrapper

@process_fill_values
@process_missings
def convolve(matrix, kernel_size=3, kernel=None, iters=1,
             fill_value=None, preserve_missings=True, margin=np.inf, **_):
    """ Convolve the matrix with a given kernel.
    A special treatment is given to missing points (marked with either `fill_value` or `np.nan`),
    and to areas with high variance.

    Parameters
    ----------
    matrix : ndarray
        Array to convolve values in.
    kernel_size : int
        If the kernel is not provided, shape of the square kernel with ones.
    kernel : ndarray or None
        Kernel to convolve with.
    iters : int
        Number of convolve iterations to perform.
    fill_value : number
        Value which is interpreted as `np.nan` in computations.
    preserve_missings : bool
        If True, then all the missing values remain missing in the resulting array.
        If False, then missing values are filled with weighted average of nearby points.
    margin : number
        If the distance between anchor point and the point inside filter is bigger than the margin,
        then the point is ignored in convolutions.
        Can be used for separate smoothening on sides of discontinuity.
    """
    _ = fill_value # This value is passed only to the decorator

    if kernel is None:
        kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    kernel_size = kernel.shape[0]
    result = np.pad(matrix, kernel_size, constant_values=np.nan)

    # Apply `_convolve` multiple times. Note that there is no dtype conversion in between
    for _ in range(iters):
        result = _convolve(src=result, kernel=kernel, preserve_missings=preserve_missings, margin=margin)

    result = result[kernel_size:-kernel_size, kernel_size:-kernel_size]

    return result

def smooth_out(matrix, kernel_size=3, kernel=None, iters=1, fill_value=None,
               preserve_missings=True, margin=np.inf, sigma=2.0, **kwargs):
    """ Matrix smoothening via convolution with a gaussian kernel. """
    if kernel is None:
        kernel = make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    result = convolve(matrix=matrix, kernel=kernel, iters=iters, fill_value=fill_value,
                      preserve_missings=preserve_missings, margin=margin, **kwargs)
    return result

sigma_doc = "sigma : float\n\tStandard deviation for a gaussian kernel creation."
smooth_out.__doc__ += '\n' + '\n'.join(convolve.__doc__.split('\n')[1:]) + sigma_doc

@process_fill_values
def interpolate(matrix, kernel_size=3, kernel=None, iters=1, fill_value=None,
                min_neighbors=0, margin=None, sigma=2.0, **_):
    """ Make 2d interpolation in missing points, marked with either `fill_value` or `np.nan`.
    Interpolation is made as a weighted average of neighboring points, where weights are defined as
    a gaussian kernel (if kernel is None).

    Parameters
    ----------
    matrix : ndarray
        Array to make interpolation in.
    kernel_size : int
        If the kernel is not provided, shape of the square gaussian kernel.
    kernel : ndarray or None
        Kernel to apply to missing points.
    iters : int
        Number of interpolation iterations to perform.
    fill_value : number
        Value to interpolate besides `np.nan`.
    min_neighbors: int or float
        Minimal of non-missing neighboring points in a window to interpolate a central point.
        If int, then it is an amount of points.
        If float, then it is a points ratio.
    margin : number
        A maximum ptp between values in a squared window for which we apply interpolation.
    sigma : float
        Standard deviation for a gaussian kernel creation.
    """
    _ = fill_value # This value is passed only to the decorator

    if kernel is None:
        kernel = make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

    if isinstance(min_neighbors, float):
        min_neighbors = round(min_neighbors * kernel.size)

    kernel_size = kernel.shape[0]
    result = np.pad(matrix, kernel_size, constant_values=np.nan)

    # Apply `_interpolate` multiple times. Note that there is no dtype conversion in between
    for _ in range(iters):
        result = _interpolate(src=result, kernel=kernel, min_neighbors=min_neighbors, margin=margin)

    result = result[kernel_size:-kernel_size, kernel_size:-kernel_size]

    return result

@process_fill_values
@process_missings
def median_filter(matrix, kernel_size=3, iters=1, fill_value=None, preserve_missings=True, margin=np.inf, **_):
    """ 2d median filter with special care for nan values (marked with either `fill_value` or `np.nan`),
    and to areas with high variance.

    Parameters
    ----------
    matrix : ndarray
        Array to filter values in.
    kernel_size : int
        Shape of the square kernel in which to apply filter.
    iters : int
        Number of filter iterations to perform.
    fill_value : number
        Value to ignore in computations.
    preserve_missings : bool
        If True, then all the missing values remain missing in the resulting array.
        If False, then missing values are filled with weighted average of nearby points.
    margin : number
        If the distance between anchor point and the point inside filter is bigger than the margin,
        then the point is ignored in filtering.
        Can be used for separate smoothening on sides of discontinuity.
    """
    _ = fill_value # This value is passed only to the decorator

    result = np.pad(matrix, kernel_size, constant_values=np.nan)

    # Apply `_medfilt` multiple times. Note that there is no dtype conversion in between
    for _ in range(iters):
        result = _medfilt(src=result, kernel_size=kernel_size, preserve_missings=preserve_missings, margin=margin)

    result = result[kernel_size:-kernel_size, kernel_size:-kernel_size]

    return result

@njit(parallel=True)
def _convolve(src, kernel, preserve_missings, margin):
    """ Jit-accelerated function to apply 2d convolution with special care for nan values. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel.shape[0] // 2
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(k, i_range - k):
        for xline in range(k, x_range - k):
            central = src[iline, xline]

            if (preserve_missings is True) and isnan(central):
                continue # Do nothing with nans

            # Get values in the squared window and apply kernel to them
            element = src[iline-k:iline+k+1, xline-k:xline+k+1]

            s, sum_weights = np.float32(0), np.float32(0)
            for item, weight in zip(element.ravel(), raveled_kernel):
                if not isnan(item) and (abs(item - central) <= margin or isnan(central)):
                    s += item * weight
                    sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst

@njit(parallel=True)
def _interpolate(src, kernel, min_neighbors=1, margin=None):
    """ Jit-accelerated function to apply 2d interpolation to nan values. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel.shape[0] // 2
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(k, i_range - k):
        for xline in range(k, x_range - k):
            central = src[iline, xline]

            if not isnan(central):
                continue # We interpolate values only to nan points

            # Get neighbors and check whether we can interpolate them
            element = src[iline-k:iline+k+1, xline-k:xline+k+1].ravel()

            notnan_neighbors = kernel.size - np.isnan(element).sum()
            if notnan_neighbors < min_neighbors:
                continue

            # Compare ptp with margin
            if margin is not None:
                nanmax, nanmin = np.float32(element[0]), np.float32(element[0])

                for item in element:
                    if not isnan(item):
                        if isnan(nanmax):
                            nanmax = item
                            nanmin = item
                        else:
                            nanmax = max(item, nanmax)
                            nanmin = min(item, nanmin)

                if nanmax - nanmin > margin:
                    continue

            # Apply kernel to neighbors to get value for interpolated point
            s, sum_weights = np.float32(0), np.float32(0)
            for item, weight in zip(element, raveled_kernel):
                if not isnan(item):
                    s += item * weight
                    sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst

@njit(parallel=True)
def _medfilt(src, kernel_size, preserve_missings, margin):
    """ Jit-accelerated function to apply 2d median filter with special care for nan values. """
    # margin = 0: median across all non-equal-to-self elements in kernel
    # margin = -1: median across all elements in kernel
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel_size // 2

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(k, i_range - k):
        for xline in range(k, x_range - k):
            central = src[iline, xline]

            if (preserve_missings is True) and isnan(central):
                continue # Do nothing with nans

            element = src[iline-k:iline+k+1, xline-k:xline+k+1].ravel()

            # Find elements which are close or distant for the `central`
            # 0 for close, 1 for distant, 2 for nan
            indicator = np.zeros_like(element)

            for i, item in enumerate(element):
                if not isnan(item):
                    if (abs(item - central) > margin) or isnan(central):
                        indicator[i] = np.float32(1)
                else:
                    indicator[i] = np.float32(2)

            # If there are more close points than distant in the window, then find median of close points
            n_close = (indicator == np.float32(0)).sum()
            mask_distant = indicator == np.float32(1)
            n_distant = mask_distant.sum()
            if n_distant > n_close:
                dst[iline, xline] = np.median(element[mask_distant])
    return dst

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
        valid_points_indices = np.argwhere((0 < indices_2) & (indices_2 < max_idx_2))

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
