""" Utility functions. """
from math import isnan, atan
import inspect

from tqdm import tqdm
import numpy as np
import pandas as pd
import segyio
import torch
import torch.nn.functional as F

from numba import njit, prange



def file_print(msg, path, mode='w'):
    """ Print to file. """
    # pylint: disable=redefined-outer-name
    with open(path, mode) as file:
        print(msg, file=file)



#TODO: rethink
def make_subcube(path, geometry, path_save, i_range, x_range):
    """ Make subcube from .sgy cube by removing some of its first and
    last ilines and xlines.

    Parameters
    ----------
    path : str
        Location of original .sgy cube.
    geometry : SeismicGeometry
        Infered information about original cube.
    path_save : str
        Place to save subcube.
    i_range : array-like
        Ilines to include in subcube.
    x_range : array-like
        Xlines to include in subcube.

    Notes
    -----
    Common use of this function is to remove not fully filled slices of .sgy cubes.
    """
    i_low, i_high = i_range[0], i_range[-1]
    x_low, x_high = x_range[0], x_range[-1]

    with segyio.open(path, 'r', strict=False) as src:
        src.mmap()
        spec = segyio.spec()
        spec.sorting = int(src.sorting)
        spec.format = int(src.format)
        spec.samples = range(geometry.depth)
        spec.ilines = geometry.ilines[i_low:i_high]
        spec.xlines = geometry.xlines[x_low:x_high]

        with segyio.create(path_save, spec) as dst:
            # Copy all textual headers, including possible extended
            for i in range(1 + src.ext_headers):
                dst.text[i] = src.text[i]

            c = 0
            for il_ in tqdm(spec.ilines):
                for xl_ in spec.xlines:
                    tr_ = geometry.il_xl_trace[(il_, xl_)]
                    dst.header[c] = src.header[tr_]
                    dst.header[c][segyio.TraceField.FieldRecord] = il_
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_FILE] = il_

                    dst.header[c][segyio.TraceField.TraceNumber] = xl_ - geometry.xlines_offset
                    dst.header[c][segyio.TraceField.TRACE_SEQUENCE_LINE] = xl_ - geometry.xlines_offset
                    dst.trace[c] = src.trace[tr_]
                    c += 1
            dst.bin = src.bin
            dst.bin = {segyio.BinField.Traces: c}

    # Check that repaired cube can be opened in 'strict' mode
    with segyio.open(path_save, 'r', strict=True) as _:
        pass

#TODO: rename, add some defaults
def convert_point_cloud(path, path_save, names=None, order=None, transform=None):
    """ Change set of columns in file with point cloud labels.
    Usually is used to remove redundant columns.

    Parameters
    ----------
    path : str
        Path to file to convert.
    path_save : str
        Path for the new file to be saved to.
    names : str or sequence of str
        Names of columns in the original file. Default is Petrel's export format, which is
        ('_', '_', 'iline', '_', '_', 'xline', 'cdp_x', 'cdp_y', 'height'), where `_` symbol stands for
        redundant keywords like `INLINE`.
    order : str or sequence of str
        Names and order of columns to keep. Default is ('iline', 'xline', 'height').
    """
    #pylint: disable=anomalous-backslash-in-string
    names = names or ['_', '_', 'iline', '_', '_', 'xline',
                      'cdp_x', 'cdp_y', 'height']
    order = order or ['iline', 'xline', 'height']

    names = [names] if isinstance(names, str) else names
    order = [order] if isinstance(order, str) else order

    df = pd.read_csv(path, sep='\s+', names=names, usecols=set(order))
    df.dropna(inplace=True)

    if 'iline' in order and 'xline' in order:
        df.sort_values(['iline', 'xline'], inplace=True)

    data = df.loc[:, order]
    if transform:
        data = data.apply(transform)
    data.to_csv(path_save, sep=' ', index=False, header=False)


def save_point_cloud(metric, save_path, geometry=None):
    """ Save 2D map as a .txt point cloud. Can be opened by GENERAL format reader in geological software. """
    idx_1, idx_2 = np.asarray(~np.isnan(metric)).nonzero()
    points = np.hstack([idx_1.reshape(-1, 1),
                        idx_2.reshape(-1, 1),
                        metric[idx_1, idx_2].reshape(-1, 1)])

    if geometry is not None:
        points[:, 0] += geometry.ilines_offset
        points[:, 1] += geometry.xlines_offset

    df = pd.DataFrame(points, columns=['iline', 'xline', 'metric_value'])
    df.sort_values(['iline', 'xline'], inplace=True)
    df.to_csv(save_path, sep=' ', columns=['iline', 'xline', 'metric_value'],
              index=False, header=False)


def gen_crop_coordinates(point, horizon_matrix, zero_traces,
                         stride, shape, depth, fill_value, zeros_threshold=0,
                         empty_threshold=5, safe_stripe=0, num_points=2):
    """ Generate crop coordinates next to the point with maximum horizon covered area.

    Parameters
    ----------
    point : array-like
        Coordinates of the point.
    horizon_matrix : ndarray
        `Full_matrix` attribute of the horizon.
    zero_traces : ndarray
        A boolean ndarray indicating zero traces in the cube.
    stride : int
        Distance between the point and a corner of a crop.
    shape : array-like
        The desired shape of the crops.
        Note that final shapes are made in both xline and iline directions. So if
        crop_shape is (1, 64, 64), crops of both (1, 64, 64) and (64, 1, 64) shape
        will be defined.    fill_value : int
    zeros_threshold : int
        A maximum number of bad traces in a crop.
    empty_threshold : int
        A minimum number of points with unknown horizon per crop.
    safe_stripe : int
        Distance between a crop and the ends of the cube.
    num_points : int
        Returned number of crops. The maximum is four.
    """
    candidates, shapes = [], []
    orders, intersections = [], []
    hor_height = horizon_matrix[point[0], point[1]]
    ilines_len, xlines_len = horizon_matrix.shape

    tested_iline_positions = [max(0, point[0] - stride),
                              min(point[0] - shape[1] + stride, ilines_len - shape[1])]

    for il in tested_iline_positions:
        if (il > safe_stripe) and (il + shape[1] < ilines_len - safe_stripe):
            num_missing_traces = np.sum(zero_traces[il: il + shape[1],
                                                    point[1]: point[1] + shape[0]])
            if num_missing_traces <= zeros_threshold:
                horizon_patch = horizon_matrix[il: il + shape[1],
                                               point[1]:point[1] + shape[0]]
                num_empty = np.sum(horizon_patch == fill_value)
                if num_empty > empty_threshold:
                    candidates.append([il, point[1],
                                       min(hor_height - shape[2] // 2, depth - shape[2] - 1)])
                    shapes.append([shape[1], shape[0], shape[2]])
                    orders.append([0, 2, 1])
                    intersections.append(shape[1] - num_empty)

    tested_xline_positions = [max(0, point[1] - stride),
                              min(point[1] - shape[1] + stride, xlines_len - shape[1])]

    for xl in tested_xline_positions:
        if (xl > safe_stripe) and (xl + shape[1] < xlines_len - safe_stripe):
            num_missing_traces = np.sum(zero_traces[point[0]: point[0] + shape[0],
                                                    xl: xl + shape[1]])
            if num_missing_traces <= zeros_threshold:
                horizon_patch = horizon_matrix[point[0]:point[0] + shape[0],
                                               xl: xl + shape[1]]
                num_empty = np.sum(horizon_patch == fill_value)
                if num_empty > empty_threshold:
                    candidates.append([point[0], xl,
                                       min(hor_height - shape[2] // 2, depth - shape[2] - 1)])
                    shapes.append(shape)
                    orders.append([2, 0, 1])
                    intersections.append(shape[1] - num_empty)

    if len(candidates) == 0:
        return None

    candidates_array = np.array(candidates)
    shapes_array = np.array(shapes)
    orders_array = np.array(orders)

    top = np.argsort(np.array(intersections))[:num_points]
    return (candidates_array[top], \
                shapes_array[top], \
                orders_array[top])


def make_axis_grid(axis_range, stride, length, crop_shape):
    """ Make separate grids for every axis. """
    grid = np.arange(*axis_range, stride)
    grid_ = [x for x in grid if x + crop_shape < length]
    if len(grid) != len(grid_):
        grid_ += [axis_range[1] - crop_shape]
    return sorted(grid_)

def fill_defaults(value, default):
    """ Transform int or tuple with Nones to tuple with values from default.

    Parameters
    ----------
    value : None, int or tuple
        value to transform
    default : tuple

    Returns
    -------
    tuple

    Examples
    --------
        None --> default
        5 --> (5, 5, 5)
        (None, None, 3) --> (default[0], default[1], 3)
    """
    if value is None:
        value = default
    elif isinstance(value, int):
        value = tuple([value] * 3)
    elif isinstance(value, tuple):
        value = tuple([item if item else default[i] for i, item in enumerate(value)])
    return value

def parse_axis(axis, index_headers=None):
    """ Convert string representation of an axis into integer, if needed. """
    if isinstance(axis, str):
        if index_headers and axis in index_headers:
            axis = index_headers.index(axis)
        elif axis in ['i', 'il', 'iline']:
            axis = 0
        elif axis in ['x', 'xl', 'xline']:
            axis = 1
        elif axis in ['h', 'height', 'depth']:
            axis = 2
    return axis

def _adjust_shape_for_rotation(shape, angle):
    """ Compute adjusted 2D crop shape to rotate it and get central crop without padding.

    Parameters
    ----------
    shape : tuple
        Target сrop shape.
    angle : float

    Returns
    -------
    tuple
        Adjusted crop shape.
    """
    angle = abs(2 * np.pi * angle / 360)
    limit = atan(shape[1] / shape[0])
    x_max, y_max = shape
    if angle != 0:
        if angle < limit:
            x_max = shape[0] * np.cos(angle) + shape[1] * np.sin(angle) + 1
        else:
            x_max = (shape[0] ** 2 + shape[1] ** 2) ** 0.5 + 1

        if angle < np.pi / 2 - limit:
            y_max = shape[0] * np.sin(angle) + shape[1] * np.cos(angle) + 1
        else:
            y_max = (shape[0] ** 2 + shape[1] ** 2) ** 0.5 + 1
    return (int(np.ceil(x_max)), int(np.ceil(y_max)))

def adjust_shape_3d(shape, angle, scale=(1, 1, 1)):
    """ Compute adjusted 3D crop shape to rotate it and get central crop without padding. Adjustments is based on
    proposition that rotation angles are defined as Tait-Bryan angles and scale performed before rotations.
    The sequence of extrinsic rotations axes is (iline axis, xline axis, depth axis).

    Parameters
    ----------
    shape : tuple
        Target сrop shape.
    angle : float or tuple of floats
        Rotation angle about each axis.
    scale : int or tuple, optional
        Scale for each axis.

    Returns
    -------
    tuple
        Adjusted crop shape.
    """
    angle = angle if isinstance(angle, (tuple, list)) else (angle, 0, 0)
    scale = scale if isinstance(scale, (tuple, list)) else (scale, scale, 1)
    shape = np.ceil(np.array(shape) / np.array(scale)).astype(int)
    if angle[2] != 0:
        shape[2], shape[0] = _adjust_shape_for_rotation((shape[2], shape[0]), angle[2])
    if angle[1] != 0:
        shape[2], shape[1] = _adjust_shape_for_rotation((shape[2], shape[1]), angle[1])
    if angle[0] != 0:
        shape[0], shape[1] = _adjust_shape_for_rotation((shape[0], shape[1]), angle[0])
    return tuple(shape)

@njit
def groupby_mean(array):
    """ Faster version of mean-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s, c = array[0, -1], 1

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s += array[i, -1]
            c += 1
        else:
            output[position, :2] = prev
            output[position, -1] = s / c
            position += 1

            prev = curr
            s, c = array[i, -1], 1

    output[position, :2] = prev
    output[position, -1] = s / c
    position += 1
    return output[:position]


@njit
def groupby_min(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = min(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]


@njit
def groupby_max(array):
    """ Faster version of min-groupby of data along the first two columns.
    Input array is supposed to have (N, 3) shape.
    """
    n = len(array)

    output = np.zeros_like(array)
    position = 0

    prev = array[0, :2]
    s = array[0, -1]

    for i in range(1, n):
        curr = array[i, :2]

        if prev[0] == curr[0] and prev[1] == curr[1]:
            s = max(s, array[i, -1])
        else:
            output[position, :2] = prev
            output[position, -1] = s
            position += 1

            prev = curr
            s = array[i, -1]

    output[position, :2] = prev
    output[position, -1] = s
    position += 1
    return output[:position]


@njit
def round_to_array(values, ticks):
    """ Jit-accelerated function to round values from one array to the
    nearest value from the other in a vectorized fashion. Faster than numpy version.

    Parameters
    ----------
    values : array-like
        Array to modify.
    ticks : array-like
        Values to cast to. Must be sorted in the ascending order.

    Returns
    -------
    array-like
        Array with values from `values` rounded to the nearest from corresponding entry of `ticks`.
    """
    for i, p in enumerate(values):
        if p <= ticks[0]:
            values[i] = ticks[0]
        elif p >= ticks[-1]:
            values[i] = ticks[-1]
        else:
            ix = np.searchsorted(ticks, p)

            if abs(ticks[ix] - p) <= abs(ticks[ix-1] - p):
                values[i] = ticks[ix]
            else:
                values[i] = ticks[ix-1]
    return values


@njit
def find_min_max(array):
    """ Get both min and max values in just one pass through array."""
    n = array.size
    max_val = min_val = array[0]
    for i in range(1, n):
        min_val = min(array[i], min_val)
        max_val = max(array[i], max_val)
    return min_val, max_val



def compute_running_mean(x, kernel_size):
    """ Fast analogue of scipy.signal.convolve2d with gaussian filter. """
    k = kernel_size // 2
    padded_x = np.pad(x, (k, k), mode='symmetric')
    cumsum = np.cumsum(padded_x, axis=1)
    cumsum = np.cumsum(cumsum, axis=0)
    return _compute_running_mean_jit(x, kernel_size, cumsum)

@njit
def _compute_running_mean_jit(x, kernel_size, cumsum):
    """ Jit accelerated running mean. """
    #pylint: disable=invalid-name
    k = kernel_size // 2
    result = np.zeros_like(x).astype(np.float32)

    canvas = np.zeros((cumsum.shape[0] + 2, cumsum.shape[1] + 2))
    canvas[1:-1, 1:-1] = cumsum
    cumsum = canvas

    for i in range(k, x.shape[0] + k):
        for j in range(k, x.shape[1] + k):
            d = cumsum[i + k + 1, j + k + 1]
            a = cumsum[i - k, j  - k]
            b = cumsum[i - k, j + 1 + k]
            c = cumsum[i + 1 + k, j - k]
            result[i - k, j - k] = float(d - b - c + a) /  float(kernel_size ** 2)
    return result


def mode(array):
    """ Compute mode of the array along the last axis. """
    nan_mask = np.max(array, axis=-1)
    return nb_mode(array, nan_mask)

@njit
def nb_mode(array, mask):
    """ Compute mode of the array along the last axis. """
    #pylint: disable=not-an-iterable
    i_range, x_range = array.shape[:2]
    temp = np.full((i_range, x_range), np.nan)

    for il in prange(i_range):
        for xl in prange(x_range):
            if not isnan(mask[il, xl]):

                current = array[il, xl, :]
                counter = {}
                frequency = 0
                for i in current:
                    if i in counter:
                        counter[i] += 1
                    else:
                        counter[i] = 0

                    if counter[i] > frequency:
                        element = i
                        frequency = counter[i]

                temp[il, xl] = element
    return temp


@njit
def filter_simplices(simplices, points, matrix, threshold=5.):
    """ Remove simplices outside of matrix. """
    #pylint: disable=consider-using-enumerate
    mask = np.ones(len(simplices), dtype=np.int32)

    for i in range(len(simplices)):
        tri = points[simplices[i]].astype(np.int32)

        middle_i, middle_x = np.mean(tri[:, 0]), np.mean(tri[:, 1])
        heights = np.array([matrix[tri[0, 0], tri[0, 1]],
                            matrix[tri[1, 0], tri[1, 1]],
                            matrix[tri[2, 0], tri[2, 1]]])

        if matrix[int(middle_i), int(middle_x)] < 0 or np.std(heights) > threshold:
            mask[i] = 0

    return simplices[mask == 1]

def compute_attribute(array, window, device='cuda:0', attribute='semblance'):
    """ Compute semblance for the cube. """
    if isinstance(window, int):
        window = np.ones(3, dtype=np.int32) * window
    window = np.minimum(np.array(window), array.shape)

    inputs = torch.Tensor(array).to(device)
    inputs = inputs.view(1, 1, *inputs.shape)
    padding = [(w // 2, w - w // 2 - 1) for w in window]

    num = F.pad(inputs, (0, 0, *padding[1], *padding[0], 0, 0, 0, 0))
    num = F.conv3d(num, torch.ones((1, 1, window[0], window[1], 1), dtype=torch.float32).to(device)) ** 2
    num = F.pad(num, (*padding[2], 0, 0, 0, 0, 0, 0, 0, 0))
    num = F.conv3d(num, torch.ones((1, 1, 1, 1, window[2]), dtype=torch.float32).to(device))

    denum = F.pad(inputs, (*padding[2], *padding[1], *padding[0], 0, 0, 0, 0))
    denum = F.conv3d(denum ** 2, torch.ones((1, 1, *window), dtype=torch.float32).to(device))

    normilizing = torch.ones(inputs.shape[:-1], dtype=torch.float32).to(device)
    normilizing = F.pad(normilizing, (*padding[1], *padding[0], 0, 0, 0, 0))
    normilizing = F.conv2d(normilizing, torch.ones((1, 1, window[0], window[1]), dtype=torch.float32).to(device))

    denum *= normilizing.view(*normilizing.shape, 1)
    return np.nan_to_num((num / denum).cpu().numpy()[0, 0], nan=1.)

def retrieve_function_arguments(function, dictionary):
    """ Retrieve both positional and keyword arguments for a passed `function` from a `dictionary`.
    Note that retrieved values are removed from the passed `dictionary` in-place. """
    # pylint: disable=protected-access
    parameters = inspect.signature(function).parameters
    arguments_with_defaults = {k: v.default for k, v in parameters.items() if v.default != inspect._empty}
    return {k: dictionary.pop(k, v) for k, v in arguments_with_defaults.items()}
