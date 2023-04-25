""" Helpers for the natural 3-dimensional space coordinates processing. """
import numpy as np
from numba import njit
import cv2 as cv

# Coordinates operations
def dilate_coords(coords, dilate=3, axis=0, max_value=None):
    """ Dilate coordinates with (dilate, 1) structure along the given axis.

    Note, the function returns unique and sorted coords.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Coordinates to dilate along the axis. Sorting is not required.
    axis : {0, 1, 2}
        Axis along which to dilate coordinates.
    max_value : None or int, optional
        The maximum possible value for coordinates along the provided axis.
        Used for values clipping into valid range.
    """
    dilated_coords = np.tile(coords, (dilate, 1))

    # Create dilated coordinates
    for i in range(dilate):
        start_idx, end_idx = i*len(coords), (i + 1)*len(coords)
        dilated_coords[start_idx:end_idx, axis] += i - dilate//2

    # Clip to the valid values
    mask = dilated_coords[:, axis] >= 0

    if max_value is not None:
        mask &= dilated_coords[:, axis] < max_value

    dilated_coords = dilated_coords[mask]

    # Get sorted unique values
    dilated_coords = np.unique(dilated_coords, axis=0)
    return dilated_coords

@njit
def depthwise_groupby_max(coords, values):
    """ Thin coordinates depend on values - choose coords corresponding to max values for each coordinate along
    the last axis (depth). Rough approximation of `find_peaks`.

    Under the hood, this function is a groupby max along the depth axis.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Coordinates for thinning. Sorting is not required.
    values : np.ndarray of (N, 1) shape
        Values corresponding to coordinates to decide which one should last for each depth (last column in coords).
    """
    order = np.argsort(coords[:, -1])

    output_coords = np.zeros_like(coords)
    output_values = np.zeros_like(values)
    position = 0

    idx = order[0]

    argmax_coord = coords[idx, :]
    max_value = values[idx]

    previous_depth = argmax_coord[-1]
    previous_value = values[idx]

    for idx in order[1:]:
        current_depth = coords[idx, -1]
        current_value = values[idx]

        if previous_depth == current_depth:
            if previous_value < current_value:
                argmax_coord = coords[idx, :]
                max_value = current_value

                previous_value = current_value

        else:
            output_coords[position, :] = argmax_coord
            output_values[position] = max_value

            position += 1

            argmax_coord = coords[idx, :]
            max_value = current_value

            previous_depth = current_depth
            previous_value = current_value

    # last depth update
    output_coords[position, :] = argmax_coord
    output_values[position] = max_value
    position += 1

    return output_coords[:position, :], output_values[:position]

# Distance evaluation
@njit
def bboxes_intersected(bbox_1, bbox_2, axes=(0, 1, 2)):
    """ Check bounding boxes intersection on preferred axes.

    Bboxes are intersected if they have at least 1 overlapping point.

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    axes : sequence of int values from {0, 1, 2}
        Axes to check bboxes intersection.
    """
    for axis in axes:
        overlap_size = min(bbox_1[axis, 1], bbox_2[axis, 1]) - max(bbox_1[axis, 0], bbox_2[axis, 0]) + 1

        if overlap_size < 1:
            return False
    return True

@njit
def bboxes_adjacent(bbox_1, bbox_2, adjacency=1):
    """ Bounding boxes adjacency ranges.

    Bboxes are adjacent if they are distant not more than on `adjacency` points.

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    adjacency : int
        Amount of points between two bboxes to decide that they are adjacent.
    """
    borders = np.empty((3, 2), dtype=np.int32)

    for i in range(3):
        borders_i_0 = max(bbox_1[i, 0], bbox_2[i, 0])
        borders_i_1 = min(bbox_1[i, 1], bbox_2[i, 1])

        if borders_i_1 - borders_i_0 < -adjacency:
            return None

        borders[i, 0] = min(borders_i_0, borders_i_1)
        borders[i, 1] = max(borders_i_0, borders_i_1)

    return borders

def bboxes_embedded(bbox_1, bbox_2, margin=3):
    """ Check that one bounding box is inside in another (embedded).

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    margin : int
        Possible bboxes difference (on each axis) to decide that one is inside another.
    """
    swap = np.count_nonzero(bbox_1[:, 1] >= bbox_2[:, 1]) <= 1 # is second not inside first

    if swap:
        bbox_1, bbox_2 = bbox_2, bbox_1

    for i in range(3):
        is_embedded = (bbox_2[i, 0] >= bbox_1[i, 0] - margin) and (bbox_2[i, 1] <= bbox_1[i, 1] + margin)

        if not is_embedded:
            return is_embedded, swap

    return is_embedded, swap

@njit
def compute_distances(coords_1, coords_2, max_threshold=10000):
    """ Find approximate minimal and maximal distances between two arrays of coordinates.

    A little bit faster than difference between np.ndarrays with `np.max` and `np.min`.

    Parameters
    ----------
    coords_1, coords_2 : np.ndarrays of (N, 1) shape
        Coords for which find distances. Must be unique values, sorting is not required.
    max_threshold : int, float or None
        Early stopping: threshold for max distance value.
    """
    min_distance = max_threshold
    max_distance = 0

    for coord_1, coord_2 in zip(coords_1, coords_2):
        distance = np.abs(coord_1 - coord_2)

        if distance >= max_threshold:
            return -1, distance

        if distance > max_distance:
            max_distance = distance

        if distance < min_distance:
            min_distance = distance

    return min_distance, max_distance

def find_contour(coords, projection_axis):
    """ Find closed contour of coords projection.

    Under the hood, we make a 2D coords projection and find its contour.

    Note, returned contour coordinates are equal to 0 for the projection axis.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        3D object coordinates. Sorting is not required.
    projection_axis : {0, 1}
        Axis for making 2D projection.
        Note, this function doesn't work for axis = 2.
    """
    bbox = np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)])
    bbox = bbox[(1 - projection_axis, 2), :]

    # Create object image mask
    origin = bbox[:, 0]
    image_shape = bbox[:, 1] - bbox[:, 0] + 1

    mask = np.zeros(image_shape, np.uint8)
    mask[coords[:, 1 - projection_axis] - origin[0], coords[:, 2] - origin[1]] = 1

    # Get only the main object contour: object can contain holes with their own contours
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    # Extract unique and sorted coords
    contour = contours[0].reshape(len(contours[0]), 2) # Can be non-unique

    contour_coords = np.zeros((len(contour), 3), np.int32)
    contour_coords[:, 1 - projection_axis] = contour[:, 1] + origin[0]
    contour_coords[:, 2] = contour[:, 0] + origin[1]

    contour_coords = np.unique(contour_coords, axis=0) # np.unique is here for sorting and unification
    return contour_coords

@njit
def restore_coords_from_projection(coords, projection_buffer, axis):
    """ Restore `axis` coordinates for 2D projection from original coordinates.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Original coords from which restore the axis values. Sorting is not required.
    projection_buffer : np.ndarray
        Buffer with projection coordinates. Sorting is not required.
        Note, it is changed inplace.
    axis : {0, 1, 2}
        Axis for which restore coordinates.
    """
    known_axes = np.array([i for i in range(3) if i != axis])

    for i, buffer_line in enumerate(projection_buffer):
        values =  coords[(coords[:, known_axes[0]] == buffer_line[known_axes[0]]) & \
                         (coords[:, known_axes[1]] == buffer_line[known_axes[1]]),
                         axis]

        projection_buffer[i, axis] = min(values) if len(values) > 0 else -1

    projection_buffer = projection_buffer[projection_buffer[:, axis] != -1]
    return projection_buffer
