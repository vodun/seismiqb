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
        Coordinates to dilate along the axis.
    axis : {0, 1, 2}
        Axis along which to dilate coordinates.
    max_value : None or int, optional
        The maximum possible value for coordinates along the provided axis.
    """
    dilated_coords = np.tile(coords, (dilate, 1))

    # Create dilated coordinates
    for i in range(dilate):
        start_idx, end_idx = i*len(coords), (i + 1)*len(coords)
        dilated_coords[start_idx:end_idx, axis] += i - dilate//2

    # Clip to the valid values
    if max_value is not None:
        dilated_coords = dilated_coords[(dilated_coords[:, axis] >= 0) & (dilated_coords[:, axis] <= max_value)]
    else:
        dilated_coords = dilated_coords[dilated_coords[:, axis] >= 0]

    # Get sorted unique values
    dilated_coords = np.unique(dilated_coords, axis=0)
    return dilated_coords

@njit
def thin_coords(coords, values):
    """ Thin coordinates depend on values - choose coords corresponding to max values for each coordinate along
    the last axis (depth). Rough approximation of `find_peaks`.

    Under the hood, this function is a groupby max along the depth axis.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Coordinates for thinning.
    values : np.ndarray of (N, 1) shape
        Values corresponding to coordinates to decide which one should last for each depth (last column in coords).
    """
    order = np.argsort(coords[:, -1])[::-1]

    output = np.zeros_like(coords)
    position = 0

    idx = order[0]

    argmax_coord = coords[idx, :]
    previous_depth = argmax_coord[-1]
    previous_value = values[idx]

    for idx in order[1:]:
        current_depth = coords[idx, -1]
        current_value = values[idx]

        if previous_depth == current_depth:
            if previous_value < current_value:
                argmax_coord = coords[idx, :]

                previous_value = current_value

        else:
            output[position, :] = argmax_coord

            position += 1

            argmax_coord = coords[idx, :]
            previous_depth = current_depth
            previous_value = current_value

    # last depth update
    output[position, :] = argmax_coord
    position += 1

    return output[:position, :]

# Distance evaluation
@njit
def bboxes_intersected(bbox_1, bbox_2, axes=(0, 1, 2)):
    """ Check bboxes intersection on preferred axes.

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
def bboxes_adjacent(bbox_1, bbox_2):
    """ Bboxes intersection or adjacency ranges if bboxes are intersected/adjacent.

    Bboxes are adjacent if they are distant not more than on 1 point.

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
    """
    borders = np.empty((3, 2), dtype=np.int32)

    for i in range(3):
        borders_i_0 = max(bbox_1[i, 0], bbox_2[i, 0])
        borders_i_1 = min(bbox_1[i, 1], bbox_2[i, 1])

        if borders_i_1 - borders_i_0 < -1:
            return None

        borders[i, 0] = min(borders_i_0, borders_i_1)
        borders[i, 1] = max(borders_i_0, borders_i_1)

    return borders

def bboxes_embedded(bbox_1, bbox_2, margin=3):
    """ Check that one bbox is embedded in another.

    Parameters
    ----------
    bbox_1, bbox_2 : np.ndarrays of (3, 2) shape.
        Objects bboxes.
    margin : int
        Possible bboxes difference (on each axis) to decide that one is in another.
    """
    is_second_inside_first = np.count_nonzero(bbox_1[:, 1] >= bbox_2[:, 1]) > 1

    if not is_second_inside_first:
        bbox_1, bbox_2 = bbox_2.copy(), bbox_1.copy()
    else:
        bbox_1, bbox_2 = bbox_1.copy(), bbox_2.copy()

    bbox_1[:, 0] -= margin
    bbox_1[:, 1] += margin

    is_embedded = (bbox_2[:, 0] >= bbox_1[:, 0]).all() and (bbox_2[:, 1] <= bbox_1[:, 1]).all()
    return is_embedded, is_second_inside_first

@njit
def min_max_depthwise_distances(coords_1, coords_2, depths_ranges, step, axis, max_threshold=None):
    """ Find approximate minimal and maximal axis-wise central distances between coordinates.

    Parameters
    ----------
    coords_1, coords_2 : np.ndarrays of (N, 3) shape
        Coords for which find axis-wise distances.
    depths_ranges : sequence of two int values
        Depth ranges along which to find distances.
        Recommended to provide depth intersection ranges for coords.
    step : int
        Depth step to find distances.
        For speed up we can find distances not for all depths.
        For results accuracy, provide step = 1.
    axis : {0, 1}
        Axis along which to find distances.
    max_threshold : int, float or None
        Early stopping: threshold for max distance value.
        If the max value is more than this value, then no sense in finding more accurate results
        because coordinates are too far from each other.
        If None, then o threshold is applied.
    """
    min_distance = max_threshold or 100
    max_distance = 0

    for depth in range(depths_ranges[0], depths_ranges[1]+1, step):
        # Find distance for the current depth
        coords_1_depth_slice = coords_1[coords_1[:, -1] == depth, axis]
        coords_2_depth_slice = coords_2[coords_2[:, -1] == depth, axis]

        distance = np.abs(coords_1_depth_slice[len(coords_1_depth_slice)//2] - \
                          coords_2_depth_slice[len(coords_2_depth_slice)//2])

        # Result processing
        if (max_threshold is not None) and (distance >= max_threshold):
            return None, distance

        if distance > max_distance:
            max_distance = distance

        if distance < min_distance:
            min_distance = distance

    return min_distance, max_distance

def find_contour(coords, projection_axis):
    """ Find closed contour of 2d projection.

    Note, returned contour coordinates are equal to 0 for the projection axis.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        3D object coordinates.
    projection_axis : {0, 1}
        Axis for making 2d projection for which we find contour.
        Note, this function doesn't work for axis = 2.
    """
    # Make 2d projection
    bbox = np.column_stack([np.min(coords, axis=0), np.max(coords, axis=0)])
    bbox = np.delete(bbox, projection_axis, 0)

    # Create object mask
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
def restore_coords_from_projection(coords, buffer, axis):
    """ Restore `axis` coordinates for 2d projection from original coordinates.

    Parameters
    ----------
    coords : np.ndarray of (N, 3) shape
        Original coords from which restore the axis values.
    buffer : np.ndarray
        Buffer with projection coordinates.
        Note, t is changed inplace.
    axis : {0, 1, 2}
        Axis for which restore coordinates.
    """
    known_axes = np.array([i for i in range(3) if i != axis])

    for i, buffer_line in enumerate(buffer):
        values =  coords[(coords[:, known_axes[0]] == buffer_line[known_axes[0]]) & \
                         (coords[:, known_axes[1]] == buffer_line[known_axes[1]]),
                         axis]

        buffer[i, axis] = min(values) if len(values) > 0 else -1

    buffer = buffer[buffer[:, axis] != -1]
    return buffer
