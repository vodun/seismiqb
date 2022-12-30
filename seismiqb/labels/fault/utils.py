""" Helpers for coordinates processing. """
import numpy as np
from numba import njit

from ...utils import groupby_min, groupby_max, groupby_all

# Coordinates operations
def dilate_coords(coords, dilate=3, axis=0, max_value=None):
    """ Dilate coordinates with (dilate, 1) structure. """
    dilated_coords = np.tile(coords, (dilate, 1))

    for counter, i in enumerate(range(-(dilate//2), dilate//2 + 1)):
        start_idx, end_idx = counter*len(coords), (counter + 1)*len(coords)
        dilated_coords[start_idx:end_idx, axis] += i

    if max_value is not None:
        dilated_coords = dilated_coords[(dilated_coords[:, axis] >= 0) & (dilated_coords[:, axis] <= max_value)]
    else:
        dilated_coords = dilated_coords[dilated_coords[:, axis] >= 0]

    dilated_coords = np.unique(dilated_coords, axis=0) # TODO: think about np.unique replacement
    return dilated_coords

@njit
def thin_coords(coords, values):
    """ Thin coords depend on values (choose coordinates corresponding to max values along the last axis).
    Rough approximation of `find_peaks` for coordinates.
    """
    order = np.argsort(coords[:, -1])[::-1]

    output = np.zeros_like(coords)
    position = 0

    idx = order[0]

    point_to_save = coords[idx, :]
    previous_depth = point_to_save[-1]
    previous_value = values[idx]

    for i in range(1, len(coords)):
        idx = order[i]
        current_depth = coords[idx, -1]
        current_value = values[idx]

        if previous_depth == current_depth:
            if previous_value < current_value:
                point_to_save = coords[idx, :]

                previous_value = current_value

        else:
            output[position, :] = point_to_save

            position += 1

            point_to_save = coords[idx, :]
            previous_depth = current_depth
            previous_value = current_value

    # last depth update
    output[position, :] = point_to_save
    position += 1

    return output[:position, :]

# Set operations
@njit
def n_differences_for_coords(coords_1, coords_2, max_threshold=None):
    """ Calculate amount of values which are presented in coords_1 and NOT in coords_2.

    Note, that coords must be sorted by all columns.
    # `max_threshold` for early exit
    ..!!..
    """
    # Initial values
    n_presented_only_in_coords_1 = 0

    point_1 = coords_1[0, :]
    point_2 = coords_2[0, :]

    counter_1 = 1
    counter_2 = 1

    # Iter over coordinates arrays
    while (counter_1 < len(coords_1)) and (counter_2 < len(coords_2)):
        previous_point_1 = point_1

        if (point_1 == point_2).all():
            point_1, counter_1 = _iter_next(point=point_1, coords=coords_1, counter=counter_1)
            point_2, counter_2 = _iter_next(point=point_2, coords=coords_2, counter=counter_2)
        else:
            diff = point_1 - point_2

            for elem in diff:
                if elem > 0:
                    point_2, counter_2 = _iter_next(point=point_2, coords=coords_2, counter=counter_2)
                    break

                if elem < 0:
                    n_presented_only_in_coords_1 += 1

                    if (max_threshold is not None) and (n_presented_only_in_coords_1 >= max_threshold):
                        return n_presented_only_in_coords_1

                    point_1, counter_1 = _iter_next(point=point_1, coords=coords_1, counter=counter_1)
                    break

    # Encounter last element in coords_1
    if counter_1 < len(coords_1):
        # We iter over the coords_2 array, but some points in coords_1 weren't encountered
        n_presented_only_in_coords_1 += n_unique(coords_1[counter_1-1:])
    else:
        # We iter over the coords_1 array, but the last point in coords_1 wasn't encountered
        if (point_1 != previous_point_1).any():
            n_presented_only_in_coords_1 += 1

    return n_presented_only_in_coords_1

@njit
def _iter_next(point, coords, counter):
    """ Iterating for non-unique elements.

    Note, that this function needs sorted data.
    """
    previous_point = point

    while (counter < len(coords)) and (point == previous_point).all():
        point = coords[counter, :]
        counter += 1

    return point, counter

@njit
def n_unique(array):
    """ Number of unique rows in array.

    Roughly similar to len(np.unique(array, axis=0)), we need this implementation because
    numba doesn't support `axis` parameter for `np.unique`.

    Note, that this function needs sorted data.
    """
    previous_row = array[0, :]
    n_unique_rows = 1

    for i in range(1, len(array)):
        if (array[i] != previous_row).any():
            previous_row = array[i]
            n_unique_rows += 1

    return n_unique_rows


# Distance evaluation
def bboxes_intersected(bbox_1, bbox_2, axes=(0, 1, 2)):
    """ Check bboxes intersections on axes. """
    for axis in axes:
        borders_delta = min(bbox_1[axis, 1], bbox_2[axis, 1]) - max(bbox_1[axis, 0], bbox_2[axis, 0])

        if borders_delta < 0:
            return False
    return True

@njit
def bboxes_adjoin(bbox_1, bbox_2, axis=2):
    """ Check that bboxes are adjoint on axis and return intersection/adjoint indices. """
    axis = 2 if axis == -1 else axis

    for i in range(3):
        min_ = min(bbox_1[i, 1], bbox_2[i, 1])
        max_ = max(bbox_1[i, 0], bbox_2[i, 0])

        if min_ - max_ < -1: # distant bboxes
            return None, None

        if i == axis:
            intersection_borders = (min_, max_)

    return min(intersection_borders), max(intersection_borders) # intersection / adjoint indices for the axis

@njit
def max_depthwise_distance(coords_1, coords_2, depths_ranges, step, axis, max_threshold=None):
    """ Find maximal depth-wise central distance between coordinates."""
    max_distance = 0

    for depth in range(depths_ranges[0], depths_ranges[1]+1, step):
        coords_1_depth_slice = coords_1[coords_1[:, -1] == depth, axis]
        coords_2_depth_slice = coords_2[coords_2[:, -1] == depth, axis]

        distance = np.abs(coords_1_depth_slice[len(coords_1_depth_slice)//2] - \
                          coords_2_depth_slice[len(coords_2_depth_slice)//2])

        if (max_threshold is not None) and (distance >= max_threshold):
            return distance

        if distance > max_distance:
            max_distance = distance

    return max_distance


# Object oriented operations
def find_border(coords, find_lower_border, projection_axis):
    """ Find non-closed border part of the 3d object (upper or lower border).

    Under the hood, we find border of a 2d projection on `projection_axis` and restore 3d coordinates.
    ..!!..

    Parameters
    ----------
    find_lower_border : bool
        Find lower or upper border for object.
    """
    anchor_axis = 1 if projection_axis == 0 else 0

    # Make 2d projection on projection_axis
    points_projection = coords.copy()
    points_projection[:, projection_axis] = 0

    # Find depth-wise contour
    depthwise_contour = groupby_max(points_projection) if find_lower_border else groupby_min(points_projection)

    # Find non-projection axis-wise contour
    # We swap anchor and depths axes for correct groupby appliance (groupby is applied only for the last axis)
    transposed_projection = points_projection
    transposed_projection[:, [-1, anchor_axis]] = transposed_projection[:, [anchor_axis, -1]]
    transposed_projection = transposed_projection[transposed_projection[:, 0].argsort()] # groupby needs sorted data

    # Get min and max values (left and right borders)
    anchor_wise_group = groupby_all(transposed_projection)

    depths = anchor_wise_group[:, anchor_axis]
    anchor_axis_mins = anchor_wise_group[:, 3]
    anchor_axis_maxs = anchor_wise_group[:, 4]

    anchor_axis_contour = np.zeros((len(depths)*2, 3), int)

    anchor_axis_contour[:len(depths), anchor_axis] = anchor_axis_mins
    anchor_axis_contour[:len(depths), -1] = depths

    anchor_axis_contour[len(depths):, anchor_axis] = anchor_axis_maxs
    anchor_axis_contour[len(depths):, -1] = depths

    contour_points = np.concatenate([anchor_axis_contour, depthwise_contour], axis=0)

    # Restore 3d coordinates
    contour_points = restore_coords_from_projection(coords=coords, buffer=contour_points, axis=projection_axis)
    return contour_points

@njit
def restore_coords_from_projection(coords, buffer, axis):
    """ Find `axis` coordinates from coordinates and their projection.

    ..!!..
    """
    known_axes = [i for i in range(3) if i != axis]

    for i, buffer_line in enumerate(buffer):
        buffer[i, axis] = min(coords[(coords[:, known_axes[0]] == buffer_line[known_axes[0]]) & \
                                     (coords[:, known_axes[1]] == buffer_line[known_axes[1]]),
                                     axis])
    return buffer
