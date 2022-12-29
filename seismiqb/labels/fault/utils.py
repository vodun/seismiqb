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

    line_to_save = coords[idx, :]
    previous_depth = line_to_save[-1]
    previous_value = values[idx]

    for i in range(1, len(coords)):
        idx = order[i]
        current_depth = coords[idx, -1]
        current_value = values[idx]

        if previous_depth == current_depth:
            if previous_value < current_value:
                line_to_save = coords[idx, :]

                previous_value = current_value

        else:
            output[position, :] = line_to_save

            position += 1

            line_to_save = coords[idx, :]
            previous_depth = current_depth
            previous_value = current_value

    # last depth update
    output[position, :] = line_to_save
    position += 1

    return output[:position, :]

# Set operations
@njit
def _iter_next(point, previous_point, coords, counter):
    """ Iterating for non-unique elements.

    Note, that this function needs sorted data.
    """
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

@njit
def n_differences_for_coords(coords_1, coords_2, max_threshold=None):
    """ Calculate amount of values which are presented in coords_1 and not in coords_2.

    Note, that coords must be sorted by all axes.
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
        previous_point_2 = point_2

        if (point_1 == point_2).all():
            point_1, counter_1 = _iter_next(point=point_1, previous_point=previous_point_1,
                                            coords=coords_1, counter=counter_1)
            point_2, counter_2 = _iter_next(point=point_2, previous_point=previous_point_2,
                                            coords=coords_2, counter=counter_2)
        else:
            diff = point_1 - point_2

            for elem in diff:
                if elem > 0:
                    point_2, counter_2 = _iter_next(point=point_2, previous_point=previous_point_2,
                                                    coords=coords_2, counter=counter_2)
                    break

                if elem < 0:
                    n_presented_only_in_coords_1 += 1

                    if (max_threshold is not None) and (n_presented_only_in_coords_1 >= max_threshold):
                        return n_presented_only_in_coords_1

                    point_1, counter_1 = _iter_next(point=point_1, previous_point=previous_point_1,
                                                    coords=coords_1, counter=counter_1)
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

# Distance evaluation
def bboxes_intersected(bbox_1, bbox_2, axes=(0, 1, 2)):
    """ Check bboxes intersections on axes. """
    for axis in axes:
        delta = min(bbox_1[axis, 1], bbox_2[axis, 1]) - max(bbox_1[axis, 0], bbox_2[axis, 0])

        if delta < 0:
            return False
    return True

@njit
def bboxes_adjoin(bbox_1, bbox_2, axis=2):
    """ Check that bboxes are adjoint on axis. """
    axis = 2 if axis == -1 else axis

    for i in range(3):
        min_ = min(bbox_1[i, 1], bbox_2[i, 1])
        max_ = max(bbox_1[i, 0], bbox_2[i, 0])

        if min_ - max_ < -1: # distant bboxes
            return None, None

        if i == axis:
            intersection_borders = (min_, max_)

    return min(intersection_borders), max(intersection_borders) # intersection / adjoinance indices for the axis

@njit
def depthwise_distances(coords_1, coords_2, depths_ranges, step, axis, max_threshold=None):
    """ Depth-wise central distances between coordinates."""
    borders_distance = -1

    for depth in range(depths_ranges[0], depths_ranges[1]+1, step):
        coords_1_depth_slice = coords_1[coords_1[:, -1] == depth, axis]
        coords_2_depth_slice = coords_2[coords_2[:, -1] == depth, axis]

        distance = np.abs(coords_1_depth_slice[len(coords_1_depth_slice)//2] - \
                          coords_2_depth_slice[len(coords_2_depth_slice)//2])

        if (max_threshold is not None) and (distance >= max_threshold):
            return distance

        if distance > borders_distance:
            borders_distance = distance

    return borders_distance


# Object oriented operations
def find_border(coords, find_lower_border, projection_axis):
    """ Find non-closed border part of the 3d object (upper or lower border).

    Under the hood, we find border of a 2d projection on `projection_axis` and restore 3d coordinates.
    ..!!..
    """
    ankhor_axis = 1 if projection_axis == 0 else 0

    # Make 2d projection on projection_axis
    points_projection = coords.copy()
    points_projection[:, projection_axis] = 0

    # Find depth-wise contour
    if find_lower_border:
        deptwise_contour = groupby_max(points_projection)
    else:
        deptwise_contour = groupby_min(points_projection)

    # Find non-projection axis-wise contour
    # We swap ankhor_axis and depths axis for correct groupby appliance (groupby is applied only for the last axis)
    transposed_projection = points_projection
    transposed_projection[:, [-1, ankhor_axis]] = transposed_projection[:, [ankhor_axis, -1]]
    transposed_projection = transposed_projection[transposed_projection[:, 0].argsort()]

    # Get min and max values (left and right borders)
    orientation_wise_group = groupby_all(transposed_projection)

    depths = orientation_wise_group[:, ankhor_axis]
    orientation_mins = orientation_wise_group[:, 3]
    orientation_maxs = orientation_wise_group[:, 4]

    orientation_contour = np.zeros((len(depths)*2, 3), int)

    orientation_contour[:len(depths), ankhor_axis] = orientation_mins
    orientation_contour[:len(depths), -1] = depths

    orientation_contour[len(depths):, ankhor_axis] = orientation_maxs
    orientation_contour[len(depths):, -1] = depths

    contour_points = np.concatenate([orientation_contour, deptwise_contour], axis=0)

    # Restore 3d coordinates
    contour_points = restore_coords_from_projection(coords=coords, buffer=contour_points, axis=projection_axis)
    return contour_points

@njit
def restore_coords_from_projection(coords, buffer, axis):
    """ Find `axis` coordinates from projection.

    ..!!..
    """
    known_axes = [i for i in range(3) if i != axis]

    for i, buffer_line in enumerate(buffer):
        buffer[i, axis] = min(coords[(coords[:, known_axes[0]] == buffer_line[known_axes[0]]) & \
                                     (coords[:, known_axes[1]] == buffer_line[known_axes[1]]),
                                     axis])
    return buffer
