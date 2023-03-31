""" Utils for set of faults, e.g. filtering. """
import numpy as np
from .coords_utils import bboxes_adjacent


def filter_faults(faults, min_fault_len=2000, min_height=20, **sticks_kwargs):
    """ Filter too small faults.

    Faults are filtered by amount of points, length and height.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` instances
        Faults for filtering.
    min_fault_len : int
        Filtering value: filter out faults with length less than `min_fault_len`.
    min_height : int
        Filtering value: filter out faults with height less than `min_height`.
        Note, that height is evaluated from sticks.
    sticks_kwargs : dict, optional
        Arguments for fault conversion into sticks view.
    """
    config_sticks = {
        'sticks_step': 10,
        'stick_nodes_step': 10,
        'move_bounds': False,
        **sticks_kwargs
    }

    filtered_faults = []

    for fault in faults:
        if (len(fault.points) < 30) or (len(fault) < min_fault_len):
            continue

        fault.points_to_sticks(sticks_step=config_sticks['sticks_step'],
                               stick_nodes_step=config_sticks['stick_nodes_step'],
                               move_bounds=config_sticks['move_bounds'])

        if np.concatenate([item[:, 2] for item in fault.sticks]).ptp() < min_height:
            continue

        filtered_faults.append(fault)

    return filtered_faults


def filter_disconnected_faults(faults, direction=0, height_threshold=200, width_threshold=40, **kwargs):
    """ Filter small enough faults whithout neighbors.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` or :class:`~.FaultPrototype` instances
        Faults for filtering.
    direction : {0, 1}
        Faults direction.
    height_threshold : int
        Filter out faults disconnected with height less than `height_threshold`.
    width_threshold : int
        Filter out faults disconnected with width less than `width_threshold`.
    **kwargs : dict
        Adjacency kwargs for :func:`._group_adjacent_faults`.
    """
    groups, _ = _group_adjacent_faults(faults, **kwargs)

    grouped_faults_indices = set(groups.keys())

    for group_members in groups.values():
        grouped_faults_indices = grouped_faults_indices.union(group_members)

    filtered_faults = []

    for i, fault in enumerate(faults):
        if i in grouped_faults_indices:
            filtered_faults.append(fault)

        else:
            height = fault.bbox[2, 1] - fault.bbox[2, 0]
            width = fault.bbox[direction, 1] - fault.bbox[direction, 0]

            if height > height_threshold or width > width_threshold:
                filtered_faults.append(fault)

    return filtered_faults

def _group_adjacent_faults(faults, adjacency=5, adjacent_points_threshold=5):
    """ Add faults into groups by adjacency criterion.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` or :class:`~.FaultPrototype` instances
        Faults for filtering.
    adjacency : int
        Axis-wise distance between two faults to consider them to be in one group.
    adjacent_points_threshold : int
        Minimal amount of fault points into adjacency area to consider two faults are in one group.
    """
    groups = {} # owner -> items
    owners = {} # item -> owner

    for i, fault_1 in enumerate(faults):
        if i not in owners.keys():
            owners[i] = i

        for j, fault_2 in enumerate(faults[i+1:]):
            adjacent_borders = bboxes_adjacent(fault_1.bbox, fault_2.bbox, adjacency=adjacency)

            if adjacent_borders is None:
                continue

            # Check points amount in the adjacency area
            for fault in (fault_1, fault_2):
                adjacent_points = fault.points[(fault.points[:, 0] >= adjacent_borders[0][0]) & \
                                                (fault.points[:, 0] <= adjacent_borders[0][1]) & \
                                                (fault.points[:, 1] >= adjacent_borders[1][0]) & \
                                                (fault.points[:, 1] <= adjacent_borders[1][1]) & \
                                                (fault.points[:, 2] >= adjacent_borders[2][0]) & \
                                                (fault.points[:, 2] <= adjacent_borders[2][1])]

                if len(adjacent_points) < adjacent_points_threshold:
                    adjacent_borders = None
                    break

            if adjacent_borders is None:
                continue

            owners[i+1+j] = owners[i]

            if owners[i] not in groups.keys():
                groups[owners[i]] = set()

            groups[owners[i]].add(i+1+j)

    return groups, owners


def filter_sticked_faults(faults, direction, sticks_step=10, stick_nodes_step=50):
    """ Filter fault with too small sticks amount and filter edge sticks if needed.

    Optional filtering, which can be sometimes required.
    Note, it is an experimental feature and can be changed in future.

    Parameters
    ----------
    faults : sequence of :class:`~.Fault` instances
        Faults for filtering.
        Note, they must be outputs from :meth:`.filter_faults`.
    direction : {0, 1}
        Fault prediction direction.
    sticks_step : int
        A value of `sticks_step` argument with which faults sticks were created.
    stick_nodes_step : int
        A value of `stick_nodes_step` argument with which faults sticks were created.
    """
    selected_faults = []

    for fault in faults:
        # 2 sticks - not enough
        if len(fault.sticks) <= 2:
            continue

        # Remove too short edges
        # Left edge
        if len(fault.sticks[0]) == 2:
            first_stick_height = fault.sticks[0][1][2] - fault.sticks[0][0][2]
            second_stick_height = fault.sticks[1][1][2] - fault.sticks[1][0][2]

            height_ratio = np.abs(first_stick_height / second_stick_height)

            if height_ratio <= 0.5:
                fault._sticks = fault._sticks[1:]

        if len(fault.sticks) <= 2:
            continue

        # Right edge
        if len(fault.sticks[-1]) == 2:
            last_stick_height = fault.sticks[-1][1][2] - fault.sticks[-1][0][2]
            previous_stick_height = fault.sticks[-2][1][2] - fault.sticks[-2][0][2]

            height_ratio = np.abs(last_stick_height / previous_stick_height)

            if height_ratio <= 0.5:
                fault._sticks = fault._sticks[1:]

        if len(fault.sticks) <= 2:
            continue

        selected_faults.append(fault)

    return selected_faults
