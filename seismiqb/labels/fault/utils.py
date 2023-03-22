""" Utils for set of faults, e.g. filtering. """
import numpy as np


def filter_faults(faults, min_fault_len=2000, min_height=20, **sticks_kwargs):
    """ Filter too small faults.

    Faults are filtered by amount of points, length and height.

    Parameters
    ----------
    faults : sequence of `~.Fault` instances
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


def filter_sticked_faults(faults, direction, sticks_step=10, stick_nodes_step=50):
    """ Filter fault with too small sticks amount and filter edge sticks if needed.

    Optional filtering, which can be sometimes required.
    Note, it is an experimental feature and can be changed in future.

    Parameters
    ----------
    faults : sequence of `~.Fault` instances
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

        # Sticks should be on multiples of `sticks_step` traces - remove extra edges
        first_stick_line = fault.sticks[0][0][direction]

        if first_stick_line % sticks_step != 0:
            fault._sticks = fault._sticks[1:]

        last_stick_line = fault.sticks[-1][0][direction]

        if last_stick_line % sticks_step != 0:
            fault._sticks = fault._sticks[:-1]

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
