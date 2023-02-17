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
