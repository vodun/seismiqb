""" Functions to get line coordinates by two points. """

import numpy as np
from numba import njit

@njit
def _iline_to_xline(x, loc, angle):
    """ Crossline as a function of inline through `loc` with `angle`. """
    return np.tan(angle) * (x - loc[0]) + loc[1]

@njit
def _xline_to_iline(y, loc, angle):
    """ Inline as a function of crossline through `loc` with `angle`. """
    return np.tan(np.pi / 2 - angle) * (y - loc[1]) + loc[0]

@njit
def extend_line(loc_a, loc_b, shape):
    """ Get bound points for line through `loc_a` and `loc_b`. """
    max_i, max_x = shape
    direction = loc_b - loc_a
    distance = np.power(direction, 2).sum() ** 0.5
    if direction[0] != 0:
        angle = np.arctan(direction[1] / direction[0])
    else:
        angle = np.pi
    pos = loc_a

    xline_left = _iline_to_xline(0, pos, angle)
    if xline_left < 0:
        start = (_xline_to_iline(0, pos, angle), 0)
    elif xline_left > max_x:
        start = (_xline_to_iline(max_x-1, pos, angle), max_x-1)
    else:
        start = (0, xline_left)

    xline_right = _iline_to_xline(max_i-1, pos, angle)
    if xline_right < 0:
        end = (_xline_to_iline(0, pos, angle), 0)
    elif xline_right > max_x:
        end = (_xline_to_iline(max_x-1, pos, angle), max_x-1)
    else:
        end = (max_i-1, xline_right)

    return np.array(start), np.array(end)
