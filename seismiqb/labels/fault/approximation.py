""" Approximation utilities to convert cloud of points to sticks. """

import numpy as np
from sklearn.decomposition import PCA
import cv2
from numba import njit

from .postprocessing import thin_line, split_array


def points_to_sticks(points, sticks_step=10, nodes_step=10, fault_orientation=None, stick_orientation=2):
    """ Get sticks from fault which is represented as a cloud of points.

    Parameters
    ----------
    points : np.ndarray
        Fault points.
    sticks_step : int
        Number of slides between sticks.
    nodes_step : int
        Maximal distance between stick nodes
    stick_orientation : int (0, 1 or 2)
        Direction of each stick

    Returns
    -------
    numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    """
    if fault_orientation is None:
        pca = PCA(1)
        pca.fit(points)
        fault_orientation = 0 if np.abs(pca.components_[0][0]) > np.abs(pca.components_[0][1]) else 1

    if stick_orientation != 2:
        fault_orientation = 2

    points = points[np.argsort(points[:, fault_orientation])]
    slides = split_array(points, points[:, fault_orientation])

    sticks = []

    indices = np.arange(0, len(slides), sticks_step)
    if len(slides) - 1 not in indices:
        indices = list(indices) + [len(slides)-1]
    for slide_points in np.array(slides)[indices]:
        slide_points = slide_points[np.argsort(slide_points[:, stick_orientation])]
        slide_points = thin_line(slide_points, stick_orientation)
        if len(slide_points) > 2:
            nodes = _get_stick_nodes(slide_points, fault_orientation, stick_orientation, nodes_step).astype('float32')
            nodes = _add_points_to_stick(nodes, nodes_step, fault_orientation, stick_orientation).astype('int32')
        else:
            nodes = slide_points
        if len(nodes) > 0:
            sticks.append(nodes)
    return sticks


def _get_stick_nodes(points, fault_orientation, stick_orientation, threshold=5):
    """ Get sticks from the line (with some width) defined by cloud of points

    Parameters
    ----------
    points : numpy.ndarray
        3D points located on one 2D slide
    fault_orientation : int (0, 1 or 2)
        Direction of the fault
    stick_orientation : int (0, 1 or 2)
        Direction of each stick
    threshold : int, optional
        Threshold to remove nodes which are too close, by default 5

    Returns
    -------
    numpy.ndarray
        Stick nodes
    """
    if len(points) <= 2:
        return points

    normal = 3 - fault_orientation - stick_orientation

    mask = np.zeros(points.ptp(axis=0)[[normal, stick_orientation]] + 1)
    mask[
        points[:, normal] - points[:, normal].min(),
        points[:, stick_orientation] - points[:, stick_orientation].min()
    ] = 1

    line_threshold = cv2.threshold(mask.astype(np.uint8) * 255, 127, 255, 0)[1]
    line_contours = cv2.findContours(line_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)[0]
    nodes = np.unique(np.squeeze(np.concatenate(line_contours)), axis=0) #TODO: unique?

    new_points = np.zeros((len(nodes), 3))
    new_points[:, fault_orientation] = points[0, fault_orientation]
    new_points[:, stick_orientation] = nodes[:, 0] + points[:, stick_orientation].min()
    new_points[:, normal] = nodes[:, 1] + points[:, normal].min()
    new_points = new_points[np.argsort(new_points[:, stick_orientation])]

    if threshold > 0:
        # Remove nodes which are too close
        mask = np.concatenate([[True], np.abs(new_points[2:] - new_points[1:-1]).sum(axis=1) > threshold, [True]])
        new_points = new_points[mask]
    return new_points

@njit
def _add_points_to_stick(sticks, step, fault_orientation, stick_orientation=2):
    """ Add points between nodes which are too far. """
    normal = 3 - fault_orientation - stick_orientation
    ptp = int(sticks[-1][stick_orientation]) - int(sticks[0][stick_orientation]) + 1
    new_sticks = np.zeros((ptp, 3), dtype='float32')

    pos = 0
    for i in range(len(sticks)-1):
        p1, p2 = sticks[i], sticks[i + 1]
        diff = p2[stick_orientation] - p1[stick_orientation]
        if diff > step:
            x = np.arange(p1[stick_orientation], p2[stick_orientation], step) # make more accurate
            y = ((x - p1[stick_orientation]) * p2[normal] + (p2[stick_orientation] - x) * p1[normal]) / diff
            additional_points = np.empty((len(x), 3), dtype='float32')
            additional_points[:, normal] = y
            additional_points[:, fault_orientation] = p1[fault_orientation]
            additional_points[:, stick_orientation] = x
        else:
            additional_points = p1.reshape(1, 3)
        new_sticks[pos:pos+len(additional_points)] = additional_points
        pos += len(additional_points)

    new_sticks[pos] = p2

    return new_sticks[:pos+1]
