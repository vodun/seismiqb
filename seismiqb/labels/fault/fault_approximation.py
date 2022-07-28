""" Approximation utilities to convert cloud of points to sticks. """

import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from .fault_postprocessing import thin_line


def points_to_sticks(points, sticks_step=10, stick_nodes_step=10, axis=None):
    """ Get sticks from fault which is represented as a cloud of points.

    Parameters
    ----------
    points : np.ndarray
        Fault points.
    sticks_step : int
        Number of slides between sticks.
    stick_nodes_step : int
        Distance between stick nodes

    Returns
    -------
    numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    """
    if axis is None:
        pca = PCA(1)
        pca.fit(points)
        axis = 0 if np.abs(pca.components_[0][0]) > np.abs(pca.components_[0][1]) else 1

    points = points[np.argsort(points[:, axis])]
    projections = np.split(points, np.unique(points[:, axis], return_index=True)[1][1:])
    projections = [item for item in projections if item[:, 2].max() - item[:, 2].min() > 5]
    step = min(sticks_step, len(projections)-1)
    if step == 0:
        return []
    projections = projections[::step]
    res = []

    for p in projections:
        p = p[np.argsort(p[:, -1])]
        points_ = thin_line(p, axis=-1).astype(int)
        loc = p[0, axis]
        nodes = approximate_points(points_[:, [1-axis, 2]], stick_nodes_step)
        nodes_ = np.zeros((len(nodes), 3))
        nodes_[:, [1-axis, 2]] = nodes
        nodes_[:, axis] = loc
        res += [nodes_]
    return res

def approximate_points(points, n_points):
    """ Approximate points by stick. """
    pca = PCA(1)
    array = pca.fit_transform(points)
    step = n_points
    initial = np.arange(array.min(), array.max() + step / 2, step)
    indices = np.unique(nearest_neighbors(initial.reshape(-1, 1), array.reshape(-1, 1), 1))
    return points[indices]

def nearest_neighbors(values, all_values, n_neighbors=10):
    """ Find nearest neighbours for each `value` items in `all_values`. """
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(all_values)
    return nn.kneighbors(values)[1].flatten()
