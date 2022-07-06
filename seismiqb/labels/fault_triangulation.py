""" Triangulation functions. """
import numpy as np
from numba import njit



@njit
def triangle_rasterization(points, width=1):
    """ Transform triangle to surface of the fixed thickness.

    Parameters
    ----------
    points : numpy.ndarray
        array of size 3 x 3: each row is a vertex of triangle
    width : int
        thicc

    Return
    ------
    numpy.ndarray
        array of size N x 3 where N is a number of points in rasterization.
    """
    max_n_points = np.int32(triangle_volume(points, width))
    _points = np.empty((max_n_points, 3))
    i = 0
    r_margin = width - width // 2
    l_margin = width // 2
    for x in range(int(np.min(points[:, 0]))-l_margin, int(np.max(points[:, 0]))+r_margin): # pylint: disable=not-an-iterable
        for y in range(int(np.min(points[:, 1]))-l_margin, int(np.max(points[:, 1])+r_margin)):
            for z in range(int(np.min(points[:, 2]))-l_margin, int(np.max(points[:, 2]))+r_margin):
                node = np.array([x, y, z])
                if distance_to_triangle(points, node) <= width / 2:
                    _points[i] = node
                    i += 1
    return _points[:i]

@njit
def triangle_volume(points, width):
    """ Compute triangle volume to estimate the number of points. """
    a = points[0] - points[1]
    a = np.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)

    b = points[0] - points[2]
    b = np.sqrt(b[0] ** 2 + b[1] ** 2 + b[2] ** 2)

    c = points[2] - points[1]
    c = np.sqrt(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)

    p = (a + b + c) / 2
    S = (p * (p - a) * (p - b) * (p - c)) ** 0.5
    r = S / p
    r_ = (r + width)
    p_ = p * r_ / r
    return (p_ * r_) * (width + 1)


def sticks_to_simplices(sticks, return_indices=False):
    """ Compute triangulation of the fault.

    Parameters
    ----------
    sticks : numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    return_indices : bool
        If True, function will return indices of stick nodes in flatten array.

    Return
    ------
    numpy.ndarray
        numpy.ndarray of length N where N is the number of simplices. Each item is a sequence of coordinates of each
        vertex (if `return_indices=False`) or indices of nodes in initial flatten array.
    """
    simplices = []
    nodes = np.concatenate(sticks)

    if return_indices:
        n_nodes = np.cumsum([0, *[len(item) for item in sticks]])
        sticks = np.array([np.arange(len(sticks[i])) + n_nodes[i] for i in range(len(sticks))], dtype=object)
    for s1, s2 in zip(sticks[:-1], sticks[1:]):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        n = len(s1)
        nodes_to_connect = [item for sublist in zip(s1, s2[:n]) for item in sublist]
        if len(nodes_to_connect) > 2:
            triangles = [nodes_to_connect[i:i+3] for i in range(len(nodes_to_connect[:-2]))]
        else:
            triangles = []
        triangles += [[s1[-1], s2[i], s2[i+1]] for i in range(n-1, len(s2)-1)]
        simplices += triangles
    return np.array(simplices), nodes

def simplices_to_points(simplices, nodes, width=1):
    """ Interpolate triangulation.

    Parameters
    ----------
    simplices : numpy.ndarray
        Array of shape (n_simplices, 3) with indices of nodes to connect into triangle.
    nodes : numpy.ndarray
        Array of shape (n_nodes, 3) with coordinates.
    width : int, optional
        Thickness of the simplex to draw, by default 1.

    Returns
    -------
    numpy.ndarray
        Array of shape (n_points, 3)
    """
    points = []
    for triangle in simplices:
        points.append(triangle_rasterization(nodes[triangle].astype('float32'), width))
    return np.concatenate(points, axis=0).astype('int32')

@njit
def distance_to_triangle(triangle, node):
    """ https://gist.github.com/joshuashaffer/99d58e4ccbd37ca5d96e """
    # pylint: disable=invalid-name, too-many-nested-blocks, too-many-branches, too-many-statements
    B = triangle[0, :]
    E0 = triangle[1, :] - B
    E1 = triangle[2, :] - B
    D = B - node
    a = np.dot(E0, E0)
    b = np.dot(E0, E1)
    c = np.dot(E1, E1)
    d = np.dot(E0, D)
    e = np.dot(E1, D)
    f = np.dot(D, D)

    det = a * c - b * b
    s = b * e - c * d
    t = b * d - a * e

    if det == 0:
        return 0.

    # Terrible tree of conditionals to determine in which region of the diagram
    # shown above the projection of the point into the triangle-plane lies.
    if (s + t) <= det:
        if s < 0.0:
            if t < 0.0:
                # region4
                if d < 0:
                    t = 0.0
                    if -d >= a:
                        s = 1.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
                else:
                    s = 0.0
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        if -e >= c:
                            t = 1.0
                            sqrdistance = c + 2.0 * e + f
                        else:
                            t = -e / c
                            sqrdistance = e * t + f

                            # of region 4
            else:
                # region 3
                s = 0
                if e >= 0:
                    t = 0
                    sqrdistance = f
                else:
                    if -e >= c:
                        t = 1
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 3
        else:
            if t < 0:
                # region 5
                t = 0
                if d >= 0:
                    s = 0
                    sqrdistance = f
                else:
                    if -d >= a:
                        s = 1
                        sqrdistance = a + 2.0 * d + f  # GF 20101013 fixed typo d*s ->2*d
                    else:
                        s = -d / a
                        sqrdistance = d * s + f
            else:
                # region 0
                invDet = 1.0 / det
                s = s * invDet
                t = t * invDet
                sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f
    else:
        if s < 0.0:
            # region 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:  # minimum on edge s+t=1
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                if numer >= denom:
                    s = 1.0
                    t = 0.0
                    sqrdistance = a + 2.0 * d + f  # GF 20101014 fixed typo 2*b -> 2*d
                else:
                    s = numer / denom
                    t = 1 - s
                    sqrdistance = s * (a * s + b * t + 2 * d) + t * (b * s + c * t + 2 * e) + f

            else:  # minimum on edge s=0
                s = 0.0
                if tmp1 <= 0.0:
                    t = 1
                    sqrdistance = c + 2.0 * e + f
                else:
                    if e >= 0.0:
                        t = 0.0
                        sqrdistance = f
                    else:
                        t = -e / c
                        sqrdistance = e * t + f
                        # of region 2
        else:
            if t < 0.0:
                # region6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        t = 1.0
                        s = 0
                        sqrdistance = c + 2.0 * e + f
                    else:
                        t = numer / denom
                        s = 1 - t
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

                else:
                    t = 0.0
                    if tmp1 <= 0.0:
                        s = 1
                        sqrdistance = a + 2.0 * d + f
                    else:
                        if d >= 0.0:
                            s = 0.0
                            sqrdistance = f
                        else:
                            s = -d / a
                            sqrdistance = d * s + f
            else:
                # region 1
                numer = c + e - b - d
                if numer <= 0:
                    s = 0.0
                    t = 1.0
                    sqrdistance = c + 2.0 * e + f
                else:
                    denom = a - 2.0 * b + c
                    if numer >= denom:
                        s = 1.0
                        t = 0.0
                        sqrdistance = a + 2.0 * d + f
                    else:
                        s = numer / denom
                        t = 1 - s
                        sqrdistance = s * (a * s + b * t + 2.0 * d) + t * (b * s + c * t + 2.0 * e) + f

    # account for numerical round-off error
    sqrdistance = max(sqrdistance, 0)
    dist = np.sqrt(sqrdistance)
    return dist
