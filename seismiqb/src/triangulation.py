""" Triangulation functions. """
import numpy as np
from numba import njit

@njit
def shortest_distance(coords, coefs):
    return np.abs(np.sum(coords * coefs[:-1]) + coefs[-1]) / np.sqrt(np.sum(coefs[:-1] ** 2))

@njit
def equation_plane(points):
    pq = points[1] - points[0]
    pr = points[2] - points[0]
    product = np.cross(pq, pr)
    bias = -np.sum(product * points[0])
    return np.array([product[0], product[1], product[2], bias])

@njit
def get_z(coords, coefs):
    return -(coords[0] * coefs[0] + coords[1] * coefs[1] + coefs[3]) / coefs[2]

@njit
def equation_line(points):
    a = points[1][1] - points[0][1]
    b = points[0][0] - points[1][0]
    c = -a*(points[0][0]) - b*(points[0][1])
    return np.array([a, b, c])

@njit
def sign(coords, coefs):
    return (np.sign(np.sum(coords * coefs[:-1]) + coefs[-1]))

@njit
def bound_lines(nodes):
    line_coefs = np.zeros((3, 4))
    indices = np.array([[1, 2], [0, 2], [0, 1]])
    for i in range(3):
        _coefs = equation_line(nodes[indices[i]])
        line_coefs[i] = [_coefs[0], _coefs[1], _coefs[2], sign(nodes[i], _coefs)]
    return line_coefs

@njit
def check_sign(a, b):
    return (a == 0) or (a == b)

@njit
def in_projection(point, coefs):
    return np.all(np.array([check_sign(sign(point, coefs[i, :-1]), coefs[i, -1]) for i in range(3)]))

#@njit
def triangle_rasterization(points, width=1):
    shape = np.array([np.max(points[:, i]) - np.min(points[:, i]) for i in range(3)])
    order = np.argsort(shape)[::-1]
    points = points[:, order]
    shape = shape[order]

    bounds = np.array([[np.min(points[:, i]), np.max(points[:, i])] for i in range(3)]).T
    degenerate_coords = np.sum(shape == 0)

    # if degenerate_coords == 3:
    #     _points = bounds[1:]
    # elif degenerate_coords == 2:
    #     _points = np.zeros((2, 3))
    #     _points[:, 0] = np.arange(bounds[0][0], bounds[1][0]+1)
    #     # _points[:, 1:] = bounds[0, 1:]
    # else:
    coefs = equation_plane(points)

    line_coefs = bound_lines(points[:, :-1])
    _points = np.zeros((int((shape[0]+1) * (shape[1]+1) * 4 * width), 3))
    i = 0
    for x in range(int(bounds[0][0]), int(bounds[1][0]+1)):
        for y in range(int(bounds[0][1]), int(bounds[1][1]+1)):
            center = int(get_z(np.array([x, y]), coefs))
            for z in range(center - 2 * width, center + 2 * width):
                node = np.array([x, y, z])
                if in_projection(node[:-1], line_coefs) and shortest_distance(node, coefs) < width * np.sqrt(3) / 2:
                    _points[i] = node
                    i += 1
    reverse = np.arange(len(order))[np.argsort(order)]
    return _points[:i, reverse]

def triangulation(points):
    triangles = []
    for s1, s2 in zip(points[:-1], points[1:]):
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        n = len(s1)
        nodes = [item for sublist in zip(s1, s2[:n+1]) for item in sublist]
        nodes = [nodes[i:i+3] for i in range(len(nodes[:-3])+1)]
        nodes += [[s1[-1], s2[i], s2[i+1]] for i in range(n-1, len(s2)-1)]
        triangles += nodes
    return np.array(triangles)
