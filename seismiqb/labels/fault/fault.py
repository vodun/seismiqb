""" Fault class and processing methods. """

import os
import numpy as np
from numba import prange, njit

from .fault_triangulation import sticks_to_simplices, triangle_rasterization
from .fault_approximation import points_to_sticks
from .fault_visualization import FaultVisualizationMixin
from ..mixins import CoordinatesMixin
from .fault_formats import FaultSticksMixin, FaultSerializationMixin


class Fault(FaultSticksMixin, FaultSerializationMixin, CoordinatesMixin, FaultVisualizationMixin):
    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    def __init__(self, storage, field, name=None, direction=None, **kwargs):
        self.name = name
        self.field = field

        self._points = None
        self._sticks = None
        self._nodes = None
        self._simplices = None
        self.direction = None

        if isinstance(storage, str):
            format = 'file'
        elif isinstance(storage, np.ndarray):
            format = 'points'
        elif isinstance(storage, dict):
            format = 'objects'
        getattr(self, f'from_{format}')(storage, **kwargs)

        # if self.direction is None:
        self.set_direction(direction)
        self.create_stats()

    def filter(self):
        pass

    def has_component(self, component):
        return getattr(self, '_'+component) is not None

    def create_stats(self):
        if self.has_component('points'):
            data = self.points
        elif self.has_component('nodes'):
            data = self.nodes
        else:
            data = np.concatenate(self.sticks)

        i_min, x_min, h_min = np.min(data, axis=0)
        i_max, x_max, h_max = np.max(data, axis=0)

        self.h_min, self.h_max = int(h_min), int(h_max)
        self.i_min, self.i_max, self.x_min, self.x_max = int(i_min), int(i_max), int(x_min), int(x_max)

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                            [self.x_min, self.x_max],
                            [self.h_min, self.h_max]],
                            dtype=np.int32)

    def set_direction(self, direction):
        """ Find azimuth of the fault. """
        if direction is None:
            if self._sticks is not None and len(self._sticks) > 0:
                if len(np.unique(self._sticks[0][:, 0])) == 1:
                    self.direction = 0
                elif len(np.unique(self._sticks[0][:, 1])) == 1:
                    self.direction = 1
            if self.direction is None and len(self.points) > 0:
                mean_depth = int(np.median(self.points[:, 2]))
                depth_slice = self.points[self.points[:, 2] == mean_depth]
                self.direction = 0 if depth_slice[:, 0].ptp() > depth_slice[:, 1].ptp() else 1
            else:
                self.direction = 0
            # TODO: azimuth from charisma
        elif isinstance(direction, int):
            self.direction = direction
        elif isinstance(direction[self.field.short_name], int):
            self.direction = direction[self.field.short_name]
        else:
            self.direction = direction[self.field.short_name][self.name]

    def points_to_sticks(self, slices=None, sticks_step=10, stick_nodes_step=10):
        """ Create sticks from fault points. """
        points = self.points.copy()
        if slices is not None:
            for i in range(3):
                points = points[points[:, i] <= slices[i].stop]
                points = points[points[:, i] >= slices[i].start]
        if len(points) <= 3:
            self._sticks = []
        else:
            self._sticks = points_to_sticks(points, sticks_step, stick_nodes_step, self.direction)

    def from_file(self, path, **kwargs):
        """ Init from path to either CHARISMA, REDUCED_CHARISMA or FAULT_STICKS csv-like file
        or from npy/npz.
        """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = os.path.basename(path)
        self.short_name = os.path.splitext(self.name)[0]

        ext = os.path.splitext(path)[1][1:]

        if ext == 'npz':
            self.load_npz(path)
            self.format = 'file-npz'
        elif ext == 'npy':
            self.load_npy(path)
            self.format = 'file-npy'
        else:
            self.load_fault_sticks(path, **kwargs)
            self.format = 'file-sticks'

    def from_objects(self, storage, **kwargs):
        """ Load fault from dict with 'points', 'nodes', 'simplices' and 'sticks'. """
        for key in ['points', 'sticks', 'nodes', 'simplices']:
            setattr(self, '_' + key, storage.get(key))

    # Transformation of attributes: sticks -> (nodes, simplices) -> points -> sticks

    @property
    def simplices(self):
        if self._simplices is None:
            if self._points is None and self._sticks is None:
                raise AttributeError("`simplices` can't be created.")

            self._simplices, self._nodes = sticks_to_simplices(self.sticks, return_indices=True)

        return self._simplices

    @property
    def nodes(self):
        if self._nodes is None:
            if self._points is None and self._sticks is None:
                raise AttributeError("`nodes` can't be created.")

            self._simplices, self._nodes = sticks_to_simplices(self.sticks, return_indices=True)

        return self._nodes

    @property
    def points(self):
        if self._points is None:
            if self._simplices is None and self._sticks is None:
                raise AttributeError("`points` can't be created.")
            if len(self.simplices) > 1:
                self.simplices_to_points()
            elif len(self.nodes) > 0:
                fake_fault = get_fake_one_stick_fault(self)
                points = fake_fault.points
                self._points = points[points[:, self.direction] == self.sticks[0][0, self.direction]]

        return self._points

    @property
    def sticks(self):
        if self._sticks is None:
            if self._simplices is None and self._points is None:
                raise AttributeError("`points` can't be created.")
            self.points_to_sticks()

        return self._sticks

    def simplices_to_points(self, width=1):
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
        for triangle in self.simplices:
            points.append(triangle_rasterization(self.nodes[triangle].astype('float32'), width))
        self._points = np.concatenate(points, axis=0).astype('int32')

    def add_to_mask(self, mask, locations=None, width=1, **kwargs):
        """ Add fault to background. """
        _ = kwargs
        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                            [locations[1].start, locations[1].stop],
                            [locations[2].start, locations[2].stop]],
                            dtype=np.int32)
        points = self.points

        if (self.bbox[:, 1] < mask_bbox[:, 0]).any() or (self.bbox[:, 0] >= mask_bbox[:, 1]).any():
            return mask

        insert_fault_into_mask(mask, points, mask_bbox, width=width, axis=1-self.direction)
        return mask

    def __len__(self):
        """ Number of labeled traces. """
        # TODO
        return np.prod(self.bbox[:, 1] - self.bbox[:, 0] + 1)

@njit(parallel=True)
def insert_fault_into_mask(mask, points, mask_bbox, width, axis):
    """ Add new points into binary mask. """
    #pylint: disable=not-an-iterable

    for i in prange(len(points)):
        point = points[i]
        if (point[0] >= mask_bbox[0][0]) and (point[0] < mask_bbox[0][1]):
            if (point[1] >= mask_bbox[1][0]) and (point[1] < mask_bbox[1][1]):
                if (point[2] >= mask_bbox[2][0]) and (point[2] < mask_bbox[2][1]):
                    slices = [slice(point[j] - mask_bbox[j][0], point[j] - mask_bbox[j][0]+1) for j in range(3)]
                    if width > 1:
                        slices[axis] = slice(
                            max(0, point[axis] - mask_bbox[axis][0] - (width // 2)),
                            min(point[axis] - mask_bbox[axis][0] + width // 2 + width % 2, mask.shape[axis])
                        )
                    mask[slices[0], slices[1], slices[2]] = 1

def get_fake_one_stick_fault(fault):
    if len(set(fault.sticks[0][:, 0])) > 1 and len(set(fault.sticks[0][:, 1])) > 1:
        raise ValueError('!!')
    stick = fault.sticks[0]
    stick_2 = stick.copy()
    loc = stick[0, fault.direction]
    stick_2[:, fault.direction] = loc - 1 if loc >= 1 else loc + 1

    fake_fault = Fault({'sticks': np.array([stick, stick_2])}, direction=fault.direction,
                       field=fault.field)

    return fake_fault
