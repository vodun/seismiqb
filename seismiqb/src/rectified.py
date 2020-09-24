""" 2D surface class and metrics. """

import os

import numpy as np

from .horizon import Horizon
from .utils import IndexedDict



class RectifiedAmplitudes:
    """"""
    def __init__(self, horizons, window, offset, scale):
        self.horizons = horizons
        self.window = window
        self.offset = offset
        self.scale = scale
        self._data = None

    def load_crop(self, locations):
        """"""
        x_slice, y_slice, z_slice = locations
        depth_range = [key for key in self.data.keys() if z_slice.start in key]
        if len(depth_range) == 1:
            return self.data[depth_range[0]][x_slice, y_slice]
        elif len(depth_range) > 1:
            raise KeyError(f'More than one amplitudes cutout corresponds to given depth range {z_slice}: {depth_range}')
        else:
            raise KeyError(f'No cutout amplitudes found at given depth')
    @property
    def data(self):
        """"""
        if self._data is None:
            self.load_data()
        return self._data

    def load_data(self):
        self._data = IndexedDict({})
        for horizon in self.horizons:
            amplitudes = horizon.get_cube_values(window=self.window, offset=self.offset, scale=self.scale)
            amplitudes[horizon.full_matrix == horizon.FILL_VALUE] = np.nan
            depth_range = range(horizon._depths[0], horizon._depths[-1])
            self._data[depth_range] = amplitudes

    def reload_data(self, window=None, offset=None, scale=None):
        """"""
        self.window = window if window is not None else self.window
        self.offset = offset if offset is not None else self.offset
        self.scale = scale if scale is not None else self.scale
        self.load_data()

class RectifiedSurface(Horizon):
    """"""
    def add_to_mask(self, mask, locations=None, alpha=1):
        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (_, mask_h) = mask_bbox

        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)

        if i_max >= i_min and x_max >= x_min:
            overlap = self.matrix[i_min - self.i_min:i_max - self.i_min,
                                  x_min - self.x_min:x_max - self.x_min]

            idx_i, idx_x = np.asarray(overlap != self.FILL_VALUE).nonzero()
            heights = overlap[idx_i, idx_x]
            if heights.min() <= mask_h <= heights.max():
                idx_i += i_min - mask_i_min
                idx_x += x_min - mask_x_min
                mask[idx_i, idx_x] = alpha

        return mask
