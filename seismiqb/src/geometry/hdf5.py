import os
import re
import sys
import shutil
import itertools

from textwrap import dedent
from random import random
from itertools import product
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import h5py
import segyio
import cv2

from ..utils import find_min_max, file_print, compute_attribute, make_axis_grid, \
                   fill_defaults, parse_axis, get_environ_flag
from ..utility_classes import lru_cache, SafeIO
from ..plotters import plot_image


from .base import SeismicGeometry


class SeismicGeometryHDF5(SeismicGeometry):
    """ Class to infer information about HDF5 cubes and provide convenient methods of working with them.

    In order to initialize instance, one must supply `path` to the HDF5 cube.

    All the attributes are loaded directly from HDF5 file itself, so most of the attributes from SEG-Y file
    are preserved, with the exception of `dataframe` and `uniques`.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, path, **kwargs):
        self.structured = True
        self.file_hdf5 = None

        super().__init__(path, **kwargs)

    def process(self, **kwargs):
        """ Put info from `.hdf5` groups to attributes.
        No passing through data whatsoever.
        """
        _ = kwargs
        self.file_hdf5 = h5py.File(self.path, mode='r')
        self.add_attributes()

    def add_attributes(self):
        """ Store values from `hdf5` file to attributes. """
        self.index_headers = self.INDEX_POST
        self.load_meta()
        if hasattr(self, 'lens'):
            self.cube_shape = np.asarray([self.ilines_len, self.xlines_len, self.depth]) # BC
        else:
            self.cube_shape = self.file_hdf5['cube'].shape
            self.lens = self.cube_shape
        self.has_stats = True


    # Methods to load actual data from HDF5
    def load_crop(self, locations, axis=None, **kwargs):
        """ Load 3D crop from the cube.
        Automatically chooses the fastest axis to use: as `hdf5` files store multiple copies of data with
        various orientations, some axis are faster than others depending on exact crop location and size.

        Parameters
        locations : sequence of slices
            Location to load: slices along the first index, the second, and depth.
        axis : str or int
            Identificator of the axis to use to load data.
            Can be `iline`, `xline`, `height`, `depth`, `i`, `x`, `h`, 0, 1, 2.
        """
        if axis is None:
            shape = np.array([(slc.stop - slc.start) for slc in locations])
            axis = np.argmin(shape)
        else:
            mapping = {0: 0, 1: 1, 2: 2,
                       'i': 0, 'x': 1, 'h': 2,
                       'iline': 0, 'xline': 1, 'height': 2, 'depth': 2}
            axis = mapping[axis]

        if axis == 1 and 'cube_x' in self.file_hdf5:
            crop = self._load_x(*locations, **kwargs)
        elif axis == 2 and 'cube_h' in self.file_hdf5:
            crop = self._load_h(*locations, **kwargs)
        else: # backward compatibility
            crop = self._load_i(*locations, **kwargs)
        return crop

    def _load_i(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.file_hdf5['cube']
        return np.stack([self._cached_load(cube_hdf5, iline, **kwargs)[xlines, :][:, heights]
                         for iline in range(ilines.start, ilines.stop)])

    def _load_x(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.file_hdf5['cube_x']
        return np.stack([self._cached_load(cube_hdf5, xline, **kwargs)[heights, :][:, ilines].transpose([1, 0])
                         for xline in range(xlines.start, xlines.stop)], axis=1)

    def _load_h(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.file_hdf5['cube_h']
        return np.stack([self._cached_load(cube_hdf5, height, **kwargs)[ilines, :][:, xlines]
                         for height in range(heights.start, heights.stop)], axis=2)

    @lru_cache(128)
    def _cached_load(self, cube, loc, **kwargs):
        """ Load one slide of data from a certain cube projection.
        Caches the result in a thread-safe manner.
        """
        _ = kwargs
        return cube[loc, :, :]

    def load_slide(self, loc, axis='iline', **kwargs):
        """ Load desired slide along desired axis. """
        axis = parse_axis(axis)

        if axis == 0:
            cube = self.file_hdf5['cube']
            slide = self._cached_load(cube, loc, **kwargs)
        elif axis == 1:
            cube = self.file_hdf5['cube_x']
            slide = self._cached_load(cube, loc, **kwargs).T
        elif axis == 2:
            cube = self.file_hdf5['cube_h']
            slide = self._cached_load(cube, loc, **kwargs)
        return slide

    def __getitem__(self, key):
        """ Retrieve amplitudes from cube. Uses the usual `Numpy` semantics for indexing 3D array. """
        key_ = list(key)
        if len(key_) != len(self.cube_shape):
            key_ += [slice(None)] * (len(self.cube_shape) - len(key_))

        key, squeeze = [], []
        for i, item in enumerate(key_):
            max_size = self.cube_shape[i]

            if isinstance(item, slice):
                slc = slice(item.start or 0, item.stop or max_size)
            elif isinstance(item, int):
                item = item if item >= 0 else max_size - item
                slc = slice(item, item + 1)
                squeeze.append(i)
            key.append(slc)

        shape = [(slc.stop - slc.start) for slc in key]
        axis = np.argmin(shape)
        if axis == 0:
            crop = self.file_hdf5['cube'][key[0], key[1], key[2]]
        elif axis == 1:
            crop = self.file_hdf5['cube_x'][key[1], key[2], key[0]].transpose((2, 0, 1))
        elif axis == 2:
            crop = self.file_hdf5['cube_h'][key[2], key[0], key[1]].transpose((1, 2, 0))

        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop
