""" Geometry stored in memory """
import os

import numpy as np
import h5pickle as h5py

from .hdf5 import SeismicGeometryHDF5


class SeismicGeometryArray(SeismicGeometryHDF5):
    """ Numpy array stored in memory as a  SeismicGeometry"""
    #pylint: disable=attribute-defined-outside-init
    def process(self, arrays, **kwargs):
        self.array = arrays[self.path]

        self.available_axis = [0]
        self.available_names = ['cube_i']

        self.axis_to_cube = {0: self.array}
        setattr(self, 'cube_i', self.array)

        self.add_attributes(**kwargs)

        if not self.has_stats:
            self.v_q01, self.v_q99 = np.quantile(self.array, [0.01, 0.99])
            self.zero_traces = np.zeros(self.cube_shape[:2], dtype=np.int32)
            self.zero_traces[np.std(self.array, axis=2) == 0] = 1
            self.has_stats = True
            self.store_meta()
