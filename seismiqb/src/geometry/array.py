""" Geometry stored in memory """
import tempfile


import numpy as np

from .hdf5 import SeismicGeometryHDF5
from .base import SeismicGeometry


class DummyFile:
    """ Object that allows creating a SeismicDataset from an aray in memory.
        creates a temporary dummy file to be used in index and links it to the array in memory

        Parameters
        ----------
        data : np. array
            seismic cube stored in a numpy array
    """
    SUFFIX = '.' + SeismicGeometry.ARRAY_ALIASES[0]

    def __init__(self, data):
        #pylint: disable=consider-using-with
        self.data =  data
        self.__file = tempfile.NamedTemporaryFile(suffix=self.SUFFIX)
        self.path = self.__file.name

    def close(self):
        """ closes (and deletes) underlying temporary file """
        if self.__file:
            try:
                self.__file.close()
            finally:
                self.__file = None


class SeismicGeometryArray(SeismicGeometryHDF5):
    """ Numpy array stored in memory as a  SeismicGeometry"""
    #pylint: disable=attribute-defined-outside-init, access-member-before-definition
    def process(self, array, num_keep=10000, **kwargs):
        """ Store references to data array. """
        self.array = array

        self.available_axis = [0]
        self.available_names = ['cube_i']

        self.axis_to_cube = {0: self.array}
        setattr(self, 'cube_i', self.array)

        self.add_attributes(**kwargs)

        if not self.has_stats:
            self.zero_traces = np.zeros(self.cube_shape[:2], dtype=np.int8)
            self.zero_traces[(np.min(self.array, axis=2) == 0) & (np.max(self.array, axis=2) == 0)] = 1

            nonzero_indices = np.nonzero(self.zero_traces == 0)
            step_traces = max(1, len(nonzero_indices[0]) // num_keep)
            traces = self.array[nonzero_indices[0][::step_traces], nonzero_indices[1][::step_traces]]

            self.v_q01, self.v_q99 = np.quantile(traces, [0.01, 0.99])
            self.v_mean, self.v_std = np.nan, np.nan
            self.v_min, self.v_max = np.nan, np.nan
            self.v_uniques = np.nan

            self.has_stats = True

        # Placeholders for some stats
        self.area = -1.
        self.segy_path = '_dummypath'
        self.quantized = False

    @property
    def file_size(self):
        """ Get size of a stored array in gygabytes. """
        return round(self.array.nbytes / (1024**3), 3)

    @property
    def nbytes(self):
        """ Get size of stored array in bytes. """
        return - (1024 ** 3)
