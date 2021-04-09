""" NPZ geometry. """
import sys

import numpy as np

from .base import SeismicGeometry



class SeismicGeometryNPZ(SeismicGeometry):
    """ Create a Geometry instance from a `numpy`-saved file. Stores everything in memory.
    Can simultaneously work with multiple type of cube attributes, e.g. amplitudes, GLCM, RMS, etc.
    """
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, path, **kwargs):
        self.structured = True
        self.file_npz = None
        self.names = None
        self.data = {}

        super().__init__(path, **kwargs)

    def process(self, order=(0, 1, 2), **kwargs):
        """ Create all the missing attributes. """
        self.index_headers = SeismicGeometry.INDEX_POST
        self.file_npz = np.load(self.path, allow_pickle=True, mmap_mode='r')

        self.names = list(self.file_npz.keys())
        self.data = {key : np.transpose(self.file_npz[key], order) for key in self.names}

        data = self.data[self.names[0]]
        self.cube_shape = np.array(data.shape)
        self.lens = self.cube_shape[:2]
        self.zero_traces = np.zeros(self.lens)

        # Attributes
        self.depth = self.cube_shape[2]
        self.delay, self.sample_rate = 0, 0
        self.value_min = np.min(data)
        self.value_max = np.max(data)
        self.q001, self.q01, self.q99, self.q999 = np.quantile(data, [0.001, 0.01, 0.99, 0.999])


    # Methods to load actual data from NPZ
    def load_crop(self, locations, names=None, **kwargs):
        """ Load 3D crop from the cube.

        Parameters
        locations : sequence of slices
            Location to load: slices along the first index, the second, and depth.
        names : sequence
            Names of data attributes to load.
        """
        _ = kwargs
        names = names or self.names[:1]
        shape = np.array([(slc.stop - slc.start) for slc in locations])
        axis = np.argmin(shape)

        crops = [self.data[key][locations[0], locations[1], locations[2]] for key in names]
        crop = np.concatenate(crops, axis=axis)
        return crop

    def load_slide(self, loc, axis='iline', **kwargs):
        """ Load desired slide along desired axis. """
        _ = kwargs
        locations = self.make_slide_locations(loc, axis)
        crop = self.load_crop(locations, names=['data'])
        return crop.squeeze()

    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        return sum(sys.getsizeof(self.data[key]) for key in self.names)

    def __getattr__(self, key):
        """ Use default `object` getattr, without `.meta` magic. """
        return object.__getattribute__(self, key)

    def __getitem__(self, key):
        """ Get data from the first named array. """
        return self.data[self.names[0]][key]
