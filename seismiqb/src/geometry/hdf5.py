""" HDF5 geometry. """
import os

import numpy as np
import h5pickle as h5py

from .converted import SeismicGeometryConverted


class SeismicGeometryHDF5(SeismicGeometryConverted):
    """ Infer or create an `HDF5` file with multiple projections of the same data inside. """
    #pylint: disable=attribute-defined-outside-init
    def process(self, mode='r', projections='ixh', shape=None, **kwargs):
        """ Detect available projections in the cube and store handlers to them in attributes. """
        if mode == 'a':
            mode = 'r+' if os.path.exists(self.path) else 'w-'
        self.mode = mode

        if self.mode in ['r', 'r+']:
            self.file = h5py.File(self.path, mode=mode)

        elif self.mode=='w-':
            # TODO Create new HDF5 file with required projections
            pass

        # Check available projections
        self.available_axis = [axis for axis, name in self.AXIS_TO_NAME.items()
                               if name in self.file]
        self.available_names = [self.AXIS_TO_NAME[axis] for axis in self.available_axis]

        # Save cube handlers to instance
        self.axis_to_cube = {}
        for axis in self.available_axis:
            name = self.AXIS_TO_NAME[axis]
            cube = self.file[name]

            self.axis_to_cube[axis] = cube
            setattr(self, name, cube)

        # Parse attributes from meta / set defaults
        self.add_attributes(**kwargs)


    def add_projection(self):
        """ TODO. """
        raise NotImplementedError

    def __getitem__(self, key):
        """ Select the fastest axis and use native `HDF5` slicing to retrieve data. """
        key, shape, squeeze = self.process_key(key)

        axis = self.get_optimal_axis(shape)
        cube = self.axis_to_cube[axis]
        order = self.AXIS_TO_ORDER[axis]
        transpose = self.AXIS_TO_TRANSPOSE[axis]

        slc = np.array(key)[order]
        crop = cube[tuple(slc)].transpose(transpose)

        if self.dtype == np.int8:
            crop = crop.astype(np.float32)
        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop

    def __setitem__(self, key, value):
        """ TODO. """
        raise NotImplementedError
