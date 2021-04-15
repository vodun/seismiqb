""" HDF5 geometry. """
import os
import numpy as np

from ..utility_classes import lru_cache
from .base import SeismicGeometry


class SeismicGeometryConverted(SeismicGeometry):
    """ Contains common methods and logic for converted formats of seismic storages: currently, `HDF5` and `BLOSC`.

    Underlying data storage must contain cube projections under the `cube_i`, `cube_x` and `cube_h` keys,
    which allow indexing with square brackets to get actual data: for example, to get the 100-iline slice::
        slide = hdf5_file['cube_i'][100, :, :]
    Some of the projections may be missing — in this case, other (possibly, slower) projections are used to load data.
    Projections msut be stored in the `axis_to_cube` attribute

    The storage itself contains only data; attributes, stats and relevant geological info are stored in the `.meta`.

    This class provides API for loading data: `load_slide` and `load_crop` methods.
    """
    #pylint: disable=attribute-defined-outside-init
    AXIS_TO_NAME = {0: 'cube_i', 1: 'cube_x', 2: 'cube_h'}
    AXIS_TO_ORDER = {0: [0, 1, 2], 1: [1, 2, 0], 2: [2, 0, 1]}
    AXIS_TO_TRANSPOSE = {0: [0, 1, 2], 1: [2, 0, 1], 2: [1, 2, 0]}

    def process(self):
        """ Create and process file handler. Must be implemented in the child classes. """


    def add_attributes(self, **kwargs):
        """ If meta is available, retrieves values from it. Otherwise, uses defaults.
        Also infers some values from the data itself. No passing through data.
        """
        self.structured = True
        self.index_headers = self.INDEX_POST

        if os.path.exists(self.path_meta):
            self.load_meta()
            self.has_stats = True
        else:
            self.set_default_attributes(**kwargs)
            self.has_stats = False

        # Parse attributes from file itself
        axis = self.available_axis[0]
        cube = self.axis_to_cube[axis]

        self.cube_shape = np.array(cube.shape)[self.AXIS_TO_ORDER[axis]]
        self.lens = self.cube_shape[:2]
        self.depth = self.cube_shape[-1]
        self.dtype = cube.dtype
        self.quantized = cube.dtype == np.int8

    def set_default_attributes(self, **kwargs):
        """ Set values of attributes to defaults. """
        self.delay, self.sample_rate = 0.0, 1.0

        for key, value in kwargs.items():
            setattr(self, key, value)


    # Methods to load actual data from underlying storage
    def get_optimal_axis(self, shape):
        """ Choose the fastest axis from available projections, based on shape. """
        for axis in np.argsort(shape):
            if axis in self.available_axis:
                return axis
        return None

    def load_crop(self, locations, axis=None, **kwargs):
        """ Load 3D crop from the cube.
        Automatically chooses the fastest projection to use.

        Parameters
        ----------
        locations : sequence of slices
            Location to load: slices along the first index, the second, and depth.
        axis : str or int
            Identificator of the axis to use to load data.
            Can be `iline`, `xline`, `height`, `depth`, `i`, `x`, `h`, 0, 1, 2.
        """
        locations, shape, _ = self.process_key(locations)

        # Choose axis
        if axis is None:
            axis = self.get_optimal_axis(shape)
        else:
            axis = self.parse_axis(axis)
            if axis not in self.available_axis:
                raise ValueError(f'Axis {axis} is not available in the {self.name}!')

        # Retrieve cube handler and ordering for axis
        cube = self.axis_to_cube[axis]
        buffer_shape = np.array(shape)[self.AXIS_TO_ORDER[axis]]
        transpose = self.AXIS_TO_TRANSPOSE[axis]

        # Create memory buffer and load data into it
        buffer = np.empty(buffer_shape, dtype=self.dtype)
        method = getattr(self, f'_load_{axis}')
        crop = method(buffer, cube, *locations, **kwargs)

        # Set correct dtype and axis ordering
        if self.dtype == np.int8:
            crop = crop.astype(np.float32)
        return crop.transpose(transpose)

    def _load_0(self, buffer, cube, ilines, xlines, heights, **kwargs):
        """ Load data from iline projection. """
        for i, iline in enumerate(range(ilines.start, ilines.stop)):
            buffer[i] = self._cached_load(cube, iline, **kwargs)[xlines, :][:, heights]
        return buffer

    def _load_1(self, buffer, cube, ilines, xlines, heights, **kwargs):
        """ Load data from xline projection. """
        for i, xline in enumerate(range(xlines.start, xlines.stop)):
            buffer[i] = self._cached_load(cube, xline, **kwargs)[heights, :][:, ilines]
        return buffer

    def _load_2(self, buffer, cube, ilines, xlines, heights, **kwargs):
        """ Load data from depth projection. """
        for i, height in enumerate(range(heights.start, heights.stop)):
            buffer[i] = self._cached_load(cube, height, **kwargs)[ilines, :][:, xlines]
        return buffer

    @lru_cache(128)
    def _cached_load(self, cube, loc, **kwargs):
        """ Load one slide of data from a supplied cube projection. Caches the result in a thread-safe manner. """
        _ = kwargs
        return cube[loc, :, :]

    @lru_cache(128)
    def _cached_construct(self, loc, axis,**kwargs):
        """ Create one slide of data from other projections. """
        _ = kwargs

        locations = [slice(None) for _ in range(3)]
        locations[axis] = slice(loc, loc + 1)
        return self.load_crop(locations, **kwargs).squeeze()

    def load_slide(self, loc, axis='iline', **kwargs):
        """ Load desired slide along desired axis.
        If the `axis` projection is available, loads directly from it.
        Otherwise, creates the slide based on the fastest of other projections.
        """
        axis = self.parse_axis(axis)

        if axis in self.available_axis:
            cube = self.axis_to_cube[axis]
            slide = self._cached_load(cube, loc, **kwargs)
        else:
            slide = self._cached_construct(loc, axis, **kwargs)

        if axis == 1:
            slide = slide.T
        return slide
