""" !!. """

import numpy as np
import h5py

from .base import Geometry
from ..utils import lru_cache



class GeometryHDF5(Geometry):
    """ !!. """
    FILE_OPENER = h5py.File

    def init(self, path, mode='r+', **kwargs):
        """ !!. """
        # Open the file
        self.file = self.FILE_OPENER(path, mode)

        # Check available projections
        self.available_axis = [axis for axis, name in self.AXIS_TO_NAME.items()
                               if name in self.file]
        self.available_names = [self.AXIS_TO_NAME[axis] for axis in self.available_axis]

        # Save projection handlers to instance
        self.axis_to_projection = {}
        for axis in self.available_axis:
            name = self.AXIS_TO_NAME[axis]
            projection = self.file[name]

            self.axis_to_projection[axis] = projection
            setattr(self, name, projection)

        # Parse attributes from meta / set defaults
        self.add_attributes(**kwargs)

    def add_attributes(self, **kwargs):
        """ !!. """
        # Innate attributes of converted geometry
        self.index_headers = ('INLINE_3D', 'CROSSLINE_3D')
        self.converted = True

        # Get from meta / set defaults
        if self.meta_exists:
            self.load_meta(names=self.PRESERVED + self.PRESERVED_LAZY)
            self.has_stats = True
        else:
            self.set_default_index_attributes(**kwargs)
            self.has_stats = False

        # Infer attributes from the available projections; validate others
        axis = self.available_axis[0]
        projection = self.axis_to_projection[axis]

        shape = np.array(projection.shape)[self.AXIS_TO_TRANSPOSE[axis]]
        if hasattr(self, 'shape'):
            if (getattr(self, 'shape') != shape).any():
                raise ValueError('Projection shape is not the same as shape, loaded from meta!')
        else:
            self.shape = shape
            *self.lengths, self.depth = shape

        self.dtype = projection.dtype
        self.quantized = (projection.dtype == np.int8)

    def set_default_index_attributes(self, **kwargs):
        """ !!. """
        self.delay, self.sample_rate = 0.0, 1.0
        for key, value in kwargs.items():
            setattr(self, key, value)


    # General utilities
    def get_optimal_axis(self, locations=None, shape=None):
        """ Choose the fastest axis from available projections, based on shape. """
        shape = shape or self.locations_to_shape(locations)

        for axis in np.argsort(shape):
            if axis in self.available_axis:
                return axis
        return None


    # Load data: 2D
    def load_slide(self, index, axis=0, limits=None, buffer=None, safe=True, use_cache=False):
        """ !!. """
        axis = self.parse_axis(axis)

        if limits is not None and axis==2:
            raise ValueError('Providing `limits` with `axis=2` is meaningless!')

        if use_cache is False:
            return self.load_slide_native(index=index, axis=axis, limits=limits, buffer=buffer, safe=safe)
        return self.load_slide_cached(index=index, axis=axis, limits=limits, buffer=buffer)

    def load_slide_native(self, index, axis=0, limits=None, buffer=None, safe=True):
        """ !!. """
        if safe or buffer is None or buffer.dtype != self.dtype:
            buffer = self.load_slide_native_safe(index=index, axis=axis, limits=limits, buffer=buffer)
        else:
            self.load_slide_native_unsafe(index=index, axis=axis, limits=limits, buffer=buffer)
        return buffer

    def load_slide_native_safe(self, index, axis=0, limits=None, buffer=None):
        """ Use public API: can't read directly into buffer, requires a copy. !!. """
        # Prepare locations
        loading_axis = axis if axis in self.available_axis else self.available_axis[0]
        locations = self.make_slide_locations(index=index, axis=axis)
        locations = [locations[idx] for idx in self.AXIS_TO_ORDER[loading_axis]]

        if limits is not None:
            locations[-1] = self.process_limits(limits)
        locations = tuple(locations)

        # Load data
        slide = self.axis_to_projection[loading_axis][locations]
        if self.quantized:
            if buffer is None or buffer.dtype != slide.dtype:
                slide = slide.astype(np.float32)

        # Re-order and squeeze the requested axis
        transposition = self.AXIS_TO_TRANSPOSE[loading_axis]
        slide = slide.transpose(transposition)
        slide = slide.squeeze(axis)

        # Write back to buffer
        if buffer is not None:
            buffer[:] = slide
        else:
            buffer = slide
        return buffer

    def load_slide_native_unsafe(self, index, axis=0, limits=None, buffer=None):
        """ !!. """
        # Prepare locations
        loading_axis = axis if axis in self.available_axis else self.available_axis[0]
        locations = self.make_slide_locations(index=index, axis=axis)
        locations = [locations[idx] for idx in self.AXIS_TO_ORDER[loading_axis]]

        if limits is not None:
            locations[-1] = self.process_limits(limits)
        locations = tuple(locations)

        # View buffer in projections ordering
        buffer = np.expand_dims(buffer, axis)
        transposition = self.AXIS_TO_ORDER[loading_axis]
        buffer = buffer.transpose(transposition)

        # Load data
        self.axis_to_projection[loading_axis].read_direct(buffer, locations)

        # View buffer in original ordering
        transposition = self.AXIS_TO_TRANSPOSE[loading_axis]
        buffer = buffer.transpose(transposition)
        buffer = buffer.squeeze(axis)
        return buffer

    @lru_cache(128)
    def load_slide_cached(self, index, axis=0, limits=None, buffer=None):
        """ !!. """
        _ = buffer
        return self.load_slide_native_safe(index=index, axis=axis, limits=limits, buffer=None)


    # Load data: 3D
    def load_crop(self, locations, buffer=None, use_cache=False, safe=True):
        """ !!. """
        if use_cache is False:
            return self.load_crop_native(locations=locations, buffer=buffer, safe=safe)
        return self.load_crop_cached(locations=locations, buffer=buffer)

    def load_crop_native(self, locations, axis=None, buffer=None, safe=True):
        """ !!. """
        axis = axis or self.get_optimal_axis(locations=locations)
        if axis not in self.available_axis:
            raise ValueError('Axis={axis} is not available!')

        if safe or axis == 2 or buffer is None or buffer.dtype != self.dtype:
            buffer = self.load_crop_native_safe(locations=locations, axis=axis, buffer=buffer)
        else:
            self.load_crop_native_unsafe(locations=locations, axis=axis, buffer=buffer)
        return buffer

    def load_crop_native_safe(self, locations, axis=None, buffer=None):
        """ Use public API: can't read directly into buffer, requires a copy. !!. """
        # Prepare locations
        locations = [locations[idx] for idx in self.AXIS_TO_ORDER[axis]]
        locations = tuple(locations)

        # Load data
        crop = self.axis_to_projection[axis][locations]
        if self.quantized:
            if buffer is None or buffer.dtype != crop.dtype:
                crop = crop.astype(np.float32)

        # Re-order and squeeze the requested axis
        transposition = self.AXIS_TO_TRANSPOSE[axis]
        crop = crop.transpose(transposition)

        # Write back to buffer
        if buffer is not None:
            buffer[:] = crop
        else:
            buffer = crop
        return buffer

    def load_crop_native_unsafe(self, locations, axis=None, buffer=None):
        """ !!. """
        # Prepare locations
        locations = [locations[idx] for idx in self.AXIS_TO_ORDER[axis]]
        locations = tuple(locations)

        # View buffer in projections ordering
        transposition = self.AXIS_TO_ORDER[axis]
        buffer = buffer.transpose(transposition)

        # Load data
        self.axis_to_projection[axis].read_direct(buffer, locations)

        # View buffer in original ordering
        transposition = self.AXIS_TO_TRANSPOSE[axis]
        buffer = buffer.transpose(transposition)
        return buffer

    def load_crop_cached(self, locations, axis=None, buffer=None):
        """ !!. """
        # Parse parameters
        shape = self.locations_to_shape(locations)
        axis = axis or self.get_optimal_axis(shape=shape)

        locations = [locations[idx] for idx in self.AXIS_TO_ORDER[axis]]
        locations = tuple(locations)

        # Prepare buffer
        if buffer is None:
            buffer = np.empty(shape, dtype=np.float32)
        transposition = self.AXIS_TO_ORDER[axis]
        buffer = buffer.transpose(transposition)

        # Load data
        for i, idx in enumerate(range(locations[0].start, locations[0].stop)):
            buffer[i] = self.load_slide_cached(index=idx, axis=axis)[locations[1], locations[2]]

        # View buffer in original ordering
        transposition = self.AXIS_TO_TRANSPOSE[axis]
        buffer = buffer.transpose(transposition)
        return buffer
