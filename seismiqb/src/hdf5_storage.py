import os

import numpy as np
import h5py

from .utils import lru_cache, parse_axis

class FileHDF5:
    STRAIGHT = {0: [0, 1, 2], 1: [1, 2, 0], 2: [2, 0, 1]}
    TRANSPOSE = {0: [0, 1, 2], 1: [2, 0, 1], 2: [1, 2, 0]}
    NAMES = {0: 'cube', 1: 'cube_x', 2: 'cube_h'}

    def __init__(self, filename, projections=None, shape=None, mode='r'):
        self.filename = filename
        if mode in ('r', 'r+'):
            self.file_hdf5 = h5py.File(filename, mode)
            self.projections = [axis for axis in range(3) if self.NAMES[axis] in self.file_hdf5]
            axis = self.projections[0]
            self.shape = tuple(np.array(self.get_cube(axis).shape)[self.TRANSPOSE[axis]])
        elif mode == 'a':
            if os.path.exists(filename):
                os.remove(filename)
            self.file_hdf5 = h5py.File(filename, 'a')
            self.projections = projections or [0, 1, 2]
            self.projections = [parse_axis(item) for item in self.projections]
            self.shape = shape
            for p in self.projections:
                _shape = np.array(shape)[self.STRAIGHT[p]]
                self.file_hdf5.create_dataset(self.NAMES[p], _shape)

    def parse_axis(self, axis):
        """ Convert string representation of an axis into integer, if needed. """
        if isinstance(axis, str):
            if axis in self.index_headers:
                axis = self.index_headers.index(axis)
            elif axis in ['i', 'il', 'iline']:
                axis = 0
            elif axis in ['x', 'xl', 'xline']:
                axis = 1
            elif axis in ['h', 'height', 'depth']:
                axis = 2
        return axis

    def get_cube(self, projection):
        projection = parse_axis(projection)
        return self.file_hdf5[self.NAMES[projection]]

    def close(self):
        self.file_hdf5.close()

    def _process_key(self, key):
        key_ = [key] if isinstance(key, slice) else list(key)
        key, squeeze = [], []
        if len(key_) != len(self.shape):
            key_ += [slice(None)] * (len(self.shape) - len(key_))
        for i, item in enumerate(key_):
            max_size = self.shape[i]

            if isinstance(item, slice):
                slc = slice(item.start or 0, item.stop or max_size)
            elif isinstance(item, int):
                item = item if item >= 0 else max_size - item
                slc = slice(item, item + 1)
                squeeze.append(i)
            key.append(slc)
        return key, squeeze

    def __getitem__(self, key):
        key, squeeze = self._process_key(key)
        projection = self.optimal_cube(key)
        slices = np.array(key)[self.STRAIGHT[projection]]
        crop = self.get_cube(projection)[slices[0], slices[1], slices[2]].transpose(self.TRANSPOSE[projection])
        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop

    def __setitem__(self, key, value):
        key, _ = self._process_key(key)
        for projection in self.projections:
            slices = np.array(key)[self.STRAIGHT[projection]]
            self.get_cube(projection)[slices[0], slices[1], slices[2]] = value.transpose(self.STRAIGHT[projection])

    def load_slide(self, loc, axis=0, **kwargs):
        axis = parse_axis(axis)
        locations = [slice(None) for _ in range(3)]
        locations[axis] = slice(loc, loc+1)
        slc = [slice(None) for _ in range(3)]
        slc[axis] = 0
        return self.load_crop(locations, **kwargs)[slc]

    @lru_cache(128)
    def _cached_load(self, cube, loc, axis=0, **kwargs):
        """ Load one slide of data from a certain cube projection.
        Caches the result in a thread-safe manner.
        """
        _ = kwargs
        slc = [slice(None), slice(None), slice(None)]
        slc[parse_axis(axis)] = loc
        return cube[slc[0], slc[1], slc[2]]

    def optimal_cube(self, locations):
        shape = np.array([((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.shape)])
        indices = np.argsort(shape)
        for axis in indices:
            if axis in self.projections:
                break
        return axis

    def load_crop(self, locations, projection=None, **kwargs):
        projection = projection or self.optimal_cube(locations)
        if projection == 1:
            crop = self._load_x(*locations, **kwargs)
        elif projection == 2:
            crop = self._load_h(*locations, **kwargs)
        else: # backward compatibility
            crop = self._load_i(*locations, **kwargs)
        return crop

    def _load_i(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube(0)
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self._cached_load(cube_hdf5, iline, **kwargs)[xlines, :][:, heights]
                        for iline in range(ilines.start or start, ilines.stop or stop)])

    def _load_x(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube(1)
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self._cached_load(cube_hdf5, xline, **kwargs)[heights, :][:, ilines].transpose([1, 0])
                         for xline in range(xlines.start or start, xlines.stop or stop)], axis=1)

    def _load_h(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube(2)
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self._cached_load(cube_hdf5, height, **kwargs)[ilines, :][:, xlines]
                         for height in range(heights.start or start, heights.stop or stop)], axis=2)

    def add_projection(self, projections, stride=100):
        projections = [parse_axis(item) for item in projections]
        src_axis = self.projections[0]
        src_cube = self.get_cube(src_axis)
        shape = np.array(src_cube.shape)[self.TRANSPOSE[src_axis]]

        for i in range(0, src_cube.shape[0], stride):
            slide = src_cube[i:i+stride]
        slide = slide.transpose(self.TRANSPOSE[src_axis])
        main_axis = self.STRAIGHT[src_axis][0]
        for axis in projections:
            if axis not in self.projections:
                _shape = np.array(shape)[self.STRAIGHT[axis]]
                if self.NAMES[axis] not in self.file_hdf5:
                    self.file_hdf5.create_dataset(self.NAMES[axis], _shape)
                slices = [slice(None) for i in range(3)]
                slices[self.TRANSPOSE[axis][main_axis]] = slice(i, i+stride)
                self.get_cube(axis)[tuple(slices)] = slide.transpose(self.STRAIGHT[axis])
                self.projections = self.projections + [axis]
