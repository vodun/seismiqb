import os

import numpy as np
import h5py

from .utils import lru_cache


import numpy as np
import h5py

import numpy as np
import h5py

class FileHDF5:
    AXES = {'i': 0, 'x': 1, 'h': 2}
    AXES_ALIASES = {0: 'i', 1: 'x', 2: 'h'}
    STRAIGHT = {
        'i': [0, 1, 2], 'x': [1, 2, 0], 'h': [2, 0, 1],
        0: [0, 1, 2], 1: [1, 2, 0], 2: [2, 0, 1]
    }
    TRANSPOSE = {
        'i': [0, 1, 2], 'x': [2, 0, 1], 'h': [1, 2, 0],
        0: [0, 1, 2], 1: [2, 0, 1], 2: [1, 2, 0]
    }
    NAMES = {
        'i': 'cube', 'x': 'cube_x', 'h': 'cube_h',
        0: 'cube', 1: 'cube_x', 2: 'cube_h'
    }

    def __init__(self, filename, projections=None, shape=None, mode='r'):
        self.filename = filename
        if mode in ('r', 'r+'):
            self.file_hdf5 = h5py.File(filename, mode)
            axes_in_cube = [axis for axis in 'ixh' if self.NAMES[axis] in self.file_hdf5]
            self.projections = ''.join(axes_in_cube)
            axis = self.projections[0]
            self.shape = tuple(np.array(self.get_cube(axis).shape)[self.TRANSPOSE[axis]])
        elif mode == 'a':
            if os.path.exists(filename):
                os.remove(filename)
            self.file_hdf5 = h5py.File(filename, 'a')
            self.projections = projections or 'ixh'
            self.shape = shape
            for p in self.projections:
                _shape = np.array(shape)[self.STRAIGHT[p]]
                self.file_hdf5.create_dataset(self.NAMES[p], _shape)

    def get_cube(self, projection):
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

#     def load_slide(self, loc, axis='i'):
#         load_axis = {
#             0: {'x': 2, 'h': 1},
#             1: {'i': 1, 'h': 2},
#             2: {'i': 2, 'x': 1}
#         }
#         if axis in self.projections:
#             cube = self.get_cube(axis)
#             slide = self._cached_load(cube, loc, 0)
#             if axis == 'x':
#                 slide = slide.T
#         else:
#             axis_1, axis_2 = [i for i in range(3) if i != self.AXES[axis]]
#             if self.shape[axis_1] > self.shape[axis_2]:
#                 axis_1, axis_2 = axis_2, axis_1
#             if self.AXES_ALIASES[axis_1] in self.projections:
#                 load_from = 'ixh'[axis_1]
#             else:
#                 load_from = 'ixh'[axis_2]
#             cube = self.get_cube(load_from)
#             slide = self._cached_load(cube, loc, load_axis[self.AXES[axis]][load_from])
#             if load_from == 'x' and axis != 'i' or load_from == 'h' and axis != 'h':
#                 slide = slide.T
#         return slide

    def load_slide(self, loc, axis=0, **kwargs):
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
        slc[axis] = loc
        return cube[slc[0], slc[1], slc[2]]

    def optimal_cube(self, locations):
        shape = np.array([((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.shape)])
        indices = np.argsort(shape)
        for axis in indices:
            if 'ixh'[axis] in self.projections:
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
        cube_hdf5 = self.get_cube('i')
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self._cached_load(cube_hdf5, iline, **kwargs)[xlines, :][:, heights]
                        for iline in range(ilines.start or start, ilines.stop or stop)])

    def _load_x(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube('x')
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self._cached_load(cube_hdf5, xline, **kwargs)[heights, :][:, ilines].transpose([1, 0])
                         for xline in range(xlines.start or start, xlines.stop or stop)], axis=1)

    def _load_h(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube('h')
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self._cached_load(cube_hdf5, height, **kwargs)[ilines, :][:, xlines]
                         for height in range(heights.start or start, heights.stop or stop)], axis=2)

    def add_projection(self, projections, stride=100):
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
                self.projections = self.projections + axis
