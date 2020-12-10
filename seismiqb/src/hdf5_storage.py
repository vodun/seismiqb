""" Class to store seismic cubes in HDF5 in multiple orientations. """
import os

import numpy as np
import h5py

from .utils import lru_cache, parse_axis

class FileHDF5:
    """ Class for storing 3D data in hdf5 format. To speed up loading, the file can store multiple transposed
    copies of the same cube.

    Parameters
    ----------
    filename : str
        Path to file.
    projections : str or None, optional
        String of 'i', 'h' and 'x', by default None. Is needed to create new hdf5 file.
    shape : tuple, optional
        Shape of 3D array in original orientation (iline, xline, depth), by default None.
        Is needed to create new hdf5 file.
    mode : 'r', 'r+' or 'a', optional
        Mode to open hdf5 file, by default 'r'
    """
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

    def get_cube(self, projection):
        """ Load the cube in the desired orientation.

        Parameters
        ----------
        projection : str ot int
            'i', 'x', 'h' or corresponding numerical index.

        Returns
        -------
        h5py._hl.dataset.Dataset
        """
        projection = parse_axis(projection)
        return self.file_hdf5[self.NAMES[projection]]

    def close(self):
        """ Close opened file. """
        self.file_hdf5.close()

    def _process_key(self, key):
        """ Process slices for cube to put into __getitem__ and __setitem__. """
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
        """ Load 2D slide from seismic cube in the fastest way by choosing the most appropriate orientation.

        Parameters
        ----------
        loc : int
            Slide position.
        axis : int, optional
            Corresponding axis, by default 0

        Returns
        -------
        np.ndarray
        """
        axis = parse_axis(axis)
        locations = [slice(None) for _ in range(3)]
        locations[axis] = slice(loc, loc+1)
        slc = [slice(None) for _ in range(3)]
        slc[axis] = 0
        return self.load_crop(locations, **kwargs)[slc]

    @lru_cache(128)
    def cached_load(self, cube, loc, axis=0, **kwargs):
        """ Load one slide of data from a certain cube projection.
        Caches the result in a thread-safe manner.
        """
        _ = kwargs
        slc = [slice(None), slice(None), slice(None)]
        slc[parse_axis(axis)] = loc
        return cube[slc[0], slc[1], slc[2]]

    def optimal_cube(self, locations):
        """ Choose optimal cube orientation to speed up crop loading.

        Parameters
        ----------
        locations : tuple of slices
            Crop position.

        Returns
        -------
        axis : int
            Optimal axis to load crop.

        Example
        -------
        self.projections = [0, 1],    locations = (slice(10), slice(100), slice(5)) -> axis = 0
        self.projections = [0, 1],    locations = (slice(100), slice(10), slice(5)) -> axis = 1
        self.projections = [0, 1, 2], locations = (slice(100), slice(10), slice(5)) -> axis = 2
        """
        shape = np.array([((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.shape)])
        indices = np.argsort(shape)
        axis = 0
        for axis in indices:
            if axis in self.projections:
                break
        return axis

    def load_crop(self, locations, projection=None, **kwargs):
        """ Load crop from seismic cube in the fastest way by choosing the most appropriate orientation.

        Parameters
        ----------
        locations : tuple of slices
            Slice of cube to load.
        projection : int or None, optional
            Cube projection to load crop from, by default None. None means that projection will be selected
            automatically in the most optimal way.
        """
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
        return np.stack([self.cached_load(cube_hdf5, iline, **kwargs)[xlines, :][:, heights]
                        for iline in range(ilines.start or start, ilines.stop or stop)])

    def _load_x(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube(1)
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self.cached_load(cube_hdf5, xline, **kwargs)[heights, :][:, ilines].transpose([1, 0])
                         for xline in range(xlines.start or start, xlines.stop or stop)], axis=1)

    def _load_h(self, ilines, xlines, heights, **kwargs):
        cube_hdf5 = self.get_cube(2)
        start, stop = 0, cube_hdf5.shape[0]
        return np.stack([self.cached_load(cube_hdf5, height, **kwargs)[ilines, :][:, xlines]
                         for height in range(heights.start or start, heights.stop or stop)], axis=2)

    def add_projection(self, projections, stride=100):
        """ Add additional cube orientations. To avoid load of the whole cube into memory it can be loaded
        by chunks.

        Parameters
        ----------
        projections : tuple of int or str
            Projections to add.
        stride : int, optional
            Stride for chunks, by default 100

        Notes
        -----
        New projections will be created from the first cube. Order of cubes is defined by self.projections.
        Chunks will be created along the first axis of that cube, transposed and assigned to corresponding
        region of new cube.
        """
        projections = [parse_axis(item) for item in projections]
        src_axis = self.projections[0]
        src_cube = self.get_cube(src_axis)
        shape = np.array(src_cube.shape)[self.TRANSPOSE[src_axis]]

        projections = [item for item in projections if item not in self.projections]
        self.projections += projections

        for i in range(0, src_cube.shape[0], stride):
            slide = src_cube[i:i+stride]
            slide = slide.transpose(self.TRANSPOSE[src_axis])
            main_axis = self.STRAIGHT[src_axis][0]
            for axis in projections:
                _shape = np.array(shape)[self.STRAIGHT[axis]]
                if self.NAMES[axis] not in self.file_hdf5:
                    self.file_hdf5.create_dataset(self.NAMES[axis], _shape)
                slices = [slice(None) for i in range(3)]
                slices[self.TRANSPOSE[axis][main_axis]] = slice(i, i+stride)
                self.get_cube(axis)[tuple(slices)] = slide.transpose(self.STRAIGHT[axis])
