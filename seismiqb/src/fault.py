""" Fault class and processing methods. """

import os
import glob

import numpy as np
import pandas as pd

from numba import njit, prange
from tqdm.auto import tqdm

from scipy.ndimage import measurements
from sklearn.decomposition import PCA

from .geometry import SeismicGeometry
from .horizon import Horizon
from .triangulation import triangulation, triangle_rasterization

class Fault(Horizon):
    """ Contains points of fault.

    Initialized from `storage` and `geometry`.

    Storage can be one of:
        - csv-like file in CHARISMA, REDUCED_CHARISMA or FAULT_STICKS format.
        - ndarray of (N, 3) shape.
        - hdf5 file as a binary mask for cube.
    """
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

    def __init__(self, *args, **kwargs):
        self.cube = None
        self.name = None
        super().__init__(*args, **kwargs)

    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA, REDUCED_CHARISMA or FAULT_STICKS csv-like file
        from .npy or .hdf5 file with points. """
        _ = kwargs
        self.name = os.path.basename(path)
        ext = os.path.splitext(path)[1][1:]
        if ext == 'npy':
            points = np.load(path, allow_pickle=False)
            self.from_points(points, False, **kwargs)
        elif ext == 'hdf5':
            cube = SeismicGeometry(path, **kwargs).file_hdf5['cube']
            points = np.stack(np.where(np.array(cube) == 1)).T
            self.from_points(points, False, **kwargs)
        else:
            points = self.csv_to_points(path, **kwargs)
            self.from_points(points, transform, **kwargs)

    def csv_to_points(self, path, **kwargs):
        """ Get point cloud array from file values. """
        df = self.read_file(path)
        df = self.fix_lines(df)
        sticks = self.read_sticks(df)
        sticks = self.sort_sticks(sticks)
        points = self.interpolate_3d(sticks, **kwargs)
        return points

    @classmethod
    def read_file(cls, path):
        """ Read data frame with sticks. """
        with open(path) as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])
        if line_len == 3:
            names = Horizon.REDUCED_CHARISMA_SPEC
        elif line_len == 8:
            names = cls.FAULT_STICKS
        elif line_len >= 9:
            names = Horizon.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        return pd.read_csv(path, sep=r'\s+', names=names)

    def fix_lines(self, df):
        """ Fix broken iline and crossline coordinates. If coordinates are out of the cube, 'iline' and 'xline'
        will be infered from 'cdp_x' and 'cdp_y'. """
        i_bounds = [self.geometry.ilines_offset, self.geometry.ilines_offset + self.geometry.cube_shape[0]]
        x_bounds = [self.geometry.xlines_offset, self.geometry.xlines_offset + self.geometry.cube_shape[1]]

        i_mask = np.logical_or(df.iline < i_bounds[0], df.iline >= i_bounds[1])
        x_mask = np.logical_or(df.xline < x_bounds[0], df.xline >= x_bounds[1])

        _df = df[np.logical_and(i_mask, x_mask)]

        df.loc[np.logical_and(i_mask, x_mask), ['iline', 'xline']] = np.rint(
            self.geometry.cdp_to_lines(_df[['cdp_x', 'cdp_y']].values)
        ).astype('int32')

        return df

    @classmethod
    def read_sticks(cls, df):
        """ Transform initial fault dataframe to array of sticks. """
        if 'number' in df.columns:
            col = 'number'
        elif df.iline.iloc[0] == df.iline.iloc[1]:
            col = 'iline'
        elif df.xline.iloc[0] == df.xline.iloc[1]:
            col = 'xline'
        else:
            raise ValueError('Wrong format of sticks: there is no column to group points into sticks.')
        return df.groupby(col).apply(lambda x: x[Horizon.COLUMNS].values).reset_index(drop=True)

    def sort_sticks(self, sticks):
        """ Sort sticks with respect of fault direction. """
        pca = PCA(1)
        coords = pca.fit_transform(pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values])))
        indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
        return sticks.iloc[indices]

    def interpolate_3d(self, sticks, **kwargs):
        """ Interpolate fault sticks as a surface. """
        width = kwargs.get('width', 1)
        triangles = triangulation(sticks)
        points = []
        for triangle in triangles:
            res = triangle_rasterization(triangle, width)
            points += [res]
        return np.concatenate(points, axis=0)

    def add_to_mask(self, mask, locations=None, **kwargs):
        """ Add fault to background. """
        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                            [locations[1].start, locations[1].stop],
                            [locations[2].start, locations[2].stop]],
                            dtype=np.int32)
        points = self.points
        _min = self.i_min, self.x_min, self.h_min
        _max = self.i_max, self.x_max, self.h_max

        if (_max < mask_bbox[:, 0]).any() or (_min >= mask_bbox[:, 1]).any():
            return mask
        for i in range(3):
            points = points[points[:, i] >= locations[i].start]
            points = points[points[:, i] < locations[i].stop]
        points = points - np.array(mask_bbox[:, 0]).reshape(1, 3)
        mask[points[:, 0], points[:, 1], points[:, 2]] = 1
        return mask

    @classmethod
    def check_format(cls, path, verbose=False):
        """ Find errors in fault file.

        Parameters
        ----------
        path : str
            path to file or glob expression
        verbose : bool
            response if file is succesfully readed.
        """
        for filename in glob.glob(path):
            try:
                df = cls.read_file(filename)
            except ValueError:
                print(filename, ': wrong format')
            else:
                if 'name' in df.columns and len(df.name.unique()) > 1:
                    print(filename, ': fault file must be splitted.')
                elif len(cls.read_sticks(df)) == 1:
                    print(filename, ': fault has an only one stick')
                elif any(cls.read_sticks(df).apply(len) == 1):
                    print(filename, ': fault has one point stick')
                elif verbose:
                    print(filename, ': OK')

    @classmethod
    def split_file(cls, path, dst='faults'):
        """ Split file with multiple faults into separate files. """
        folder = os.path.dirname(path)
        faults_folder = os.path.join(folder, dst)
        if faults_folder and not os.path.isdir(faults_folder):
            os.makedirs(faults_folder)
        df = pd.read_csv(path, sep=r'\s+', names=cls.FAULT_STICKS)
        df.groupby('name').apply(cls.fault_to_csv, folder=folder, dst=dst)

    @classmethod
    def fault_to_csv(cls, df, folder, dst):
        """ Save separate fault to csv. """
        df.to_csv(os.path.join(folder, dst, df.name), sep=' ', header=False, index=False)

def split_faults(array, step=None, overlap=1, pbar=False):
    """ Label faults in an array.

    Parameters
    ----------
    array : np.ndarray or SeismicGeometry
        binary mask of faults
    step : int
        size of chunks to apply `measurements.label`
    overlap : int
        size of overlap to join faults from different chunks
    pbar : bool
        progress bar

    Returns
    -------
    np.ndarray
        array of the same size with labels
    """
    if isinstance(array, SeismicGeometry):
        array = array.file_hdf5['cube']
    if step is None:
        step = len(array)
    chunks = [(start, array[start:start+step]) for start in range(0, array.shape[0], step-overlap)]
    s = np.ones((3, 3, 3))
    labels = np.zeros_like(array)
    n_objects = 0
    if pbar:
        chunks = tqdm(chunks)
    for start, item in chunks:
        objects, _n_objects = measurements.label(item, structure=s)
        objects[objects > 0] += n_objects
        coords = np.where(labels[start:start+overlap] > 0)
        transform = {k: v for k, v in zip(
            objects[:overlap][coords[0], coords[1], coords[2]],
            labels[start:start+overlap][coords[0], coords[1], coords[2]]
        ) if k != v}

        for k, v in transform.items():
            objects[objects == k] = v

        labels[start:start+step] = objects
        n_objects += _n_objects
    labels = _sequential_labels(labels)
    indices = np.unique(labels)[1:]
    sizes = faults_sizes(labels, indices)
    return labels, sizes

def faults_sizes(labels, indices):
    """ Compute sizes of faults.

    Parameters
    ----------
    labels : numpy.ndarray
        3d array with labels
    indices : numpy.ndarray
        indices of faults to compute size
    Returns
    -------
    sizes : numpy.ndarray
    """
    bounds = np.zeros((len(indices), 4))
    bounds[:, 0] = labels.shape[0]
    bounds[:, 1] = labels.shape[1]
    sizes = np.zeros_like(indices)
    return _faults_sizes(labels, indices, bounds, sizes)

@njit(parallel=True)
def _faults_sizes(labels, indices, bounds, sizes):
    for i in prange(labels.shape[0]): # pylint: disable=not-an-iterable
        for j in prange(labels.shape[1]): # pylint: disable=not-an-iterable
            for k in prange(labels.shape[2]): # pylint: disable=not-an-iterable
                if labels[i, j, k] > 0:
                    index = np.where(indices == labels[i, j, k])[0][0]
                    bounds[index, 0] = min(bounds[index][0], i)
                    bounds[index, 1] = min(bounds[index][1], j)
                    bounds[index, 2] = max(bounds[index][2], i)
                    bounds[index, 3] = max(bounds[index][3], j)

    for i in prange(len(sizes)): # pylint: disable=not-an-iterable
        sizes[i] = ((bounds[i, 2] - bounds[i, 0]) ** 2 + (bounds[i, 3] - bounds[i, 1]) ** 2) ** 0.5
    return sizes

def filter_faults(labels, threshold, sizes=None):
    """ Filter faults by size.

    Parameters
    ----------
    labels : numpy.ndarray
        3d array with labels
    threshold : float
        faults with the size less then threshold will be removed
    sizes : numpy.ndarray or sizes
        precompured sizes of faults
    Returns
    -------
    sizes : numpy.ndarray
    """
    indices = np.unique(labels)[1:]
    if sizes is None:
        sizes = faults_sizes(labels, indices)
    return _filter_faults(labels, sizes, indices, threshold)

@njit(parallel=True)
def _sequential_labels(labels):
    indices = np.unique(labels)
    for i in prange(labels.shape[0]): # pylint: disable=not-an-iterable
        for j in prange(labels.shape[1]): # pylint: disable=not-an-iterable
            for k in prange(labels.shape[2]): # pylint: disable=not-an-iterable
                if labels[i, j, k] != 0:
                    label = np.where(indices == labels[i, j, k])[0][0]
                    labels[i, j, k] = label
    return labels

@njit(parallel=True)
def _filter_faults(labels, sizes, indices, threshold):
    for i in prange(labels.shape[0]): # pylint: disable=not-an-iterable
        for j in prange(labels.shape[1]): # pylint: disable=not-an-iterable
            for k in prange(labels.shape[2]): # pylint: disable=not-an-iterable
                if labels[i, j, k] > 0:
                    index = np.where(indices == labels[i, j, k])[0][0]
                    if sizes[index] < threshold:
                        labels[i, j, k] = 0
    return labels
