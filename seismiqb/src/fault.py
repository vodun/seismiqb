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
    #pylint: disable=attribute-defined-outside-init
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

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

        if (self.bbox[:, 1] < mask_bbox[:, 0]).any() or (self.bbox[:, 0] >= mask_bbox[:, 1]).any():
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

def split_faults(array, chunk_size=None, overlap=1, pbar=False):
    """ Label faults in an array.

    Parameters
    ----------
    array : numpy.ndarray or SeismicGeometry
        binary mask of faults
    chunk_size : int
        size of chunks to apply `measurements.label`
    overlap : int
        size of overlap to join faults from different chunks
    pbar : bool
        progress bar

    Returns
    -------
    numpy.ndarray
        array of shape (N, 4) where the first 3 columns are coordinates of points and the last one
        is for labels
    """
    if isinstance(array, SeismicGeometry):
        array = array.file_hdf5['cube']
    if chunk_size is None:
        chunk_size = len(array)
    chunks = [(start, array[start:start+chunk_size+overlap]) for start in range(0, array.shape[0], chunk_size)]

    prev_overlap = np.zeros((0, *array.shape[1:]))
    labels = np.zeros((0, 4), dtype='int32')
    n_objects = 0
    s = np.ones((3, 3, 3))
    if pbar:
        chunks = tqdm(chunks)
    for start, item in chunks:
        chunk_labels, new_objects = measurements.label(item, structure=s) # compute labels for new chunk
        chunk_labels[chunk_labels > 0] += n_objects # shift all values to avoid intersecting with previous labels
        new_overlap = chunk_labels[:overlap]

        if len(prev_overlap) > 0:
            coords = np.where(prev_overlap > 0)
            if len(coords[0]) > 0:
                # while there are the same objects with different labels repeat procedure
                while (new_overlap != prev_overlap).any():
                    # find overlapping objects and change labels in chunk
                    chunk_transform = {k: v for k, v in zip(new_overlap[coords], prev_overlap[coords]) if k != v}
                    for k, v in chunk_transform.items():
                        chunk_labels[chunk_labels == k] = v
                    new_overlap = chunk_labels[:overlap]

                    # find overlapping objects and change labels in processed part of cube
                    labels_transform = {k: v for k, v in zip(prev_overlap[coords], new_overlap[coords]) if k != v}
                    for k, v in labels_transform.items():
                        labels[labels[:, 3] == k, 3] = v
                        prev_overlap[prev_overlap == k] = v

        prev_overlap = chunk_labels[-overlap:]
        chunk_labels = chunk_labels[overlap:]

        nonzero_coord = np.where(chunk_labels)
        chunk_labels = np.stack([*nonzero_coord, chunk_labels[nonzero_coord]], axis = -1)
        chunk_labels[:, 0] += start
        labels = np.concatenate([labels, chunk_labels])
        n_objects += new_objects

    labels = _sequential_labels(labels) # make labels sequential from 1 to number of labels
    sizes = faults_sizes(labels) # compute object sizes
    return labels, sizes

@njit(parallel=True)
def _sequential_labels(labels):
    indices = np.unique(labels[:, 3])
    for i in prange(len(labels)): # pylint: disable=not-an-iterable
        labels[i, 3] = np.where(indices == labels[i, 3])[0][0] + 1
    return labels

@njit(parallel=True)
def faults_sizes(labels):
    """ Compute sizes of faults.

    Parameters
    ----------
    labels : numpy.ndarray
        array of shape (N, 4) where the first 3 columns are coordinates of points and the last one
        is for labels
    Returns
    -------
    sizes : numpy.ndarray
    """
    indices = np.unique(labels[:, 3])
    sizes = np.zeros_like(indices)
    for i in prange(len(indices)): # pylint: disable=not-an-iterable
        label = indices[i]
        array = labels[labels[:, 3] == label]
        sizes[label-1] = ((array[:, 0].max() - array[:, 0].min()) ** 2 + (array[:, 1].max() - array[:, 1].min())) ** 0.5
    return sizes

def filter_faults(labels, threshold, sizes=None):
    """ Filter faults by size.

    Parameters
    ----------
    labels : numpy.ndarray
        array of shape (N, 4) where the first 3 columns are coordinates of points and the last one
        is for labels
    threshold : float
        faults with the size less then threshold will be removed
    sizes : numpy.ndarray or sizes
        precompured sizes of faults
    Returns
    -------
    numpy.ndarray
        filtered array
    """
    if sizes is None:
        sizes = faults_sizes(labels)
    indices = np.where(sizes >= threshold)[0] + 1
    return labels[np.isin(labels[:, 3], indices)]
