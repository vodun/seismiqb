""" Fault class and processing methods. """

import os
import glob
from urllib.parse import non_hierarchical
import warnings

import numpy as np
import pandas as pd

from numba import prange, njit

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import measurements

from batchflow.notifier import Notifier

from .horizon import Horizon
from .fault_triangulation import sticks_to_simplices, simplices_to_points
from .fault_postprocessing import faults_sizes
from ..plotters import show_3d
from ..geometry import SeismicGeometry
from ..utils import concat_sorted, split_array



class FaultLoadingException(Exception): pass
class EmptySticksException(FaultLoadingException): pass

class Fault(Horizon):
    """ Contains points of fault.

    Initialized from `storage` and `geometry`, where storage can be one of:
        - csv-like file in CHARISMA or REDUCED_CHARISMA format.
        - npy file with ndarray of (N, 3) shape or array itself.
        - npz file with 'points', 'nodes', 'simplices' and 'sticks' or dict with the same keys.
    """
    #pylint: disable=attribute-defined-outside-init
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

    def __init__(self, storage, direction=None, *args, **kwargs):
        self._sticks = None
        self._nodes = None
        self._simplices = None
        self.direction = None

        if isinstance(storage, str):
            force_format = 'file'
        elif isinstance(storage, np.ndarray):
            force_format = 'points'
        elif isinstance(storage, dict):
            force_format = 'objects'

        super().__init__(storage, *args, force_format=force_format, reset=None, **kwargs)
        self.set_direction(direction)

    def set_direction(self, direction):
        """ Find azimuth of the fault. """
        if direction is None:
            if len(self.points) > 0:
                mean_depth = int(np.median(self.points[:, 2]))
                depth_slice = self.points[self.points[:, 2] == mean_depth]
                self.direction = 0 if depth_slice[:, 0].ptp() > depth_slice[:, 1].ptp() else 1
            else:
                self.direction = 0
            # TODO: azimuth from charisma
        elif isinstance(direction, int):
            self.direction = direction
        elif isinstance(direction[self.field.short_name], int):
            self.direction = direction[self.field.short_name]
        else:
            self.direction = direction[self.field.short_name][self.name]

    @property
    def sticks(self):
        return self._sticks

    @sticks.setter
    def sticks(self, value):
        self._sticks = value

    @property
    def nodes(self):
        return self._nodes

    @nodes.setter
    def nodes(self, value):
        self._nodes = value

    @property
    def simplices(self):
        return self._simplices

    @simplices.setter
    def simplices(self, value):
        self._simplices = value

    def from_objects(self, storage, **kwargs):
        """ Load fault from dict with 'points', 'nodes', 'simplices' and 'sticks'. """
        self.from_points(storage['points'], verify=False, **kwargs)
        for key in ['sticks', 'nodes', 'simplices']:
            setattr(self, key, storage.get(key))

    def from_file(self, path, transform=True, verify=True, direction=None, **kwargs):
        """ Init from path to either CHARISMA, REDUCED_CHARISMA or FAULT_STICKS csv-like file
        or from npy/npz.
        """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = os.path.basename(path)
        ext = os.path.splitext(path)[1][1:]

        if ext == 'npz':
            self.load_npz(path)
            self.format = 'file-npz'
        elif ext == 'npy':
            self.load_npy(path)
            self.format = 'file-npy'
        else:
            self.load_charisma(path, transform, verify, **kwargs)
            self.format = 'file-charisma'

    def load_charisma(self, path, transform=True, verify=True, fix=False, width=3, **kwargs):
        """ Get point cloud array from file values. """
        df = self.create_df(path)
        if 'cdp_x' in df.columns:
            df = self.recover_lines_from_cdp(df)

        points = df[self.REDUCED_CHARISMA_SPEC].values
        if transform:
            points = self.field_reference.geometry.lines_to_cubic(points)
        df[self.REDUCED_CHARISMA_SPEC] = points.astype(np.int32)

        if verify:
            idx = np.where((points[:, 0] >= 0) &
                        (points[:, 1] >= 0) &
                        (points[:, 2] >= 0) &
                        (points[:, 0] < self.field_reference.shape[0]) &
                        (points[:, 1] < self.field_reference.shape[1]) &
                        (points[:, 2] < self.field_reference.shape[2]))[0]
            df = df.iloc[idx]

        self.sticks = self.csv_to_sticks(df, fix)
        if len(self.sticks) > 0:
            self.simplices, self.nodes = sticks_to_simplices(self.sticks, return_indices=True)
            points = simplices_to_points(self.simplices, self.nodes, width=width)
        else:
            self.simplices, self.nodes, points = np.array([]), np.zeros((0, 3), dtype='int32'), np.zeros((0, 3), dtype='int32')
        self.from_points(points, verify=False)

    @classmethod
    def create_df(cls, path):
        with open(path, encoding='utf-8') as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])

        if line_len == 0:
            # self.simplices, self.nodes, self.points = np.array([]), np.zeros((0, 3), dtype='int32'), np.zeros((0, 3), dtype='int32')
            return pd.DataFrame({})

        if line_len == 3:
            names = cls.REDUCED_CHARISMA_SPEC
        elif line_len == 8:
            names = cls.FAULT_STICKS
        elif line_len >= 9:
            names = cls.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        return pd.read_csv(path, sep=r'\s+', names=names)

    def csv_to_sticks(self, df, fix=False):
        """ Transform initial fault dataframe to array of sticks. """
        if 'number' in df.columns: # fault file has stick index
            col = 'number'
        elif df.iline.iloc[0] == df.iline.iloc[1]: # there is stick points with the same iline
            col = 'iline'
        elif df.xline.iloc[0] == df.xline.iloc[1]: # there is stick points with the same xline
            col = 'xline'
        else:
            raise ValueError('Wrong format of sticks: there is no column to group points into sticks.')
        df = df.sort_values('height')
        sticks = df.groupby(col).apply(lambda x: x[Horizon.COLUMNS].values).reset_index(drop=True)
        if fix:
            # Remove sticks with horizontal parts.
            mask = sticks.apply(lambda x: len(np.unique(np.array(x)[:, 2])) == len(x))
            if not mask.all():
                warnings.warn(f'{self.name}: Fault has horizontal parts of sticks.')
            sticks = sticks.loc[mask]
            # Remove sticks with one node.
            mask = sticks.apply(len) > 1
            if not mask.all():
                warnings.warn(f'{self.name}: Fault has one-point sticks.')
            sticks = sticks.loc[mask]
            # Filter faults with one stick.
            if len(sticks) == 1:
                warnings.warn(f'{self.name}: Fault has an only one stick')
                sticks = pd.Series()
            elif len(sticks) == 0:
                warnings.warn(f'{self.name}: Empty file')
                sticks = pd.Series()
            #Order sticks with respect of fault direction. Is necessary to perform following triangulation.
            if len(sticks) > 0:
                pca = PCA(1)
                coords = pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values]))
                indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
                sticks = sticks.iloc[indices]
        return sticks.values

    def load_npz(self, path):
        """ Load fault points, nodes and sticks from npz file. """
        npzfile = np.load(path, allow_pickle=True) #TODO: remove allow_pickle
        self.from_points(npzfile['points'], verify=False)
        self.nodes = npzfile.get('nodes')
        self.simplices = npzfile.get('simplices')
        self.sticks = npzfile.get('sticks')

    def load_npy(self, path):
        """ Load fault points from npy file. """
        points = np.load(path)
        self.from_points(points, verify=False)

    def dump_points(self, path):
        """ Dump fault to npz. """
        path = self.field.make_path(path, name=self.short_name, makedirs=False)

        if os.path.exists(path):
            raise ValueError(f'{path} already exists.')

        folder_name = os.path.dirname(path)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        np.savez(path, points=self.points, nodes=self.nodes, simplices=self.simplices, sticks=self.sticks, allow_pickle=True) # TODO: what about allow_pickle?

    def points_to_sticks(self, slices=None, sticks_step=10, stick_nodes_step=10):
        """ Create sticks from fault points. """
        points = self.points.copy()
        if slices is not None:
            for i in range(3):
                points = points[points[:, i] <= slices[i].stop]
                points = points[points[:, i] >= slices[i].start]
        if len(points) <= 3:
            return [], [], [], []
        self.sticks = get_sticks(points, sticks_step, stick_nodes_step)
        self.simplices, self.nodes = sticks_to_simplices(self.sticks, return_indices=True)

    def add_to_mask(self, mask, locations=None, width=1, **kwargs):
        """ Add fault to background. """
        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                            [locations[1].start, locations[1].stop],
                            [locations[2].start, locations[2].stop]],
                            dtype=np.int32)
        points = self.points

        if (self.bbox[:, 1] < mask_bbox[:, 0]).any() or (self.bbox[:, 0] >= mask_bbox[:, 1]).any():
            return mask

        insert_fault_into_mask(mask, points, mask_bbox, width=width, axis=1-self.direction)
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
                df = cls.create_df(filename)
            except ValueError:
                print(filename, ': wrong format')
            else:
                if 'name' in df.columns and len(df.name.unique()) > 1:
                    print(filename, ': file must be splitted.')
                elif len(cls.csv_to_sticks(cls, df)) == 1:
                    print(filename, ': fault has an only one stick')
                elif any([len(item) == 1 for item in cls.csv_to_sticks(cls, df)]):
                    print(filename, ': fault has one point stick')
                elif verbose:
                    print(filename, ': OK')

    @classmethod
    def split_file(cls, path, dst):
        """ Split file with multiple faults into separate files. """
        if dst and not os.path.isdir(dst):
            os.makedirs(dst)
        df = pd.read_csv(path, sep=r'\s+', names=cls.FAULT_STICKS)
        df.groupby('name').apply(cls.fault_to_csv, dst=dst)

    @classmethod
    def fault_to_csv(cls, df, dst):
        """ Save the fault to csv. """
        df.to_csv(os.path.join(dst, df.name), sep=' ', header=False, index=False)

    def show_3d(self, sticks_step=10, stick_nodes_step=10, z_ratio=1., zoom_slice=None, show_axes=True,
                width=1200, height=1200, margin=20, savepath=None, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        sticks_step : int
            Number of slides between sticks.
        stick_nodes_step : int
            Distance between stick nodes
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom_slice : tuple of slices or None.
            Crop from cube to show. If None, the whole cube volume will be shown.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : int
            Size of the image.
        margin : int
            Added margin from below and above along height axis.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Fault `{self.name}` on `{self.field.displayed_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')
        if zoom_slice is None:
            zoom_slice = [slice(0, i) for i in self.field.shape]
        zoom_slice[-1] = slice(self.h_min, self.h_max)
        margin = [margin] * 3 if isinstance(margin, int) else margin
        x, y, z, simplices = self.make_triangulation(zoom_slice, sticks_step, stick_nodes_step)

        show_3d(x, y, z, simplices, title, zoom_slice, None, show_axes, aspect_ratio,
                axis_labels, width, height, margin, savepath, **kwargs)

    def make_triangulation(self, slices, sticks_step, stick_nodes_step, *args, **kwargs):
        """ Return triangulation of the fault. It will created if needed. """
        if self.simplices is None:
            self.points_to_sticks(slices, sticks_step, stick_nodes_step)
        return self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], self.simplices


    def __repr__(self):
        return f"""<Fault `{self.name}` for `{self.field.displayed_name}` at {hex(id(self))}>"""

def get_sticks(points, sticks_step, stick_nodes_step):
    """ Get sticks from fault which is represented as a cloud of points.

    Parameters
    ----------
    points : np.ndarray
        Fault points.
    sticks_step : int
        Number of slides between sticks.
    stick_nodes_step : int
        Distance between stick nodes

    Returns
    -------
    numpy.ndarray
        Array of sticks. Each item of array is a stick: sequence of 3D points.
    """
    pca = PCA(1)
    pca.fit(points)
    axis = 0 if np.abs(pca.components_[0][0]) > np.abs(pca.components_[0][1]) else 1

    points = points[np.argsort(points[:, axis])]
    projections = np.split(points, np.unique(points[:, axis], return_index=True)[1][1:])
    projections = [item for item in projections if item[:, 2].max() - item[:, 2].min() > 5]
    step = min(sticks_step, len(projections)-1)
    if step == 0:
        return []
    projections = projections[::step]
    res = []

    for p in projections:
        points_ = thicken_line(p).astype(int)
        loc = p[0, axis]
        nodes = approximate_points(points_[:, [1-axis, 2]], stick_nodes_step)
        nodes_ = np.zeros((len(nodes), 3))
        nodes_[:, [1-axis, 2]] = nodes
        nodes_[:, axis] = loc
        res += [nodes_]
    return res

def thicken_line(points):
    """ Make thick line. """
    points = points[np.argsort(points[:, -1])]
    splitted = split_array(points, points[:, -1])
    return np.stack([np.mean(item, axis=0) for item in splitted], axis=0)

def approximate_points(points, n_points):
    """ Approximate points by stick. """
    pca = PCA(1)
    array = pca.fit_transform(points)

    # step = (array.max() - array.min()) / (n_points - 1)
    step = n_points
    initial = np.arange(array.min(), array.max() + step / 2, step)
    indices = np.unique(nearest_neighbors(initial.reshape(-1, 1), array.reshape(-1, 1), 1))
    return points[indices]

def nearest_neighbors(values, all_values, n_neighbors=10):
    """ Find nearest neighbours for each `value` items in `all_values`. """
    nn = NearestNeighbors(n_neighbors=n_neighbors).fit(all_values)
    return nn.kneighbors(values)[1].flatten()

@njit(parallel=True)
def insert_fault_into_mask(mask, points, mask_bbox, width, axis):
    """ Add new points into binary mask. """
    #pylint: disable=not-an-iterable

    for i in prange(len(points)):
        point = points[i]
        if (point[0] >= mask_bbox[0][0]) and (point[0] < mask_bbox[0][1]):
            if (point[1] >= mask_bbox[1][0]) and (point[1] < mask_bbox[1][1]):
                if (point[2] >= mask_bbox[2][0]) and (point[2] < mask_bbox[2][1]):
                    slices = [slice(point[j] - mask_bbox[j][0], point[j] - mask_bbox[j][0]+1) for j in range(3)]
                    if width > 1:
                        slices[axis] = slice(
                            max(0, point[axis] - mask_bbox[axis][0] - (width // 2)),
                            min(point[axis] - mask_bbox[axis][0] + width // 2 + width % 2, mask.shape[axis])
                        )
                    mask[slices[0], slices[1], slices[2]] = 1
