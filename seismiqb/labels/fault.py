""" Fault class and processing methods. """

import os
import glob
import warnings

import numpy as np
import pandas as pd

from numba import prange, njit

from sklearn.decomposition import PCA

from .horizon import Horizon
from .fault_triangulation import sticks_to_simplices, simplices_to_points
from .fault_postprocessing import split_array, thin_line
from .fault_approximation import points_to_sticks
from .mixins import VisualizationMixin, CoordinatesMixin
from ..plotters import show_3d
from ..utils import CharismaMixin, make_interior_points_mask

class FaultSticksMixin(CharismaMixin):
    FAULT_STICKS_SPEC = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    REDUCED_FAULT_STICKS_SPEC = ['iline', 'xline', 'height', 'name', 'number']

    @classmethod
    def read_df(cls, path):
        """ Create pandas.DataFrame from FaultSticks/CHARISMA file. """
        with open(path, encoding='utf-8') as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])

        if line_len == 0:
            return pd.DataFrame({})

        if line_len == 3:
            names = cls.REDUCED_CHARISMA_SPEC
        elif line_len == 5:
            names = cls.REDUCED_FAULT_STICKS_SPEC
        elif line_len == 8:
            names = cls.FAULT_STICKS_SPEC
        elif line_len >= 9:
            names = cls.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        return pd.read_csv(path, sep=r'\s+', names=names)

    def df_to_sticks(self, df, return_direction=False):
        """ Transform initial fault dataframe to array of sticks. """
        if len(df) == 0:
            raise ValueError('Empty DataFrame (possibly wrong coordinates).')
        if 'number' in df.columns: # Dataframe has stick index
            col = 'number'
            direction = None
        elif df.iline.iloc[0] == df.iline.iloc[1]: # Use iline as an index
            col = 'iline'
            direction = 0
        elif df.xline.iloc[0] == df.xline.iloc[1]: # Use xline as an index
            col = 'xline'
            direction = 1
        else:
            raise ValueError('Wrong format of sticks: there is no column to group points into sticks.')

        df = df.sort_values('height')
        sticks = df.groupby(col).apply(lambda x: x[Horizon.COLUMNS].values).reset_index(drop=True)

        if return_direction:
            return sticks, direction
        else:
            return sticks

    def remove_broken_sticks(self, sticks):
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

        return sticks

    def load_fault_sticks(self, path, transform=True, verify=True,
                          recover_lines=True, remove_broken_sticks=False, **kwargs):
        """ Get point cloud array from file values. """
        df = self.read_df(path)

        if len(df) == 0:
            return

        if recover_lines and 'cdp_x' in df.columns:
            df = self.recover_lines_from_cdp(df)

        points = df[self.REDUCED_CHARISMA_SPEC].values

        if transform:
            points = self.field_reference.geometry.lines_to_cubic(points)
        df[self.REDUCED_CHARISMA_SPEC] = np.round(points).astype(np.int32)

        if verify:
            mask = make_interior_points_mask(points, self.field_reference.shape)
            df = df.iloc[mask]

        sticks, direction = self.df_to_sticks(df, return_direction=True)
        if remove_broken_sticks:
            sticks = self.remove_broken_sticks(sticks)

        # Order sticks with respect of fault direction. Is necessary to perform following triangulation.
        if len(sticks) > 0:
            pca = PCA(1)
            coords = pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values]))
            indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
            sticks = sticks.iloc[indices]

        self._sticks = sticks.values
        self.direction = direction

    def dump_fault_sticks(self, path, sticks_step=10, stick_nodes_step=10):
        """ Dump fault sticks. """
        path = self.field.make_path(path, name=self.field.short_name, makedirs=False)

        sticks_df = []
        for stick_idx, stick in enumerate(self.sticks):
            stick = self.field.geometry.cubic_to_lines(stick).astype(int)
            cdp = self.field.geometry.lines_to_cdp(stick[:, :2])
            df = {
                'INLINE-': 'INLINE-',
                'iline': stick[:, 0],
                'xline': stick[:, 1],
                'cdp_x': cdp[:, 0],
                'cdp_y': cdp[:, 1],
                'height': stick[:, 2],
                'name': os.path.basename(path),
                'number': stick_idx
            }
            sticks_df.append(pd.DataFrame(df))
        sticks_df = pd.concat(sticks_df)
        sticks_df.to_csv(path, header=False, index=False, sep=' ')

    def show_file(self):
        with open(self.path, 'r') as f:
            print(f.read())

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
            if os.path.splitext(filename)[1] == '.dvc':
                continue
            try:
                df = cls.read_df(filename)
                sticks = cls.df_to_sticks(cls, df)
            except ValueError:
                print(filename, ': wrong format')
            else:
                if 'name' in df.columns and len(df.name.unique()) > 1:
                    print(filename, ': file must be splitted.')
                    continue

                if len(sticks) == 1:
                    print(filename, ': fault has an only one stick')
                    continue

                if any(len(item) == 1 for item in sticks):
                    print(filename, ': fault has one point stick')
                    continue
                mask = sticks.apply(lambda x: len(np.unique(np.array(x)[:, 2])) == len(x))
                if not mask.all():
                    print(filename, ': fault has horizontal parts of sticks.')
                    continue

                if verbose:
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

class FaultSerializationMixin:
    def load_npz(self, path):
        """ Load fault points, nodes and sticks from npz file. """
        npzfile = np.load(path, allow_pickle=False)

        sticks = npzfile.get('sticks')
        sticks_labels = npzfile.get('sticks_labels')

        self.from_objects({
            'points': npzfile['points'],
            'nodes': npzfile.get('nodes'),
            'simplices': npzfile.get('simplices'),
            'sticks': self.labeled_array_to_sticks(sticks, sticks_labels),
        })

        self.direction = npzfile.get('direction')

    def load_npy(self, path):
        """ Load fault points from npy file. """
        points = np.load(path, allow_pickle=False)
        self._points = points

    def dump_points(self, path):
        """ Dump fault to npz. """
        path = self.field.make_path(path, name=self.field.short_name, makedirs=False)

        if self.has_component('sticks'):
            sticks, sticks_labels = self.sticks_to_labeled_array(self.sticks)
        else:
            sticks, sticks_labels = np.zeros((0, 3)), np.zeros((0, 1))

        np.savez(path, points=self._points, nodes=self._nodes, simplices=self._simplices,
                 sticks=sticks, sticks_labels=sticks_labels)#, direction=self.direction)


    def sticks_to_labeled_array(self, sticks):
        """ Auxilary method to dump fault into npz with allow_pickle=False. """
        labels = sum([[i] * len(item) for i, item in enumerate(sticks)], [])
        return np.concatenate(sticks), labels

    def labeled_array_to_sticks(self, sticks, labels):
        """ Auxilary method to dump fault into npz with allow_pickle=False. """
        return np.array(split_array(sticks, labels), dtype=object)


class FaultVisualizationMixin(VisualizationMixin):
    def __repr__(self):
        return f"""<Fault `{self.name}` for `{self.field.displayed_name}` at {hex(id(self))}>"""

    def show_slide(self, loc, **kwargs):
        cmap = kwargs.get('cmap', ['Greys_r', 'red'])
        width = kwargs.get('width', 5)

        kwargs = {**kwargs, 'cmap': cmap, 'width': width}
        super().show_slide(loc, **kwargs)

    def show(self, axis=0, centering=True, zoom=None, **kwargs):
        if centering and zoom is None:
            zoom = [slice(max(0, self.bbox[i][0]-20), min(self.bbox[i][1]+20, self.field.shape[i])) for i in range(3) if i != axis]

        return self.show_slide(loc=int(np.mean(self.bbox[axis])), zoom=zoom, axis=axis, **kwargs)

    def show_3d(self, sticks_step=10, stick_nodes_step=10, z_ratio=1., colors='green',
                zoom=None, margin=20, **kwargs):
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
        zoom : tuple of slices or None.
            Crop from cube to show. If None, the whole cube volume will be shown.
        show_axes : bool
            Whether to show axes and their labels, by default True
        width, height : int
            Size of the image, by default 1200, 1200
        margin : int
            Added margin from below and above along height axis, by default 20
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Fault `{self.name}` on `{self.field.displayed_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')

        if zoom is None:
            zoom = [slice(0, i) for i in self.field.shape]
        zoom[-1] = slice(self.h_min, self.h_max+1)
        margin = [margin] * 3 if isinstance(margin, int) else margin
        x, y, z, simplices = self.make_triangulation(zoom, sticks_step, stick_nodes_step)
        if isinstance(colors, str):
            colors = [colors for _ in simplices]

        show_3d(x, y, z, simplices, title=title, zoom=zoom, aspect_ratio=aspect_ratio,
                axis_labels=axis_labels, margin=margin, colors=colors, **kwargs)

    def make_triangulation(self, slices, sticks_step, stick_nodes_step, *args, **kwargs):
        """ Return triangulation of the fault. It will created if needed. """
        if len(self.simplices) > 0:
            return self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], self.simplices
        fake_fault = get_fake_one_stick_fault(self)
        return fake_fault.make_triangulation(slices, sticks_step, stick_nodes_step, *args, **kwargs)

class Fault(FaultSticksMixin, FaultSerializationMixin, CoordinatesMixin, FaultVisualizationMixin):
    def __init__(self, storage, field, name=None, direction=None, **kwargs):
        self.name = name
        self.field = field

        self._points = None
        self._sticks = None
        self._nodes = None
        self._simplices = None
        self.direction = None

        if isinstance(storage, str):
            format = 'file'
        elif isinstance(storage, np.ndarray):
            format = 'points'
        elif isinstance(storage, dict):
            format = 'objects'
        getattr(self, f'from_{format}')(storage, **kwargs)

        # if self.direction is None:
        self.set_direction(direction)
        self.create_stats()

    def filter(self):
        pass

    def has_component(self, component):
        return getattr(self, '_'+component) is not None

    def create_stats(self):
        if self.has_component('points'):
            data = self.points
        elif self.has_component('nodes'):
            data = self.nodes
        else:
            data = np.concatenate(self.sticks)

        i_min, x_min, h_min = np.min(data, axis=0)
        i_max, x_max, h_max = np.max(data, axis=0)

        self.h_min, self.h_max = int(h_min), int(h_max)
        self.i_min, self.i_max, self.x_min, self.x_max = int(i_min), int(i_max), int(x_min), int(x_max)

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1
        self.bbox = np.array([[self.i_min, self.i_max],
                            [self.x_min, self.x_max],
                            [self.h_min, self.h_max]],
                            dtype=np.int32)

    def set_direction(self, direction):
        """ Find azimuth of the fault. """
        if direction is None:
            if self._sticks is not None and len(self._sticks) > 0:
                if len(np.unique(self._sticks[0][:, 0])) == 1:
                    self.direction = 0
                elif len(np.unique(self._sticks[0][:, 1])) == 1:
                    self.direction = 1
            if self.direction is None and len(self.points) > 0:
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

    def points_to_sticks(self, slices=None, sticks_step=10, stick_nodes_step=10):
        """ Create sticks from fault points. """
        points = self.points.copy()
        if slices is not None:
            for i in range(3):
                points = points[points[:, i] <= slices[i].stop]
                points = points[points[:, i] >= slices[i].start]
        if len(points) <= 3:
            self.sticks = []
        else:
            self.sticks = self.get_sticks(points, sticks_step, stick_nodes_step)

    def from_file(self, path, **kwargs):
        """ Init from path to either CHARISMA, REDUCED_CHARISMA or FAULT_STICKS csv-like file
        or from npy/npz.
        """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = os.path.basename(path)
        self.short_name = os.path.splitext(self.name)[0]

        ext = os.path.splitext(path)[1][1:]

        if ext == 'npz':
            self.load_npz(path)
            self.format = 'file-npz'
        elif ext == 'npy':
            self.load_npy(path)
            self.format = 'file-npy'
        else:
            self.load_fault_sticks(path, **kwargs)
            self.format = 'file-sticks'

    def from_objects(self, storage, **kwargs):
        """ Load fault from dict with 'points', 'nodes', 'simplices' and 'sticks'. """
        for key in ['points', 'sticks', 'nodes', 'simplices']:
            setattr(self, '_' + key, storage.get(key))

    # Transformation of attributes: sticks -> (nodes, simplices) -> points -> sticks

    @property
    def simplices(self):
        if self._simplices is None:
            if self._points is None and self._sticks is None:
                raise AttributeError("`simplices` can't be created.")

            self._simplices, self._nodes = sticks_to_simplices(self.sticks, return_indices=True)

        return self._simplices

    @property
    def nodes(self):
        if self._nodes is None:
            if self._points is None and self._sticks is None:
                raise AttributeError("`nodes` can't be created.")

            self._simplices, self._nodes = sticks_to_simplices(self.sticks, return_indices=True)

        return self._nodes

    @property
    def points(self):
        if self._points is None:
            if self._simplices is None and self._sticks is None:
                raise AttributeError("`points` can't be created.")
            if len(self.simplices) > 1:
                self._points = simplices_to_points(self.simplices, self.nodes, width=1)
            elif len(self.nodes) > 0:
                fake_fault = get_fake_one_stick_fault(self)
                points = fake_fault.points
                self._points = points[points[:, self.direction] == self.sticks[0][0, self.direction]]

        return self._points

    @property
    def sticks(self):
        if self._sticks is None:
            if self._simplices is None and self._points is None:
                raise AttributeError("`points` can't be created.")
            self._sticks = points_to_sticks(self.points, axis=self.direction)

        return self._sticks

    def add_to_mask(self, mask, locations=None, width=1, **kwargs):
        """ Add fault to background. """
        _ = kwargs
        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                            [locations[1].start, locations[1].stop],
                            [locations[2].start, locations[2].stop]],
                            dtype=np.int32)
        points = self.points

        if (self.bbox[:, 1] < mask_bbox[:, 0]).any() or (self.bbox[:, 0] >= mask_bbox[:, 1]).any():
            return mask

        insert_fault_into_mask(mask, points, mask_bbox, width=width, axis=1-self.direction)
        return mask

    def __len__(self):
        """ Number of labeled traces. """
        # TODO
        # return np.prod(self.bbox[:, 1] - self.bbox[:, 0] + 1)
        return len(self.points)
        # if self._len is None:
        #     if self._points is not None:
        #         self._len = len(self.points)
        #     else:
        #         self._len = len(self.depths)
        # return self._len

# TODO: points_to_sticks at region

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

def get_fake_one_stick_fault(fault):
    if len(set(fault.sticks[0][:, 0])) > 1 and len(set(fault.sticks[0][:, 1])) > 1:
        raise ValueError('!!')
    stick = fault.sticks[0]
    stick_2 = stick.copy()
    loc = stick[0, fault.direction]
    stick_2[:, fault.direction] = loc - 1 if loc >= 1 else loc + 1

    fake_fault = Fault({'sticks': np.array([stick, stick_2])}, direction=fault.direction,
                       field=fault.field)

    return fake_fault