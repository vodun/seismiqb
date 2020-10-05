""" Horizon class and metrics. """

import os
import glob

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA

from .horizon import Horizon
from .triangulation import triangulation, triangle_rasterization

class Fault(Horizon):
    """ !! """
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

    def __init__(self, *args, **kwargs):
        self._max = None
        self._min = None
        super().__init__(*args, **kwargs)

    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA or REDUCED_CHARISMA csv-like file
        or from .npy file with points. """
        _ = kwargs
        self.path = path
        self.name = os.path.basename(path)
        ext = os.path.splitext(path)[1]
        if ext == '.npy':
            points = np.load(path, allow_pickle=True)
            self.from_points(points, False, **kwargs)
        else:
            points = self.csv_to_points(path, **kwargs)
            self.from_points(points, transform, **kwargs)
        self._max = self.points.max(axis=0)
        self._min = self.points.min(axis=0)

    def csv_to_points(self, path, **kwargs):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        df = self.read_file(path)
        df = self.fix_lines(df)
        sticks = self.read_sticks(df)
        sticks = self.sort_sticks(sticks)
        points = self.interpolate_3d(sticks, **kwargs)
        return points

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

        if (self._max < mask_bbox[:, 0]).any() or (self._min >= mask_bbox[:, 1]).any():
            return mask
        cond = np.ones(len(points))
        for i in range(3):
            _cond = np.logical_and(points[:, i] >= locations[i].start, points[:, i] < locations[i].stop)
            cond = np.logical_and(cond, _cond)
        points = points[cond]
        points = points - np.array(mask_bbox[:, 0]).reshape(1, 3)
        points = np.maximum(points, 0)
        points = np.minimum(points, np.array(mask.shape) - 1)
        mask[points[:, 0], points[:, 1], points[:, 2]] = 1
        return mask

    def fix_lines(self, df):
        """ Fix broken iline and crossline coordinates. """
        i_bounds = [self.geometry.ilines_offset, self.geometry.ilines_offset + self.geometry.cube_shape[0]]
        x_bounds = [self.geometry.xlines_offset, self.geometry.xlines_offset + self.geometry.cube_shape[1]]

        i_mask = np.logical_or(df.iline < i_bounds[0], df.iline >= i_bounds[1])
        x_mask = np.logical_or(df.xline < x_bounds[0], df.xline >= x_bounds[1])

        _df = df[np.logical_and(i_mask, x_mask)]

        df.loc[np.logical_and(i_mask, x_mask), ['iline', 'xline']] = np.rint(
            self.geometry.cdp_to_lines(_df[['cdp_x', 'cdp_y']].values)
        ).astype('int32')

        return df

    def sort_sticks(self, sticks):
        """ Sort sticks with respect of fault direction. """
        pca = PCA(1)
        coords = pca.fit_transform(pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values])))
        indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
        return sticks.iloc[indices]

    def dump_points(self, path, fmt='npy'):
        """ Dump interpolated fault points. """
        if fmt == 'npy':
            self.points.dump(path)
        else:
            raise ValueError('Unknown format:', fmt)

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
        """ Split file with multiple faults into separate. """
        folder = os.path.dirname(path)
        faults_folder = os.path.join(folder, dst)
        if faults_folder and not os.path.isdir(faults_folder):
            os.makedirs(faults_folder)
        df = pd.read_csv(path, sep='\s+', names=cls.FAULT_STICKS)
        def _dump(df):
            df.to_csv(os.path.join(folder, dst, df.name), sep=' ', header=False, index=False)
        df.groupby('name').apply(_dump)
