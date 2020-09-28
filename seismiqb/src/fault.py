""" Horizon class and metrics. """

import sys
import os

import numpy as np
import pandas as pd
from copy import copy
from numba import njit, prange

from PIL import ImageDraw, Image

from scipy.ndimage import find_objects
from scipy.interpolate import LinearNDInterpolator, griddata
from sklearn.decomposition import PCA

from .horizon import Horizon
from .geometry import SeismicGeometry, SeismicGeometrySEGY
from .utils import groupby_mean, groupby_min, groupby_max
from .triangulation import triangulation, triangle_rasterization

class Fault(Horizon):
    """ !! """
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

    def file_to_points(self, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        df = self.read_file(path)
        df = self.fix_lines(df)
        sticks = self.sticks(df)
        sticks = self.sort_sticks(sticks)
        points = self.interpolate_3d(sticks)
        return points

    @classmethod
    def sticks(cls, df):
        if 'number' in df.columns:
            col = 'number'
        elif df.iline.iloc[0] == df.iline.iloc[1]:
            col = 'iline'
        elif df.xline.iloc[0] == df.xline.iloc[1]:
            col = 'xline'
        else:
            raise ValueError('!!!!')
        return df.groupby(col).apply(lambda x: x[Horizon.COLUMNS].values).reset_index(drop=True)


    @classmethod
    def read_file(cls, path):
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

        return pd.read_csv(path, sep='\s+', names=names)

    def interpolate_3d(self, sticks):
        triangles = triangulation(sticks)
        points = []
        for triangle in triangles:
            res = triangle_rasterization(triangle, width=5)
            points += [res]
        return np.concatenate(points, axis=0)

    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
        mask_bbox = np.array([[locations[0].start, locations[0].stop],
                              [locations[1].start, locations[1].stop],
                              [locations[2].start, locations[2].stop]],
                             dtype=np.int32)
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        left =  width // 2
        right = width - left

        points = self.points
        for i in range(3):
            positions = np.arange(locations[i].stop)[locations[i]]
            points = points[np.isin(points[:, i], positions)]
        def _extend_line(points):
            _points = []
            for x in range(-left, right):
                for y in range(-left, right):
                    arr = points + np.array([x, y, 0]).reshape(1, 3)
                    _points.append(arr)
            return np.concatenate(_points)

        points = _extend_line(points)
        points = points - np.array([mask_i_min, mask_x_min, mask_h_min]).reshape(1, 3)
        points = np.maximum(points, 0)
        points = np.minimum(points, np.array(mask.shape) - 1)

        mask[points[:, 0], points[:, 1], points[:, 2]] = 1
        return mask

    def points_to_sticks(self, points, i_step, num):
        """ Extract iline oriented fault sticks from a solid fault surface.
        """
        ilines = np.unique(points[:, 0])
        ilines = ilines[::i_step]
        all_sticks = []
        n_stick = 0
        for il in ilines:
            curr = points[points[:, 0] == il]
            curr = curr[curr[:, -1].argsort()]
            length = len(curr)
            step = max(1, length // num)
            selected = np.vstack([curr[::step], curr[-1].reshape(1, -1)])
            sticks = np.hstack([selected, np.array([n_stick] * len(selected)).reshape(-1, 1)])
            all_sticks.append(sticks)
            n_stick += 1
        return np.vstack(all_sticks).reshape(-1, 4)


    def dump(self, path, sgy_path, i_step=1, num=5):
        """ Save Fault to the disk in CHARISMA Fault Sticks format.
        Note that this version supports iline oriented fault sticks.

        Parameters
        ----------
        path : str
            Path to a file to save fault to.
        sgy_path : str
            Path to SEG-Y version of the cube with `cdp_x`, `cdp_y` headers needed to compute
            cdp coordinates of the fault.
        i_step : int
            Ilines dump frequency allowing to save sparse fault surfaces.
        num : int
            A number of points for each stick.
        """
        points = self.cubic_to_lines(copy(self.points))
        fault_sticks = self.points_to_sticks(points, i_step, num)
        fault_sticks = np.hstack([fault_sticks[:, :-1],
                                  np.array([self.name] * len(fault_sticks)).reshape(-1, 1),
                                  fault_sticks[:, -1:]])
        df = pd.DataFrame(fault_sticks, columns=self.COLUMNS)

        if self.geometry.xline_to_cdpy is None:
            self.geometry.compute_cdp_transform(sgy_path)

        df.insert(loc=0, column='INLINE-', value=['INLINE-'] *len(df))
        df.insert(loc=3, column='cdp_y', value=self.geometry.xline_to_cdpy(pd.to_numeric(df['xline'])))
        df.insert(loc=3, column='cdp_x', value=self.geometry.iline_to_cdpx(pd.to_numeric(df['iline'])))

        path = f'{path}_{self.name}'
        df.to_csv(path, sep=' ', index=False, header=False)

    def fix_lines(self, df):
        " Fix broken iline and crossline coordinates. "
        i_bounds = [self.geometry.ilines_offset, self.geometry.ilines_offset + self.geometry.cube_shape[0]]
        x_bounds = [self.geometry.xlines_offset, self.geometry.xlines_offset + self.geometry.cube_shape[1]]

        i_mask = np.logical_or(df.iline < i_bounds[0], df.iline >= i_bounds[1])
        x_mask = np.logical_or(df.xline < x_bounds[0], df.xline >= x_bounds[1])

        _df = df[np.logical_and(i_mask, x_mask)]

        df.loc[np.logical_and(i_mask, x_mask), ['iline', 'xline']] = np.rint(self.geometry.cdp_to_lines(_df[['cdp_x', 'cdp_y']].values)).astype('int32')

        return df

    def sort_sticks(self, sticks):
        """ Sort sticks with respect of fault direction. """
        pca = PCA(1)
        coords = pca.fit_transform(pca.fit_transform(np.array([stick[0][:2] for stick in sticks.values])))
        indices = np.array([i for _, i in sorted(zip(coords, range(len(sticks))))])
        return sticks.iloc[indices]

    @classmethod
    def check_format(cls, path, verbose=False):
        try:
            df = cls.read_file(path)
        except ValueError:
            print(path, ': wrong format')
        else:
            if 'name' in df.columns and len(df.name.unique()) > 1:
                print(path, ': fault file must be splitted.')
            elif len(cls.sticks(df)) == 1:
                print(path, ': fault has an only one stick')
            elif any(cls.sticks(df).apply(len) == 1):
                print(path, ': fault has one point stick')
            elif verbose:
                print(path, ': OK')

    @classmethod
    def split_file(cls, path, dst='faults'):
        folder = os.path.dirname(path)
        faults_folder = os.path.join(folder, dst)
        if faults_folder and not os.path.isdir(faults_folder):
            os.makedirs(faults_folder)
        df = pd.read_csv(path, sep='\s+', names=cls.FAULT_STICKS)
        def _dump(df):
            df.to_csv(os.path.join(folder, dst, df.name), sep=' ', header=False, index=False)
        df.groupby('name').apply(_dump)
