""" Horizon class and metrics. """

import numpy as np
import pandas as pd
from copy import copy
from numba import njit, prange

from PIL import ImageDraw, Image

from scipy.ndimage import find_objects
from scipy.interpolate import LinearNDInterpolator
from skimage.measure import label

from .horizon import Horizon
from .geometry import SeismicGeometry
from .utils import groupby_mean, groupby_min, groupby_max

class Fault(Horizon):
    """ !! """
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    COLUMNS = ['iline', 'xline', 'height', 'name', 'number']

    def file_to_points(self, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        with open(path) as file:
            line_len = len([item for item in file.readline().split(' ') if len(item) > 0])
        if line_len == 3:
            names = Horizon.REDUCED_CHARISMA_SPEC
        elif line_len == 8:
            names = self.FAULT_STICKS
        elif line_len >= 9:
            names = Horizon.CHARISMA_SPEC
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        df = pd.read_csv(path, sep='\s+', names=names)
        df = self.fix_sticks(df)[Horizon.COLUMNS]
        # df.sort_values(Horizon.COLUMNS, inplace=True)
        return self.interpolate_3d(df.values)

    def fix_sticks(self, df):
        def _move_stick_end(df):
            if (len(df.iline.unique()) > 1) and (len(df.xline.unique()) > 1):
                df['iline'] = df.iloc[0]['iline']
            return df

        if 'number' in df.columns:
            df = df.groupby('number').apply(_move_stick_end)
        return df

    def interpolate_points(self, points):
        transpose = self._check_sticks(points)
        if transpose:
            points = points[:, [1, 0, 2]]
        slides = np.unique(points[:, 0])
        x_min, x_max = points[:, 1].min(), points[:, 1].max()
        h_min, h_max = points[:, 2].min(), points[:, 2].max()

        _points = []
        for slide in slides:
            nodes = np.array(points[points[:, 0] == slide][:, 1:], dtype='int64')
            slide_points = _line(nodes, width=1)
            _points += [np.concatenate([np.ones((len(slide_points), 1)) * slide, slide_points], axis=1)]

        _points = np.concatenate(_points, axis=0)
        if transpose:
            _points = _points[:, [1, 0, 2]]
        return _points

    def _check_sticks(self, points):
        import pdb; pdb.set_trace()
        if points[0, 0] == points[1, 0]:
            return False
        if points[0, 1] == points[1, 1]:
            return True
        raise ValueError('Wrong sticks format')

    def interpolate_3d(self, points, axis=1):
        values = points[:, axis]
        coord = np.concatenate([points[:, :axis], points[:, axis+1:]], axis=1)
        interpolator = LinearNDInterpolator(coord, values)
        grid = np.meshgrid(np.arange(coord[:, 0].min()-10, coord[:, 0].max()+10),
                           np.arange(coord[:, 1].min()-10, coord[:, 1].max())+10)
        _values = interpolator(*grid)
        _coord = np.stack([grid[i][~np.isnan(_values)] for i in range(2)], axis=1)
        _values = _values[~np.isnan(_values)]
        _points = np.insert(_coord, axis, _values, 1)
        return _points

    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
        mask_bbox = np.array([[locations[0][0], locations[0][-1]+1],
                              [locations[1][0], locations[1][-1]+1],
                              [locations[2][0], locations[2][-1]+1]],
                             dtype=np.int32)
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        left =  width // 2
        right = width - left

        points = self.points
        for i in range(3):
            points = points[np.isin(points[:, i], locations[i])]
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

@njit
def _line(nodes, width=1):
    points = np.zeros((0, 2))
    for i in prange(len(nodes)-1):
        a = nodes[i]
        b = nodes[i+1]
        points = np.concatenate((points, _segment(a, b, width)))
    return points

@njit
def _segment(a, b, width=1):
    dx = np.abs(a[0] - b[0])
    dy = np.abs(a[1] - b[1])
    sx = 2 * (a[0] < b[0]) - 1
    sy = 2 * (a[1] < b[1]) - 1
    err = dx - dy

    n_points = max(dx, dy)
    points = np.zeros((n_points, 2))
    for i in range(n_points):
        points[i] = a
        e2 = 2 * err
        x = a[0]
        y = a[1]
        if e2 >= -dy:
            err -= dy
            x = x + sx
        if e2 <= dx:
            err += dx
            y = y + sy
        a = np.array([x, y])
    if dx > dy:
        delta = np.array([1, 0])
    else:
        delta = np.array([0, 1])

    shifted_points = np.zeros((len(points) * width, 2))
    start = 0
    for w in range(-width // 2 + 1, width // 2 + 1):
        end = start + len(points)
        shifted_points[start:end] = points + delta * w
        start = end

    return shifted_points
