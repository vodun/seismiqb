""" Horizon class and metrics. """

import numpy as np
import pandas as pd
from copy import copy

from PIL import ImageDraw, Image

from scipy.ndimage import find_objects
from skimage.measure import label

from .horizon import Horizon
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
        df.sort_values(Horizon.COLUMNS, inplace=True)
        return self.interpolate_points(df.values)

    def fix_sticks(self, df):
        def _move_stick_end(df):
            if len(df.iline.unique()) > 1:
                df['iline'] = df.iloc[0]['iline']
            return df

        if 'number' in df.columns:
            df = df.groupby('number').apply(_move_stick_end)
        return df

    def interpolate_points(self, points):
        slides = np.unique(points[:, 0])
        x_min, x_max = points[:, 1].min(), points[:, 1].max()
        h_min, h_max = points[:, 2].min(), points[:, 2].max()

        _points = []
        for slide in slides:
            slide_points = line(points[points[:, 0] == slide][:, 1:], width=1)
            # line = points[points[:, 0] == slide][:, 1:]
            # line = line - np.array([x_min, h_min]).reshape(-1, 2)
            # img = Image.new('L', (int(x_max - x_min), int(h_max - h_min)), 0)
            # ImageDraw.Draw(img).line(list(line.ravel()), width=1, fill=1)
            # slide_points = np.stack(np.where(np.array(img).T), axis=1) + np.array([x_min, h_min]).reshape(-1, 2)
            _points += [np.concatenate([np.ones((len(slide_points), 1)) * slide, slide_points], axis=1)]

        return np.concatenate(_points, axis=0)

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
        """ !! """
        ilines = np.unique(points[:, 0])
        ilines = ilines[::i_step]
        all_sticks = []
        n_stick = 0
        for il in ilines:
            curr = points[points[:, 0] == il]
            length = len(curr)
            step = length // num
            if step == 0:
                continue
            selected = np.vstack([curr[::step], curr[-1].reshape(1, -1)])
            sticks = np.hstack([selected, np.array([n_stick] * len(selected)).reshape(-1, 1)])
            all_sticks.append(sticks)
            n_stick += 1
        return np.vstack(all_sticks).reshape(-1, 4)


    def dump(self, path, i_step=1, num=5):
        """ Save Fault points to the disk in CHARISMA Fault Sticks format.
        """
        fault_sticks = self.points_to_sticks(copy(self.points), i_step, num)
        fault_sticks = np.hstack([fault_sticks[:, :-1],
                                  np.array([self.name] * len(fault_sticks)).reshape(-1, 1),
                                  fault_sticks[:, -1:]])
        df = pd.DataFrame(fault_sticks, columns=self.COLUMNS)
        path = f'{path}_{self.name}'
        df.to_csv(path, sep=' ', columns=self.COLUMNS, index=False, header=False)

def line(nodes, width=1):
    nodes = np.array(nodes, dtype='int32')
    a = nodes[0]
    points = []
    for b in nodes[1:]:
        points += [segment(a, b, width)]
        a = b
    return np.concatenate(points)

def segment(a, b, width=1):
    dx = np.abs(a[0] - b[0])
    dy = np.abs(a[1] - b[1])
    sx = 2 * (a[0] < b[0]) - 1
    sy = 2 * (a[1] < b[1]) - 1
    err = dx - dy

    points = []
    while (a != b).any():
        points += [a]
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
    points = np.array(points)
    if dx > dy:
        delta = np.array([1, 0])
    else:
        delta = np.array([0, 1])
    return np.concatenate([points + delta * w for w in range(-width // 2 + 1, width // 2 + 1)])