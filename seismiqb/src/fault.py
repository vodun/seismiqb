""" Horizon class and metrics. """

import numpy as np
import pandas as pd
from PIL import ImageDraw, Image

from .horizon import Horizon

class Fault(Horizon):
    """ !! """
    FAULT_STICKS = ['INLINE', 'iline', 'xline', 'cdp_x', 'cdp_y', 'height', 'name', 'number']
    def file_to_points(self, path):
        """ Get point cloud array from file values. """
        #pylint: disable=anomalous-backslash-in-string
        with open(path) as file:
            line_len = len(file.readline().split(' '))
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
            line = points[points[:, 0] == slide][:, 1:]
            line = line - np.array([x_min, h_min]).reshape(-1, 2)
            img = Image.new('L', (int(x_max - x_min), int(h_max - h_min)), 0)
            ImageDraw.Draw(img).line(list(line.ravel()), width=1, fill=1)
            slide_points = np.stack(np.where(np.array(img).T), axis=1) + np.array([x_min, h_min]).reshape(-1, 2)
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
