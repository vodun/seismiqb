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
            names = Horizon.CHARISMA
        else:
            raise ValueError('Fault labels must be in FAULT_STICKS, CHARISMA or REDUCED_CHARISMA format.')

        df = pd.read_csv(path, sep='\s+', names=names, usecols=Horizon.COLUMNS)
        df.sort_values(Horizon.COLUMNS, inplace=True)
        return df.values

    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
        mask_bbox = np.array([[locations[0][0], locations[0][-1]+1],
                              [locations[1][0], locations[1][-1]+1],
                              [locations[2][0], locations[2][-1]+1]],
                             dtype=np.int32)
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox
        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)
        h_min, h_max = max(self.h_min, mask_h_min), min(self.h_max + 1, mask_h_max)
        slides = locations[0]
        for slide in slides:
            line = self.points[self.points[:, 0] == slide][:, 1:]
            line = line - np.array([x_min, h_min]).reshape(-1, 2)

            img = Image.new('L', (x_max-x_min, h_max-h_min), 0)
            ImageDraw.Draw(img).line(list(line.ravel()), width=width, fill=alpha)

            mask[slide - i_min, x_min:x_max, h_min:h_max] = np.array(img).T
        return mask
