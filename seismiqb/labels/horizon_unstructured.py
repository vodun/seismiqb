""" Horizon class for PRE-STACK data. """
#pylint: disable=too-many-lines, import-error
import os
from textwrap import dedent
from itertools import product

import numpy as np
import pandas as pd

from ..plotters import plot
from ..utils import round_to_array



class UnstructuredHorizon:
    """ Contains unstructured horizon.

    Initialized from `storage` and `geometry`, where `storage` is a csv-like file.

    The main inner storage is `dataframe`, that is a mapping from indexing headers to horizon depth.
    Since the index of that dataframe is the same, as the index of dataframe of a geometry, these can be combined.

    UnstructuredHorizon provides following features:
        - Method `add_to_mask` puts 1's on the location of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - Few of visualization methods: view from above, slices along iline/xline axis, etc.

    There is a lazy method creation in place: the first person who needs them would need to code them.
    """
    CHARISMA_SPEC = ['INLINE', '_', 'INLINE_3D', 'XLINE', '__', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y', 'height']
    REDUCED_CHARISMA_SPEC = ['INLINE_3D', 'CROSSLINE_3D', 'height']

    FBP_SPEC = ['FieldRecord', 'TraceNumber', 'file_id', 'FIRST_BREAK_TIME']

    def __init__(self, storage, geometry, name=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.format = None

        # Storage
        self.dataframe = None
        self.attached = False

        # Heights information
        self.h_min, self.h_max = None, None
        self.h_mean, self.h_std = None, None

        # Attributes from geometry
        self.geometry = geometry
        self.cube_name = geometry.name

        # Check format of storage, then use it to populate attributes
        if isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, pd.DataFrame):
            # points-like dataframe
            self.format = 'dataframe'

        elif isinstance(storage, np.ndarray) and storage.ndim == 2 and storage.shape[1] == 3:
            # array with row in (iline, xline, height) format
            self.format = 'points'

        getattr(self, f'from_{self.format}')(storage, **kwargs)


    def from_points(self, points, **kwargs):
        """ Not needed. """


    def from_dataframe(self, dataframe, attach=True, height_prefix='height', transform=False):
        """ Load horizon data from dataframe.

        Parameters
        ----------
        attack : bool
            Whether to store horizon data in common dataframe inside `geometry` attributes.
        transform : bool
            Whether to transform line coordinates to the cubic ones.
        height_prefix : str
            Column name with height.
        """
        if transform:
            dataframe[height_prefix] = (dataframe[height_prefix] - self.geometry.delay) / self.geometry.sample_rate
        dataframe.rename(columns={height_prefix: self.name}, inplace=True)
        dataframe.set_index(self.geometry.index_headers, inplace=True)
        self.dataframe = dataframe

        self.h_min, self.h_max = self.dataframe.min().values[0], self.dataframe.max().values[0]
        self.h_mean, self.h_std = self.dataframe.mean().values[0], self.dataframe.std().values[0]

        if attach:
            self.attach()


    def from_file(self, path, names=None, columns=None, height_prefix='height', reader_params=None, **kwargs):
        """ Init from path to csv-like file.

        Parameters
        ----------
        names : sequence of str
            Names of columns in file.
        columns : sequence of str
            Names of columns to actually load.
        height_prefix : str
            Column name with height.
        reader_params : None or dict
            Additional parameters for file reader.
        """
        #pylint: disable=anomalous-backslash-in-string
        _ = kwargs
        if names is None:
            with open(path, encoding='utf-8') as file:
                line_len = len(file.readline().split(' '))
            if line_len == 3:
                names = UnstructuredHorizon.REDUCED_CHARISMA_SPEC
            elif line_len == 9:
                names = UnstructuredHorizon.CHARISMA_SPEC
        columns = columns or self.geometry.index_headers + [height_prefix]

        self.path = path
        self.name = os.path.basename(path) if self.name is None else self.name

        defaults = {'sep': r'\s+'}
        reader_params = reader_params or {}
        reader_params = {**defaults, **reader_params}
        df = pd.read_csv(path, names=names, usecols=columns, **reader_params)

        # Convert coordinates of horizons to the one that present in cube geometry
        # df[columns] = np.rint(df[columns]).astype(np.int64)
        df[columns] = df[columns].astype(np.int64)
        for i, idx in enumerate(self.geometry.index_headers):
            df[idx] = round_to_array(df[idx].values, self.geometry.uniques[i])

        self.from_dataframe(df, transform=True, height_prefix=columns[-1])

    def attach(self):
        """ Store horizon data in common dataframe inside `geometry` attributes. """
        if not hasattr(self.geometry, 'horizons'):
            self.geometry.horizons = pd.DataFrame(index=self.geometry.dataframe.index)

        self.geometry.horizons = pd.merge(self.geometry.horizons, self.dataframe,
                                          left_index=True, right_index=True,
                                          how='left')
        self.attached = True


    def add_to_mask(self, mask, locations=None, width=3, alpha=1, iterator=None, **kwargs):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        locations : sequence of arrays
            List of desired locations to load: along the first index, the second, and depth.
        width : int
            Width of an added horizon.
        iterator : None or sequence
            If provided, indices to use to load height from dataframe.
        """
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        shift_1, shift_2, h_min = [slc.start for slc in locations]
        h_max = locations[-1].stop

        if iterator is None:
            # Usual case
            iterator = list(product(*[[self.geometry.uniques[idx][i]
                                       for i in range(locations[idx].start, locations[idx].stop)]
                                      for idx in range(2)]))
            idx_iterator = np.array(list(product(*[list(range(slc.start, slc.stop)) for slc in locations[:2]])))
            idx_1 = idx_iterator[:, 0] - shift_1
            idx_2 = idx_iterator[:, 1] - shift_2

        else:
            #TODO: remove this and make separate method inside `SeismicGeometry` for loading data with same iterator
            #TODO: think about moving horizons to `geometry` attributes altogether..
            # Currently, used in `show_slide` only:
            axis = np.argmin(np.array([len(np.unique(np.array(iterator)[:, idx])) for idx in range(2)]))
            loc = iterator[axis][0]
            other_axis = 1 - axis

            others = self.geometry.dataframe[self.geometry.dataframe.index.get_level_values(axis) == loc]
            others = others.index.get_level_values(other_axis).values
            others_iterator = np.array([np.where(others == item[other_axis])[0][0] for item in iterator])

            idx_1 = np.zeros_like(others_iterator) if axis == 0 else others_iterator
            idx_2 = np.zeros_like(others_iterator) if axis == 1 else others_iterator


        heights = self.dataframe[self.name].reindex(iterator, fill_value=np.nan).values.astype(np.int32)

        # Filter labels based on height
        heights_mask = np.asarray((np.isnan(heights) == False) & # pylint: disable=singleton-comparison
                                  (heights >= h_min + low) &
                                  (heights <= h_max - high)).nonzero()[0]

        idx_1 = idx_1[heights_mask]
        idx_2 = idx_2[heights_mask]
        heights = heights[heights_mask]
        heights -= (h_min + low)

        # Place values on current heights and shift them one unit below.
        for _ in range(width):
            mask[idx_1, idx_2, heights] = alpha
            heights += 1

        return mask

    # Methods to implement in the future
    def filter_points(self, **kwargs):
        """ Remove points that correspond to bad traces, e.g. zero traces. """

    def merge(self, **kwargs):
        """ Merge two instances into one. """

    def get_cube_values(self, **kwargs):
        """ Get cube values along the horizon.
        Can be easily done via subsequent `segyfile.depth_slice` and `reshape`.
        """

    def dump(self, **kwargs):
        """ Save the horizon to the disk. """

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.geometry.displayed_name}
        Depths range:           {self.h_min} to {self.h_max}
        Depths mean:            {self.h_mean:.6}
        Depths std:             {self.h_std:.6}
        Length:                 {len(self.dataframe)}
        """
        return dedent(msg)


    # Visualization
    def show_slide(self, loc, width=3, axis=0, stable=True, **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)
        locations = self.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([(slc.stop - slc.start) for slc in locations])

        # Create the same indices, as for seismic slide loading
        #TODO: make slide indices shareable
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis, stable=stable)
        _, iterator = self.geometry.make_slide_indices(loc=loc, axis=axis, stable=stable, return_iterator=True)
        shape[1 - axis] = -1

        # Create mask with horizon
        mask = np.zeros_like(seismic_slide.reshape(shape))
        mask = self.add_to_mask(mask, locations, width=width, iterator=iterator if stable else None)
        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)

        # set defaults if needed and plot the slide
        kwargs = {
            'title_label': (f'U-horizon `{self.name}` on `{self.cube_name}`' + '\n ' +
                      f'{self.geometry.index_headers[axis]} {loc} out of {self.geometry.lens[axis]}'),
            'xlabel': self.geometry.index_headers[1 - axis],
            'ylabel': 'Depth', 'y': 1.015,
            **kwargs
        }
        return plot([seismic_slide, mask], **kwargs)
