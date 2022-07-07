""" Horizon class for POST-STACK data. """
import os
from copy import copy
from textwrap import dedent

import numpy as np

from skimage.measure import label
from scipy.ndimage import find_objects

from .horizon_attributes import AttributesMixin
from .horizon_extraction import ExtractionMixin
from .horizon_processing import ProcessingMixin
from .horizon_visualization import VisualizationMixin
from ..utils import CacheMixin, CharismaMixin
from ..utils import groupby_mean, groupby_min, groupby_max
from ..utils import MetaDict



class Horizon(AttributesMixin, CacheMixin, CharismaMixin, ExtractionMixin, ProcessingMixin, VisualizationMixin):
    """ Contains spatially-structured horizon: each point describes a height on a particular (iline, xline).

    Initialized from `storage` and `geometry`, where storage can be one of:
        - csv-like file in CHARISMA or REDUCED_CHARISMA format.
        - ndarray of (N, 3) shape.
        - ndarray of (ilines_len, xlines_len) shape.
        - dictionary: a mapping from (iline, xline) -> height.
        - mask: ndarray of (ilines_len, xlines_len, depth) with 1's at places of horizon location.

    Main storages are `matrix` and `points` attributes:
        - `matrix` is a depth map, ndarray of (ilines_len, xlines_len) shape with each point
          corresponding to horizon height at this point. Note that shape of the matrix is generally smaller
          than cube spatial range: that allows to save space.
          Attributes `i_min` and `x_min` describe position of the matrix in relation to the cube spatial range.
          Each point with absent horizon is filled with `FILL_VALUE`.
          Note that since the dtype of `matrix` is `np.int32`, we can't use `np.nan` as the fill value.
          In order to initialize from this storage, one must supply `matrix`, `i_min`, `x_min`.

        - `points` is a (N, 3) ndarray with every row being (iline, xline, height). Note that (iline, xline) are
          stored in cube coordinates that range from 0 to `ilines_len` and 0 to `xlines_len` respectively.
          Stored height is corrected on `time_delay` and `sample_rate` of the cube.
          In order to initialize from this storage, one must supply (N, 3) ndarray.

    Depending on which attribute was created at initialization (`matrix` or `points`), the other is computed lazily
    at the time of the first access. This way, we can greatly amortize computations when dealing with huge number of
    `Horizon` instances, i.e. when extracting surfaces from predicted masks.

    Independently of type of initial storage, Horizon provides following:
        - Attributes `i_min`, `x_min`, `i_max`, `x_max`, `h_min`, `h_max`, `h_mean`, `h_std`, `bbox`,
          to completely describe location of the horizon in the 3D volume of the seismic cube.

        - Convenient methods of changing the horizon, `apply_to_matrix` and `apply_to_points`:
          these methods must be used instead of manually permuting `matrix` and `points` attributes.
          For example, filtration or smoothing of a horizon can be done with their help.

        - Method `add_to_mask` puts 1's on the `location` of a horizon inside provided `background`.

        - `get_cube_values` allows to cut seismic data along the horizon: that data can be used to evaluate
          horizon quality.

        - `evaluate` allows to quickly assess the quality of a seismic reflection;
          for more metrics, check :class:`~.HorizonMetrics`.

        - A number of properties that describe geometrical, geological and mathematical characteristics of a horizon.
          For example, `borders_matrix` and `boundaries_matrix`: the latter containes outer and inner borders;
          `coverage` is the ratio between labeled traces and non-zero traces in the seismic cube;
          `solidity` is the ratio between labeled traces and traces inside the hull of the horizon;
          `perimeter` and `number_of_holes` speak for themselves.

        - Multiple instances of Horizon can be compared against one another and, if needed,
          merged into one (either in-place or not) via `check_proximity`, `overlap_merge`, `adjacent_merge` methods.
          These methods are highly optimized in their accesses to inner attributes that are computed lazily.

        - A wealth of visualization methods: view from above, slices along iline/xline axis, etc.
    """
    #pylint: disable=too-many-public-methods, import-outside-toplevel, redefined-builtin

    # Columns that are used from the file
    COLUMNS = ['iline', 'xline', 'height']

    # Value to place into blank spaces
    FILL_VALUE = -999999


    def __init__(self, storage, field, name=None, dtype=np.int32, force_format=None, **kwargs):
        # Meta information
        self.path = None
        self.name = name
        self.dtype = dtype
        self.format = None

        # Location of the horizon inside cube spatial range
        self.i_min, self.i_max = None, None
        self.x_min, self.x_max = None, None
        self.i_length, self.x_length = None, None
        self.bbox = None
        self._len = None

        # Underlying data storages
        self._matrix = None
        self._points = None
        self._depths = None

        # Heights information
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None

        # Field reference
        self.field = field

        # Check format of storage, then use it to populate attributes
        if force_format is not None:
            self.format = force_format

        elif isinstance(storage, str):
            # path to csv-like file
            self.format = 'file'

        elif isinstance(storage, dict):
            # mapping from (iline, xline) to (height)
            self.format = 'dict'

        elif isinstance(storage, np.ndarray):
            if storage.ndim == 2 and storage.shape[1] == 3:
                # array with row in (iline, xline, height) format
                self.format = 'points'

            elif storage.ndim == 2 and (storage.shape == self.field.spatial_shape):
                # matrix of (iline, xline) shape with every value being height
                self.format = 'full_matrix'

            elif storage.ndim == 2:
                # matrix of (iline, xline) shape with every value being height
                self.format = 'matrix'

        getattr(self, f'from_{self.format}')(storage, **kwargs)


    # Logic of lazy computation of `points` or `matrix` from the other available storage; cache management
    @property
    def points(self):
        """ Storage of horizon data as (N, 3) array of (iline, xline, height) in cubic coordinates.
        If the horizon is created not from (N, 3) array, evaluated at the time of the first access.
        """
        if self._points is None and self.matrix is not None:
            points = self.matrix_to_points(self.matrix).astype(self.dtype)
            points += np.array([self.i_min, self.x_min, 0], dtype=self.dtype)
            self._points = points
        return self._points

    @points.setter
    def points(self, value):
        self._points = value

    @staticmethod
    def matrix_to_points(matrix):
        """ Convert depth-map matrix to points array. """
        idx = np.nonzero(matrix != Horizon.FILL_VALUE)
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            matrix[idx[0], idx[1]].reshape(-1, 1)])
        return points


    @property
    def matrix(self):
        """ Storage of horizon data as depth map: matrix of (ilines_length, xlines_length) with each point
        corresponding to height. Matrix is shifted to a (i_min, x_min) point so it takes less space.
        If the horizon is created not from matrix, evaluated at the time of the first access.
        """
        if self._matrix is None and self.points is not None:
            self._matrix = self.points_to_matrix(points=self.points, i_min=self.i_min, x_min=self.x_min,
                                                 i_length=self.i_length, x_length=self.x_length, dtype=self.dtype)
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @staticmethod
    def points_to_matrix(points, i_min, x_min, i_length, x_length, dtype=np.int32):
        """ Convert array of (N, 3) shape to a depth map (matrix). """
        matrix = np.full((i_length, x_length), Horizon.FILL_VALUE, dtype)

        matrix[points[:, 0].astype(np.int32) - i_min,
               points[:, 1].astype(np.int32) - x_min] = points[:, 2]

        return matrix

    @property
    def depths(self):
        """ Array of depth only. Useful for faster stats computation when initialized from a matrix. """
        if self._depths is None:
            if self._points is not None:
                self._depths = self.points[:, -1]
            else:
                self._depths = self.matrix[self.matrix != self.FILL_VALUE]
        return self._depths


    def reset_storage(self, storage=None):
        """ Reset storage along with depth-wise lazy computed stats. """
        self._depths = None
        self._h_min, self._h_max = None, None
        self._h_mean, self._h_std = None, None
        self._len = None

        if storage == 'matrix':
            self._matrix = None
            if len(self.points) > 0:
                i_min, x_min, h_min = np.min(self.points, axis=0)
                i_max, x_max, h_max = np.max(self.points, axis=0)

                self._h_min, self._h_max = h_min.astype(self.dtype), h_max.astype(self.dtype)
                self.i_min, self.i_max, self.x_min, self.x_max = int(i_min), int(i_max), int(x_min), int(x_max)

                self.i_length = (self.i_max - self.i_min) + 1
                self.x_length = (self.x_max - self.x_min) + 1
                self.bbox = np.array([[self.i_min, self.i_max],
                                    [self.x_min, self.x_max],
                                    [self.h_min, self.h_max]],
                                    dtype=np.int32)
        elif storage == 'points':
            self._points = None

        self.reset_cache()

    def copy(self, add_prefix=True):
        """ Create a new horizon with the same data.

        Returns
        -------
        A horizon object with new matrix object and a reference to the same field.
        """
        prefix = 'copy_of_' if add_prefix else ''
        return type(self)(storage=np.copy(self.matrix), field=self.field, i_min=self.i_min, x_min=self.x_min,
                          name=f'{prefix}{self.name}')

    __copy__ = copy

    def __sub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Operands types do not match. Got {type(self)} and {type(other)}.")

        presence = other.full_binary_matrix
        discrepancies = self.full_matrix[presence] != other.full_matrix[presence]
        if discrepancies.any():
            raise ValueError("Horizons have different depths where present.")

        res_matrix = self.full_matrix.copy()
        res_matrix[presence] = self.FILL_VALUE
        name = f"~{other.name}"
        result = type(self)(storage=res_matrix, field=self.field, name=name)

        return result


    # Properties, computed from lazy evaluated attributes
    @property
    def h_min(self):
        """ Minimum depth value. """
        if self._h_min is None:
            self._h_min = np.min(self.depths)
        return self._h_min

    @property
    def h_max(self):
        """ Maximum depth value. """
        if self._h_max is None:
            self._h_max = np.max(self.depths)
        return self._h_max

    @property
    def h_mean(self):
        """ Average depth value. """
        if self._h_mean is None:
            self._h_mean = np.mean(self.depths)
        return self._h_mean

    @property
    def h_std(self):
        """ Std of depths. """
        if self._h_std is None:
            self._h_std = np.std(self.depths)
        return self._h_std

    def __len__(self):
        """ Number of labeled traces. """
        if self._len is None:
            if self._points is not None:
                self._len = len(self.points)
            else:
                self._len = len(self.depths)
        return self._len


    # Initialization from different containers
    def from_points(self, points, verify=True, dst='points', reset='matrix', **kwargs):
        """ Base initialization: from point cloud array of (N, 3) shape.

        Parameters
        ----------
        points : ndarray
            Array of points. Each row describes one point inside the cube: two spatial coordinates and depth.
        verify : bool
            Whether to remove points outside of the cube range.
        dst : str
            Attribute to save result.
        reset : str or None
            Storage to reset.
        """
        _ = kwargs

        if verify:
            idx = np.where((points[:, 0] >= 0) &
                           (points[:, 1] >= 0) &
                           (points[:, 2] >= 0) &
                           (points[:, 0] < self.field.shape[0]) &
                           (points[:, 1] < self.field.shape[1]) &
                           (points[:, 2] < self.field.shape[2]))[0]
            points = points[idx]

        if self.dtype == np.int32:
            points = np.rint(points)
        if points.dtype != self.dtype:
            points = points.astype(self.dtype)
        setattr(self, dst, points)

        # Collect stats on separate axes. Note that depth stats are properties
        if reset:
            self.reset_storage(reset)


    def from_file(self, path, transform=True, **kwargs):
        """ Init from path to either CHARISMA or REDUCED_CHARISMA csv-like file. """
        path = self.field.make_path(path, makedirs=False)
        self.path = path

        self.name = os.path.basename(path) if self.name is None else self.name

        points = self.load_charisma(path=path, dtype=self.dtype, format='points',
                                    fill_value=Horizon.FILL_VALUE, transform=transform,
                                    verify=True)

        self.from_points(points, verify=False, **kwargs)


    def from_matrix(self, matrix, i_min, x_min, h_min=None, h_max=None, length=None, **kwargs):
        """ Init from matrix and location of minimum i, x points. """
        _ = kwargs

        if matrix.dtype != self.dtype:
            if self.dtype == np.int32:
                matrix = np.rint(matrix)
            matrix = matrix.astype(self.dtype)
        self.matrix = matrix

        self.i_min, self.x_min = i_min, x_min
        self.i_max, self.x_max = i_min + matrix.shape[0] - 1, x_min + matrix.shape[1] - 1

        self.i_length = (self.i_max - self.i_min) + 1
        self.x_length = (self.x_max - self.x_min) + 1

        # Populate lazy properties with supplied values
        self._h_min, self._h_max, self._len = h_min, h_max, length
        self.bbox = np.array([[self.i_min, self.i_max],
                              [self.x_min, self.x_max],
                              [self.h_min, self.h_max]],
                             dtype=np.int32)


    def from_full_matrix(self, matrix, **kwargs):
        """ Init from matrix that covers the whole cube. """
        kwargs = {
            'i_min': 0,
            'x_min': 0,
            **kwargs
        }
        self.from_matrix(matrix, **kwargs)


    def from_dict(self, dictionary, transform=True, **kwargs):
        """ Init from mapping from (iline, xline) to depths. """
        _ = kwargs

        points = self.dict_to_points(dictionary)

        if transform:
            points = self.field.lines_to_cubic(points)

        self.from_points(points)

    @staticmethod
    def dict_to_points(dictionary):
        """ Convert mapping to points array. """
        points = np.hstack([np.array(list(dictionary.keys())),
                            np.array(list(dictionary.values())).reshape(-1, 1)])
        return points


    @staticmethod
    def from_mask(mask, field=None, origin=None, connectivity=3,
                  mode='mean', threshold=0.5, minsize=0, prefix='predict', **kwargs):
        """ Convert mask to a list of horizons.
        Returned list is sorted by length of horizons.

        Parameters
        ----------
        field : Field
            Horizon parent field.
        origin : sequence
            The upper left coordinate of a `mask` in the cube coordinates.
        threshold : float
            Parameter of mask-thresholding.
        mode : str
            Method used for finding the point of a horizon for each iline, xline.
        minsize : int
            Minimum length of a horizon to be extracted.
        prefix : str
            Name of horizon to use.
        """
        _ = kwargs
        if mode in ['mean', 'avg']:
            group_function = groupby_mean
        elif mode in ['min']:
            group_function = groupby_min
        elif mode in ['max']:
            group_function = groupby_max

        # Labeled connected regions with an integer
        labeled = label(mask >= threshold, connectivity=connectivity)
        objects = find_objects(labeled)

        # Create an instance of Horizon for each separate region
        horizons = []
        for i, sl in enumerate(objects):
            max_possible_length = 1
            for j in range(3):
                max_possible_length *= sl[j].stop - sl[j].start

            if max_possible_length >= minsize:
                indices = np.nonzero(labeled[sl] == i + 1)

                if len(indices[0]) >= minsize:
                    coords = np.vstack([indices[i] + sl[i].start for i in range(3)]).T

                    points = group_function(coords) + origin
                    horizons.append(Horizon(storage=points, field=field, name=f'{prefix}_{i}'))

        horizons.sort(key=len)
        horizons = [horizon for horizon in horizons if len(horizon) != 0]
        return horizons

    def from_subset(self, matrix, name=None):
        """ Make new label with points matrix filtered by given presense matrix.

        Parameters
        ----------
        matrix : np.array
            Presense matrix of labels points. Must be in full cubes coordinates.
            If consists of 0 and 1, keep points only where values are 1.
            If consists of values from [0, 1] interval, keep points where values are greater than 0.5.
        name : str or None
            Name for new label. If None, original label name used.

        Returns
        -------
        New `Horizon` instance with filtered points matrix.
        """
        result = copy(self)
        result.name = name or self.name

        filtering_matrix = (matrix < 0.5).astype(int)
        result.filter(filtering_matrix, inplace=True)

        return result

    # Basic properties
    @property
    def shape(self):
        """ Tuple of horizon dimensions."""
        return (self.i_length, self.x_length)

    @property
    def size(self):
        """ Number of elements in the full horizon matrix."""
        return self.i_length * self.x_length

    @property
    def short_name(self):
        """ Name without extension. """
        if self.name is not None:
            return self.name.split('.')[0]
        return None

    @property
    def displayed_name(self):
        """ Alias for `short_name`. """
        if self.name is None:
            return 'unknown_horizon'
        return self.short_name

    # Horizon usage: mask generation
    def add_to_mask(self, mask, locations=None, width=3, alpha=1, **kwargs):
        """ Add horizon to a background.
        Note that background is changed in-place.

        Parameters
        ----------
        mask : ndarray
            Background to add horizon to.
        locations : ndarray
            Where the mask is located.
        width : int
            Width of an added horizon.
        alpha : number
            Value to fill background with at horizon location.
        """
        _ = kwargs
        low = width // 2
        high = max(width - low, 0)

        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        #TODO: add clear explanation about usage of advanced index in Horizon
        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)

        if i_max > i_min and x_max > x_min:
            overlap = self.matrix[i_min - self.i_min : i_max - self.i_min,
                                  x_min - self.x_min : x_max - self.x_min]

            # Coordinates of points to use in overlap local system
            idx_i, idx_x = np.asarray((overlap != self.FILL_VALUE) &
                                      (overlap >= mask_h_min + low) &
                                      (overlap <= mask_h_max - high)).nonzero()
            heights = overlap[idx_i, idx_x]

            # Convert coordinates to mask local system
            idx_i += i_min - mask_i_min
            idx_x += x_min - mask_x_min
            heights -= (mask_h_min + low)

            for shift in range(width):
                mask[idx_i, idx_x, heights + shift] = alpha

        return mask


    def add_to_regression_mask(self, mask, locations, scale=False):
        """ Add depth matrix at `locations` to `mask`. """
        mask_bbox = np.array([[slc.start, slc.stop] for slc in locations], dtype=np.int32)

        # Getting coordinates of overlap in cubic system
        (mask_i_min, mask_i_max), (mask_x_min, mask_x_max), (mask_h_min, mask_h_max) = mask_bbox

        i_min, i_max = max(self.i_min, mask_i_min), min(self.i_max + 1, mask_i_max)
        x_min, x_max = max(self.x_min, mask_x_min), min(self.x_max + 1, mask_x_max)

        if i_max > i_min and x_max > x_min:
            overlap = self.matrix[i_min - self.i_min : i_max - self.i_min,
                                  x_min - self.x_min : x_max - self.x_min]

            # Coordinates of points to use in overlap local system
            idx_i, idx_x = np.asarray((overlap != self.FILL_VALUE) &
                                      (overlap >= mask_h_min) &
                                      (overlap <= mask_h_max)).nonzero()
            heights = overlap[idx_i, idx_x].astype(np.float32)

            if scale:
                heights -= mask_h_min
                heights /= (mask_h_max - mask_h_min)

            mask[idx_i, idx_x] = heights
        return mask

    def load_slide(self, loc, axis=0, width=3):
        """ Create a mask at desired location along supplied axis. """
        axis = self.field.geometry.parse_axis(axis)
        locations = self.field.geometry.make_slide_locations(loc, axis=axis)
        shape = np.array([(slc.stop - slc.start) for slc in locations])
        width = width or max(5, shape[-1] // 100)

        mask = np.zeros(shape, dtype=np.float32)
        mask = self.add_to_mask(mask, locations=locations, width=width)
        return np.squeeze(mask)


    # Evaluate horizon on its own / against other(s)
    @property
    def metrics(self):
        """ Calculate :class:`~HorizonMetrics` on demand. """
        # pylint: disable=import-outside-toplevel
        from ..metrics import HorizonMetrics
        return HorizonMetrics(self)

    def evaluate(self, compute_metric=True, supports=50, plot=True, savepath=None, printer=print, **kwargs):
        """ Compute crucial metrics of a horizon.

        Parameters
        ----------
        compute_metrics : bool
            Whether to compute correlation map of a horizon.
        supports, savepath, plot, kwargs
            Passed directly to :meth:`HorizonMetrics.evaluate`.
        printer : callable
            Function to display message with metrics.
        """
        # Textual part
        if printer is not None:
            msg = f"""
            Number of labeled points:                         {len(self)}
            Number of points inside borders:                  {np.sum(self.filled_matrix)}
            Perimeter (length of borders):                    {self.perimeter}
            Percentage of labeled non-bad traces:             {self.coverage:4.3f}
            Percentage of labeled traces inside borders:      {self.solidity:4.3f}
            Number of holes inside borders:                   {self.number_of_holes}
            """
            printer(dedent(msg))

        # Visual part
        if compute_metric:
            from ..metrics import HorizonMetrics # pylint: disable=import-outside-toplevel
            return HorizonMetrics(self).evaluate('support_corrs', supports=supports, agg='nanmean',
                                                 plot=plot, savepath=savepath, **kwargs)
        return None


    def check_proximity(self, other):
        """ Compute a number of stats of location of `self` relative to the `other` Horizons.

        Parameters
        ----------
        self, other : Horizon
            Horizons to compare.

        Returns
        -------
        dictionary with following keys:
            - `difference_matrix` with matrix of depth differences
            - `difference_mean` for average distance
            - `difference_abs_mean` for average of absolute values of point-wise distances
            - `difference_max`, `difference_abs_max`, `difference_std`, `difference_abs_std`
            - `overlap_size` with number of overlapping points
            - `window_rate` for percentage of traces that are in 5ms from one horizon to the other
        """
        difference = np.where((self.full_matrix != self.FILL_VALUE) & (other.full_matrix != self.FILL_VALUE),
                              self.full_matrix - other.full_matrix, np.nan)
        abs_difference = np.abs(difference)

        overlap_size = np.nansum(~np.isnan(difference))
        window_rate = np.nansum(abs_difference < (5 / self.field.sample_rate)) / overlap_size

        present_at_1_absent_at_2 = ((self.full_matrix != self.FILL_VALUE)
                                    & (other.full_matrix == self.FILL_VALUE)).sum()
        present_at_2_absent_at_1 = ((self.full_matrix == self.FILL_VALUE)
                                    & (other.full_matrix != self.FILL_VALUE)).sum()

        info_dict = {
            'difference_matrix' : difference,
            'difference_mean' : np.nanmean(difference),
            'difference_max' : np.nanmax(difference),
            'difference_min' : np.nanmin(difference),
            'difference_std' : np.nanstd(difference),

            'abs_difference_mean' : np.nanmean(abs_difference),
            'abs_difference_max' : np.nanmax(abs_difference),
            'abs_difference_std' : np.nanstd(abs_difference),

            'overlap_size' : overlap_size,
            'window_rate' : window_rate,

            'present_at_1_absent_at_2' : present_at_1_absent_at_2,
            'present_at_2_absent_at_1' : present_at_2_absent_at_1,
        }
        return MetaDict(info_dict)

    def find_closest(self, *others):
        """ Find closest horizon to `self` in the list of `others`. """
        proximities = [(other, self.check_proximity(other)) for other in others
                       if other.field.name == self.field.name]

        closest, proximity_info = min(proximities, key=lambda item: item[1].get('difference_mean', np.inf))
        return closest, proximity_info

    # Alias for horizon comparisons
    def compare(self, *others, clip_value=5, ignore_zeros=True,
                printer=print, plot=True, return_figure=False, hist_kwargs=None, **kwargs):
        """ Alias for `HorizonMetrics.compare`. """
        return self.metrics.compare(*others, clip_value=clip_value, ignore_zeros=ignore_zeros,
                                    printer=printer, plot=plot, return_figure=return_figure,
                                    hist_kwargs=hist_kwargs, **kwargs)

    def compute_prediction_std(self, others):
        """ Compute std of predicted horizons along depths and restrict it to `self`. """
        std_matrix = self.metrics.compute_prediction_std(list(set([self, *others])))
        std_matrix[self.mask == False] = np.nan #pylint: disable=singleton-comparison
        return std_matrix


    def equal(self, other, threshold_missing=0):
        """ Return True if the horizons are considered equal, False otherwise.
        If the `threshold_missing` is zero, then check if the points of `self` and `other` are the same.
        If the `threshold_missing` is positive, then check that in overlapping points values are the same,
        and number of missing traces is smaller than allowed.
        """
        if threshold_missing == 0:
            return np.array_equal(self.points, other.points)

        info = self.check_proximity(other)
        n_missing = max(info['present_at_1_absent_at_2'], info['present_at_2_absent_at_1'])
        return info['difference_mean'] == 0 and n_missing < threshold_missing


    # Save horizon to disk
    def dump(self, path, transform=None):
        """ Save horizon points on disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        """
        self.dump_charisma(data=copy(self.points), path=path, format='points',
                           name=self.name, transform=transform)

    def dump_float(self, path, transform=None, kernel_size=7, sigma=2., max_depth_difference=5):
        """ Smooth out the horizon values, producing floating-point numbers, and dump to the disk.

        Parameters
        ----------
        path : str
            Path to a file to save horizon to.
        transform : None or callable
            If callable, then applied to points after converting to ilines/xlines coordinate system.
        kernel_size : int
            Size of the filtering kernel.
        sigma : number
            Standard deviation of the Gaussian kernel.
        max_depth_difference : number
            If the distance between anchor point and the point inside filter is bigger than the threshold,
            then the point is ignored in smoothening.
            Can be used for separate smoothening on sides of discontinuity.
        """
        smoothed = self.smooth_out(mode='convolve', kernel_size=kernel_size, sigma_spatial=sigma,
                                   max_depth_difference=max_depth_difference, inplace=False, dtype=np.float32)

        self.dump_charisma(data=smoothed.points, path=path, format='points', name=self.name, transform=transform)
