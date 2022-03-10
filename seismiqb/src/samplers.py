""" Generator of locations of two types: Samplers and Grids.

Samplers are making random (label-dependant) locations to train models, while
Grids create predetermined locations based on field or current state of labeled surface and are used for inference.

Locations describe the cube and the exact place to load from in the following format:
(field_id, label_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).

Locations are passed to `make_locations` method of `SeismicCropBatch`, which
transforms them into 3D slices to index the data and other useful info like origin points, shapes and orientation.

Each of the classes provides:
    - `call` method (aliased to either `sample` or `next_batch`), that generates given amount of locations
    - `to_names` method to convert the first two columns of sampled locations into string names of field and label
    - convenient visualization to explore underlying `locations` structure
"""
from itertools import product
from random import choices

import numpy as np
from numba import njit
from scipy.ndimage.morphology import binary_dilation

from .utils import filtering_function, AugmentedDict
from .labels.fault import insert_fault_into_mask
from ..batchflow import Sampler, ConstantSampler
from .plotters import MatplotlibPlotter, plot_image



class BaseSampler(Sampler):
    """ Common logic of making locations. Refer to the documentation of inherited classes for more details. """
    def _make_locations(self, field, points, matrix, crop_shape, ranges, threshold, filtering_matrix):
        # Parse parameters
        ranges = ranges if ranges is not None else [None, None, None]
        ranges = [item if item is not None else [0, c]
                  for item, c in zip(ranges, field.shape)]
        ranges = np.array(ranges)

        crop_shape = np.array(crop_shape)
        crop_shape_t = crop_shape[[1, 0, 2]]
        n_threshold = np.int32(crop_shape[0] * crop_shape[1] * threshold)

        # Keep only points, that can be a starting point for a crop of given shape
        i_mask = ((ranges[:2, 0] < points[:, :2]).all(axis=1) &
                  ((points[:, :2] + crop_shape[:2]) < ranges[:2, 1]).all(axis=1))
        x_mask = ((ranges[:2, 0] < points[:, :2]).all(axis=1) &
                  ((points[:, :2] + crop_shape_t[:2]) < ranges[:2, 1]).all(axis=1))
        mask = i_mask | x_mask

        points = points[mask]
        i_mask = i_mask[mask]
        x_mask = x_mask[mask]

        # Apply filtration
        if filtering_matrix is not None:
            points = filtering_function(points, filtering_matrix)

        # Keep only points, that produce crops with horizon larger than threshold; append flag
        # TODO: Implement threshold check via filtering points with matrix obtained by
        # convolution of horizon binary matrix and a kernel with size of crop shape
        if threshold != 0.0:
            points = spatial_check_points(points, matrix, crop_shape[:2], i_mask, x_mask, n_threshold)
        else:
            _points = np.empty((i_mask.sum() + x_mask.sum(), 4), dtype=np.int32)
            _points[:i_mask.sum(), 0:3] = points[i_mask, :]
            _points[:i_mask.sum(), 3] = 0

            _points[i_mask.sum():, 0:3] = points[x_mask, :]
            _points[i_mask.sum():, 3] = 1

            points = _points

        # Transform points to (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop)
        buffer = np.empty((len(points), 7), dtype=np.int32)
        buffer[:, 0] = points[:, 3]
        buffer[:, 1:4] = points[:, 0:3]
        buffer[:, 4:7] = points[:, 0:3]
        buffer[buffer[:, 0] == 0, 4:7] += crop_shape
        buffer[buffer[:, 0] == 1, 4:7] += crop_shape_t

        self.n = len(buffer)
        self.crop_shape = crop_shape
        self.crop_shape_t = crop_shape_t
        self.crop_height = crop_shape[2]
        self.ranges = ranges
        self.threshold = threshold
        self.n_threshold = n_threshold
        return buffer


    @property
    def orientation_matrix(self):
        """ Possible locations, mapped on field top-view map.
            - np.nan where no locations can be sampled.
            - 1 where only iline-oriented crops can be sampled.
            - 2 where only xline-oriented crops can be sampled.
            - 3 where both types of crop orientations can be sampled.
        """
        matrix = np.zeros_like(self.matrix, dtype=np.float32)
        orientations = self.locations[:, 0].astype(np.bool_)

        i_locations = self.locations[~orientations]
        matrix[i_locations[:, 1], i_locations[:, 2]] += 1

        x_locations = self.locations[orientations]
        matrix[x_locations[:, 1], x_locations[:, 2]] += 2

        matrix[matrix == 0] = np.nan
        return matrix


class WeightedSampler(BaseSampler):
    """ Base methods for dealing with weights for sampling locations.

    Under the hood weights logic are defined with:
     - weights_mode : str
       Mode for creating weights and/or providing info in visualization methods.
     - weights : np.ndarray or None
       Weights matrix with shape equals to the field spatial shape. Each seismic trace has its own weight.
       Can be None if weights initialized with `locations_weights`.
     - locations_weights : np.ndarray
       An array with weights for each location. It is used in the sampling step for making priorities for locations.
    """
    def init_weights(self, weights='const', weights_agg='max', **kwargs):
        """ Weights initialization with the `create_weights` method.

        Parameters
        ----------
        weights : str or np.ndarray
            Weights initialization mode (if str) or weights matrix (if ndarray).
            Note that the matrix shape must be equal to the field spatial shape.
        weights_agg : str
            Weights aggregation option. It is used to aggregate all weights on a crop to one location weight.
            It is provided directly to the :method:`get_locations_weights`.
        kwargs : dict
            kwargs for :meth:`~.WeightedSampler.create_weights`.
        """
        if isinstance(weights, str):
            # Create weights depend on mode
            self.weights_mode = weights
            weights, locations_weights = self.create_weights(weights_mode=weights, weights_agg=weights_agg, **kwargs)
        else:
            # Get weights for each location
            self.weights_mode = 'custom matrix'
            weights[self.field.zero_traces == 1] = np.nan
            locations_weights = get_locations_weights(locations=self.locations,
                                                      weights=weights, weights_agg=weights_agg)

        self.weights = weights
        locations_weights = np.nan_to_num(locations_weights, nan=0)
        self.locations_weights = locations_weights

    def normalize_weights(self, weights):
        """ Min-max weights normalization.

        Parameters
        ----------
        weights : np.ndarray
            Weights array. Can be a weights matrix with a shape equal to the field spatial shape.
            Also can be a `locations_weights` array.
        """
        if weights.shape == self.field.spatial_shape:
            # Remove extra weights in the array
            weights[self.field.zero_traces == 1] = np.nan

        min_, max_ = np.nanmin(weights), np.nanmax(weights)

        min_ = 0 if max_ == min_ else min_
        weights = (weights - min_)/ (max_ - min_)
        return weights

    def set_weights(self, weights=None, weights_mode=None, weights_agg='max'):
        """ Sampler weights setter: sets `weights`, `locations_weights` and `weights_mode`
        depend on `weights` parameter.

        Parameters
        ----------
        weights : callable or np.ndarray
            Setting weight instructions.
            If callable, then it applied for each location in `self.locations`.
            If np.ndarray with shape equals to the field spatial shape, then it sets `self.weights`.
            If np.ndarray with size equals to the `self.locations` length, then it sets `self.locations_weights`.
        weights_mode : str, optional
            Weights creating mode name. It is used for providing info in visualizations.
        weights_agg : str
            Weights aggregation option. It is used to aggregate all weights on a crop to one location weight.
            It is provided directly to the :method:`get_locations_weights`.
        """
        self.weights = None
        self.locations_weights = None

        if callable(weights):
            # Apply callable for each location
            self.weights_mode = weights_mode or 'custom callable'
            self.locations_weights = np.apply_along_axis(weights, 1, self.locations)

        elif weights.shape == self.field.spatial_shape:
            # Set a new weights matrix and extract `locations_weights` from it
            self.weights_mode = weights_mode or 'custom matrix'
            weights[self.field.zero_traces == 1] = np.nan
            self.weights = weights
            self.locations_weights = get_locations_weights(locations=self.locations,
                                                           weights=weights, weights_agg=weights_agg)

        elif weights.size == len(self.locations):
            # Set new locations weights and clear `weights` matrix
            self.weights_mode = weights_mode or 'custom locations weights'
            self.weights = None
            self.locations_weights = weights

        else:
            raise ValueError('Invalid value for the `weights` argument.')

    def create_weights(self, weights_mode='constant', weights_agg='max', **kwargs):
        """" A simple weights creating method. It is heavily recommended to redefine this method in inherited classes.

        Under the hood, weights are initialized with ones.
        So each location for sampling will have the same priority.

        Parameters
        ----------
        weights_mode : str
            Weights creation option. In this method do nothing, it is here for the API consistency.
        weights_agg : str
            Weights aggregation option. It is used to aggregate all weights on a crop to one location weight.
            It is provided directly to the :method:`get_locations_weights`.
        kwargs : dict
            Other parameters of weights creation. In this method do nothing, it is here for the API consistency.

        Returns
        -------
        weights : np.ndarray
            Weights matrix with shape equals to the field spatial shape.
        locations_weights : np.ndarray
            Weights array with weights for each location that aggregated from weights matrix.
        """
        _ = weights_mode, kwargs
        weights = np.ones(self.field.spatial_shape, dtype='float')
        weights[self.field.zero_traces == 1] = np.nan

        locations_weights = get_locations_weights(locations=self.locations, weights=weights, weights_agg=weights_agg)
        locations_weights = np.nan_to_num(locations_weights, nan=0)
        return weights, locations_weights

@njit
def get_locations_weights(locations, weights, weights_agg='max'):
    """ Helper method for the :class:`WeightedSampler`.
    Extracts locations weights from a weights matrix using provided aggregation.

    Parameters
    ----------
    locations : np.ndarray
        Locations in (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop) format.
    weights : np.ndarray
        A weights matrix with shape equals to the field spatial shape.
        It provides weights for each seismic trace.
    weights_agg : str
        Option for weights aggregation for each location.

    Returns
    -------
    locations_weights : np.ndarray
        An array with weights for each location.
    """
    locations_weights = np.empty((len(locations)), dtype=np.float32)

    for i, (_, i_start, x_start, _, i_stop, x_stop, _) in enumerate(locations):
        # Get weights for each trace on the crop and aggregate into one value
        weights_ = weights[i_start:i_stop, x_start:x_stop]

        if weights_agg == 'max':
            locations_weights[i] = np.nanmax(weights_)
        elif weights_agg == 'mean':
            locations_weights[i] = np.nanmean(weights_)
        else:
            raise ValueError('Invalid value for the `weights_agg` argument.')

    return locations_weights


class GeometrySampler(BaseSampler):
    """ Generator of crop locations, based on a field. Not intended to be used directly, see `SeismicSampler`.
    Makes locations that:
        - start from the non-dead trace on a field, exluding those marked by `filtering_matrix`
        - contain more than `threshold` non-dead traces inside
        - don't go beyond cube limits

    Locations are produced as np.ndarray of (size, 9) shape with following columns:
        (field_id, field_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).
    Depth location is randomized in desired `ranges`.

    Under the hood, we prepare `locations` attribute:
        - filter non-dead trace coordinates so that only points that can generate
        either inline or crossline oriented crop (or both) remain
        - apply `filtering_matrix` to remove more points
        - keep only those points and directions which create crops with more than `threshold` non-dead traces
        - store all possible locations for each of the remaining points
    For sampling, we randomly choose `size` rows from `locations` and generate height in desired range.

    Parameters
    ----------
    field : Field
        Field to base sampler on.
    crop_shape : tuple
        Shape of crop locations to generate.
    threshold : float
        Minimum proportion of labeled points in each sampled location.
    ranges : sequence, optional
        Sequence of three tuples of two ints or `None`s.
        If tuple of two ints, then defines ranges of sampling along corresponding axis.
        If None, then field limits are used (no constraints).
    filtering_matrix : np.ndarray, optional
        Map of points to remove from potentially generated locations.
    field_id, label_id : int
        Used as the first two columns of sampled values.
    """
    dim = 2 + 1 + 6 # dimensionality of sampled points: field_id and label_id, orientation, locations

    def __init__(self, field, crop_shape, threshold=0.05, ranges=None, filtering_matrix=None,
                 field_id=0, label_id=0, **kwargs):
        matrix = (1 - field.zero_traces).astype(np.float32)
        idx = np.nonzero(matrix != 0)
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1),
                            np.zeros((len(idx[0]), 1), dtype=np.int32)]).astype(np.int32)

        self.locations = self._make_locations(field=field, points=points, matrix=matrix,
                                              crop_shape=crop_shape, ranges=ranges, threshold=threshold,
                                              filtering_matrix=filtering_matrix)
        self.kwargs = kwargs

        self.field_id = field_id
        self.label_id = label_id

        self.field = field
        self.matrix = matrix
        self.name = field.short_name
        self.displayed_name = field.displayed_name
        super().__init__()

    def sample(self, size):
        """ Get exactly `size` locations. """
        idx = np.random.randint(self.n, size=size)
        sampled = self.locations[idx]

        heights = np.random.randint(low=self.ranges[2, 0],
                                    high=self.ranges[2, 1] - self.crop_height,
                                    size=size, dtype=np.int32)

        buffer = np.empty((size, 9), dtype=np.int32)
        buffer[:, 0] = self.field_id
        buffer[:, 1] = self.label_id

        buffer[:, [2, 3, 4, 6, 7]] = sampled[:, [0, 1, 2, 4, 5]]
        buffer[:, 5] = heights
        buffer[:, 8] = heights + self.crop_height
        return buffer

    def __repr__(self):
        return f'<GeometrySampler for {self.displayed_name}: '\
               f'crop_shape={tuple(self.crop_shape)}, threshold={self.threshold}>'



class HorizonSampler(WeightedSampler):
    """ Generator of crop locations, based on a single horizon. Not intended to be used directly, see `SeismicSampler`.
    Makes locations that:
        - start from the labeled point on horizon, excluding those marked by `filtering_matrix`
        - contain more than `threshold` labeled pixels inside
        - don't go beyond cube limits

    Locations are produced as np.ndarray of (size, 9) shape with following columns:
        (field_id, label_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).
    Depth location is randomized in (0.1*shape, 0.9*shape) range.

    Under the hood, we prepare `locations` attribute:
        - filter horizon points so that only points that can generate
        either inline or crossline oriented crop (or both) remain
        - apply `filtering_matrix` to remove more points
        - keep only those points and directions which create crops with more than `threshold` labels
        - store all possible locations for each of the remaining points
    For sampling, we randomly choose `size` rows from `locations`. If some of the sampled locations does not fit the
    `threshold` constraint, resample until we get exactly `size` locations.

    Parameters
    ----------
    horizon : Horizon
        Horizon to base sampler on.
    crop_shape : tuple
        Shape of crop locations to generate.
    threshold : float
        Minimum proportion of labeled points in each sampled location.
    ranges : sequence, optional
        Sequence of three tuples of two ints or `None`s.
        If tuple of two ints, then defines ranges of sampling along this axis.
        If None, then field limits are used (no constraints).
        Note that we actually use only the first two elements, corresponding to spatial ranges.
    filtering_matrix : np.ndarray, optional
        Map of points to remove from potentially generated locations.
    field_id, label_id : int
        Used as the first two columns of sampled values.
    shift_height : bool
        Whether apply random shift to height locations of sampled horizon points or not.
    """
    dim = 2 + 1 + 6 # dimensionality of sampled points: field_id and label_id, orientation, locations

    def __init__(self, horizon, crop_shape, threshold=0.05, ranges=None, filtering_matrix=None,
                 shift_height=True, spatial_shift=False,
                 field_id=0, label_id=0, **kwargs):
        field = horizon.field
        matrix = horizon.full_matrix

        self.locations = self._make_locations(field=field, points=horizon.points.copy(), matrix=matrix,
                                              crop_shape=crop_shape, ranges=ranges, threshold=threshold,
                                              filtering_matrix=filtering_matrix)
        self.kwargs = kwargs

        self.field_id = field_id
        self.label_id = label_id

        self.horizon = horizon
        self.field = field
        self.matrix = matrix
        self.name = field.short_name
        self.displayed_name = horizon.displayed_name

        if shift_height:
            shift_height = shift_height if isinstance(shift_height, tuple) else (0.9, 0.1)
        self.shift_height = shift_height

        self.spatial_shift = spatial_shift

        # Get information about ptp for each location
        hm = horizon.full_matrix.copy().astype(float)
        hm[hm == horizon.FILL_VALUE] = np.nan

        self.locations_ptps = get_locations_ptps(locations=self.locations, horizon_full_matrix=hm)

        # Weights initialization
        weights = kwargs.pop('weights', 'const')
        weights_agg = kwargs.pop('weights_agg', 'max')
        self.init_weights(weights=weights, weights_agg=weights_agg, **kwargs)
        super().__init__()

    def create_weights(self, weights_mode='constant', weights_agg='max', **kwargs):
        """" Creating weights from a string.

        Under the hood, we parse the `weights_mode` string and apply each supported creating method in it using kwargs.

        Parameters
        ----------
        weights_mode : str
            A string that provides weights creating options.
            It can be a combination of submodes:
                - 'const': create weights matrix with ones.
                - 'grad': create weights matrix as a gradient matrix.
                - 'corrs': create weights matrix as a support correlations matrix.
                - 'near_holes': create weights matrix with weights equals to one near holes
                and equals to zero in other places.
                - 'ptp': create locations weights as ptp for each location. Note that in this case
                the weights matrix is not defined.
            Note that for each option weights are min-max normalized for better combining multiple modes.
        weights_agg : str
            Weights aggregation option. It is used to aggregate all weights on a crop to one location weight.
            It is provided directly to the :method:`get_locations_weights`.
        kwargs : dict
            Dictionary where keys are `<weights_mode submodes>_kwargs` and values are dictionaries with kwargs.
            For the following modes we can manage parameters:
                - 'grad_kwargs' support all :method:`Horizon.get_gradient_map` kwargs.
                - 'near_holes_kwargs' support only the 'max_distance' argument that provides a max distance
                between a hole on the horizon surface and nonzero weighted traces.
                - 'ptp_kwargs' support next kwargs:
                    - 'ptp_high': the highest depths ptp value. Values bigger than 'ptp_high' have weights equal to 0.
                    - 'ptp_weights_map': mapping ptp values to weights values. It can be a dict or a callable.
                    For example, it can be {0: 0.5} (all locations with zero ptp have 0.5 weights) or
                    .. code-block:: python

                        lambda x: int(x > 0)

        Returns
        -------
        weights : np.ndarray or None
            Weights matrix with shape equals to the field spatial shape.
            It can be None if the `weights_mode` parameter contains 'ptp' substring. We count 'ptp'
            for each location, and the weights matrix can't be correctly calculated.
        locations_weights : np.ndarray
            Weights array with weights for each location that aggregated from the weights matrix.
        """
        # `weights` matrix initialization
        if 'const' in weights_mode:
            weights = np.ones(self.matrix.shape, dtype='float')
        else:
            weights = np.zeros(self.matrix.shape, dtype='float')

        if 'grad' in weights_mode:
            # Calculate and add gradient matrix to the `weights`
            weights_ = self.horizon.get_gradient_map(kwargs.pop('grad_kwargs', {}))
            weights[self.horizon.full_matrix == self.horizon.FILL_VALUE] = np.nan
            weights += self.normalize_weights(weights=weights_)

        if 'corrs' in weights_mode:
            # Calculate and add correlations matrix to the `weights`
            weights_ = self.horizon.metrics.evaluate('support_corrs', supports=100, agg='nanmean',
                                                     plot=False, kwargs=kwargs.pop('corrs_kwargs', {}))
            weights[self.horizon.full_matrix == self.horizon.FILL_VALUE] = np.nan
            weights += self.normalize_weights(weights=weights_)

        if 'near_holes' in weights_mode:
            # Add weights only for traces near holes
            near_holes_kwargs_ = kwargs.pop('near_holes_kwargs', {'max_distance': 50})
            # Find holes on tha surface (excluding bad traces)
            holes_mask = 1 - (self.horizon.full_binary_matrix + self.horizon.field.zero_traces)
            holes_mask = holes_mask.astype(bool)

            # Get area near holes with binary dilation
            weights_ = binary_dilation(holes_mask, iterations=near_holes_kwargs_.get('max_distance'))
            weights_ = weights_ ^ holes_mask
            weights_ = weights_.astype(float)

            # Exclude bad traces from weights and normalize them
            weights[self.horizon.full_matrix == self.horizon.FILL_VALUE] = np.nan
            weights += self.normalize_weights(weights=weights_)

        # Extract `locations_weights` from the `weights` matrix
        if np.all(weights == 0):
            locations_weights = np.zeros((len(self.locations)), dtype=float)
        else:
            locations_weights = get_locations_weights(locations=self.locations,
                                                      weights=weights, weights_agg=weights_agg)

        weights[self.horizon.full_matrix == self.horizon.FILL_VALUE] = np.nan

        # For 'ptp' mode weights matrix doesn't exist, get only `locations_weights`
        if 'ptp' in weights_mode:
            # For each location get weights as ptp of depths of the horizon
            # and prepare them depend on kwargs
            weights = None

            ptp_kwargs_ = kwargs.pop('ptp_kwargs', {'ptp_high': None, 'ptp_weights_map': {0: 0}})
            ptp_high = ptp_kwargs_.get('ptp_high')
            ptp_weights_map = ptp_kwargs_.get('ptp_weights_map')

            locations_weights_ = self.locations_ptps.copy()

            if isinstance(ptp_weights_map, dict):
                # Set specific weights for ptp values from `ptp_weights_map`
                for k, v in ptp_weights_map.items():
                    locations_weights_[self.locations_ptps == k] = v
            elif callable(ptp_weights_map):
                ptp_weights_map = np.vectorize(ptp_weights_map)
                locations_weights_ = ptp_weights_map(locations_weights_)

            if ptp_high:
                # Set to zero weights for locations with too high ptp values
                locations_weights_[self.locations_ptps > ptp_high] = 0

            locations_weights_ += self.normalize_weights(weights=locations_weights_)
            locations_weights += locations_weights_

        locations_weights = np.nan_to_num(locations_weights, nan=0)
        return weights, locations_weights

    def sample(self, size):
        """ Get exactly `size` locations. """
        if size == 0:
            return np.zeros((0, 9), np.int32)
        if self.threshold == 0.0:
            sampled = self._sample(size)
        else:
            accumulated = 0
            sampled_list = []

            while accumulated < size:
                sampled = self._sample(size*2)
                condition = spatial_check_sampled(sampled, self.matrix, self.n_threshold)

                sampled_list.append(sampled[condition])
                accumulated += condition.sum()
            sampled = np.concatenate(sampled_list)[:size]

        buffer = np.empty((size, 9), dtype=np.int32)
        buffer[:, 0] = self.field_id
        buffer[:, 1] = self.label_id
        buffer[:, 2:] = sampled
        return buffer

    def _sample(self, size):
        idx = choices(population=range(self.n), weights=self.locations_weights, k=size)
        self.sampled_locations_ptps = self.locations_ptps[idx]
        sampled = self.locations[idx]

        if self.shift_height:
            shift = np.random.randint(low=-int(self.crop_height*self.shift_height[0]),
                                      high=-int(self.crop_height*self.shift_height[1]),
                                      size=(size, 1), dtype=np.int32)
            sampled[:, [3, 6]] += shift

        if self.spatial_shift:
            shapes_i = sampled[:, 4] - sampled[:, 1]
            shift_i = np.random.randint(low=-(shapes_i*self.spatial_shift[0][0]).astype(np.int32),
                                        high=-(shapes_i*self.spatial_shift[0][1]).astype(np.int32),
                                        size=(size, 1), dtype=np.int32)
            sampled[:, [1, 4]] += shift_i

            shapes_x = sampled[:, 5] - sampled[:, 2]
            shift_x = np.random.randint(low=-(shapes_x*self.spatial_shift[1][0]).astype(np.int32),
                                        high=-(shapes_x*self.spatial_shift[1][1]).astype(np.int32),
                                        size=(size, 1), dtype=np.int32)
            sampled[:, [2, 5]] += shift_x

            np.clip(sampled[:, 1], 0, self.field.shape[0] - self.crop_shape[0], out=sampled[:, 1])
            np.clip(sampled[:, 4], 0 + self.crop_shape[0], self.field.shape[0], out=sampled[:, 4])

            np.clip(sampled[:, 2], 0, self.field.shape[1] - self.crop_shape[1], out=sampled[:, 2])
            np.clip(sampled[:, 5], 0 + self.crop_shape[1], self.field.shape[1], out=sampled[:, 5])

        np.clip(sampled[:, 3], 0, self.field.depth - self.crop_height, out=sampled[:, 3])
        np.clip(sampled[:, 6], 0 + self.crop_height, self.field.depth, out=sampled[:, 6])
        return sampled


    def __repr__(self):
        return f'<HorizonSampler for {self.displayed_name}: '\
               f'weights_mode={self.weights_mode}, '\
               f'crop_shape={tuple(self.crop_shape)}, threshold={self.threshold}, '\
               f'shift_height={self.shift_height}, spatial_shift={self.spatial_shift}>'

    @property
    def orientation_matrix(self):
        orientation_matrix = super().orientation_matrix
        if self.horizon.is_carcass:
            orientation_matrix = self.horizon.matrix_enlarge(orientation_matrix, 9)
        return orientation_matrix

@njit
def get_locations_ptps(locations, horizon_full_matrix):
    """ Helper method for the :class:`HorizonSampler`.
    Extracts ptp values of horizon depths on locations.

    Parameters
    ----------
    locations : np.ndarray
        Sampler locations.
    horizon_full_matrix : np.ndarray
        Horizon full matrix where `zero_traces` filled with np.nan value.
    """
    locations_ptps = np.empty((len(locations)), dtype=np.float32)

    for i, (_, i_start, x_start, _, i_end, x_end, _) in enumerate(locations):
        depths = horizon_full_matrix[i_start:i_end, x_start:x_end]
        min_, max_ = np.nanmin(depths), np.nanmax(depths)
        locations_ptps[i] = max_ - min_

    return locations_ptps

class FaultSampler(BaseSampler):
    """ Generator of crop locations, based on a single fault. Not intended to be used directly, see `SeismicSampler`.
    Makes locations that:
        - start from the labeled point on fault
        - don't go beyond cube limits

    Locations are produced as np.ndarray of (size, 9) shape with following columns:
        (field_id, label_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).
    Location is randomized in (-0.4*shape, 0.4*shape) range.

    For sampling, we randomly choose `size` rows from `locations`. If some of the sampled locations does not fit the
    `threshold` constraint or it is imposible to make crop of defined shape, resample until we get exactly
    `size` locations.

    Parameters
    ----------
    fault : Fault
        Fault to base sampler on.
    crop_shape : tuple
        Shape of crop locations to generate.
    threshold : float
        Minimum proportion of labeled points in each sampled location.
    ranges : sequence, optional
        Sequence of three tuples of two ints or `None`s.
        If tuple of two ints, then defines ranges of sampling along this axis.
        If None, then field limits are used (no constraints).
        Note that we actually use only the first two elements, corresponding to spatial ranges.
    field_id, label_id : int
        Used as the first two columns of sampled values.
    extend : bool
        Create locations in non-labeled slides between labeled slides.
    transpose : bool
        Create transposed crop locations or not.
    """
    dim = 2 + 1 + 6 # dimensionality of sampled points: field_id and label_id, orientation, locations

    def __init__(self, fault, crop_shape, threshold=0, ranges=None, extend=True, transpose=False,
                 field_id=0, label_id=0, **kwargs):
        field = fault.field

        self.points = fault.points
        self.nodes = fault.nodes if hasattr(fault, 'nodes') else None
        self.direction = fault.direction
        self.transpose = transpose

        self.locations = self._make_locations(field, crop_shape, ranges, threshold, extend)

        self.kwargs = kwargs

        self.field_id = field_id
        self.label_id = label_id

        self.field = field
        self.name = field.short_name
        self.displayed_name = fault.short_name
        super().__init__(self)

    @property
    def interpolated_nodes(self):
        """ Create locations in non-labeled slides between labeled slides. """
        slides = np.unique(self.nodes[:, self.direction])
        locations = []
        for i, slide in enumerate(slides):
            left = slides[max(i-1, 0)]
            right = slides[min(i+1, len(slides)-1)]
            chunk = self.nodes[self.nodes[:, self.direction] == slide]
            for j in range(left, right):
                chunk[:, self.direction] = j
                locations += [chunk.copy()]
        return np.concatenate(locations, axis=0)

    def _make_locations(self, field, crop_shape, ranges, threshold, extend):
         # Parse parameters
        ranges = ranges if ranges is not None else [None, None, None]
        ranges = [item if item is not None else [0, c]
                  for item, c in zip(ranges, field.shape)]
        ranges = np.array(ranges)

        crop_shape = np.array(crop_shape)
        crop_shape_t = crop_shape[[1, 0, 2]]
        n_threshold = np.int32(np.prod(crop_shape) * threshold)

        if self.nodes is not None:
            nodes = self.interpolated_nodes if extend else self.nodes
        else:
            nodes = self.points

        # Keep only points, that can be a starting point for a crop of given shape
        i_mask = ((ranges[:2, 0] < nodes[:, :2]).all(axis=1) &
                  ((nodes[:, :2] + crop_shape[:2]) < ranges[:2, 1]).all(axis=1))
        x_mask = ((ranges[:2, 0] < nodes[:, :2]).all(axis=1) &
                  ((nodes[:, :2] + crop_shape_t[:2]) < ranges[:2, 1]).all(axis=1))
        nodes = nodes[i_mask | x_mask]

        # Transform points to (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop)
        directions = [0, 1] if self.transpose else [self.direction]

        buffer = np.empty((len(nodes) * len(directions), 7), dtype=np.int32)

        for i, direction, in enumerate(directions):
            start, end = i * len(nodes), (i+1) * len(nodes)
            shape = crop_shape if direction == 0 else crop_shape_t
            buffer[start:end, 1:4] = nodes - shape // 2
            buffer[start:end, 4:7] = buffer[start:end, 1:4] + shape
            buffer[start:end, 0] = direction

        self.n = len(buffer)
        self.crop_shape = crop_shape
        self.crop_shape_t = crop_shape_t
        self.crop_height = crop_shape[2]
        self.ranges = ranges
        self.threshold = threshold
        self.n_threshold = n_threshold
        return buffer

    def sample(self, size):
        """ Get exactly `size` locations. """
        if size == 0:
            return np.zeros((0, 9), np.int32)
        accumulated = 0
        sampled_list = []

        while accumulated < size:
            sampled = self._sample(size*4)
            condition = volumetric_check_sampled(sampled, self.points, self.crop_shape,
                                                 self.crop_shape_t, self.n_threshold)

            sampled_list.append(sampled[condition])
            accumulated += condition.sum()
        sampled = np.concatenate(sampled_list)[:size]

        buffer = np.empty((size, 9), dtype=np.int32)
        buffer[:, 0] = self.field_id
        buffer[:, 1] = self.label_id
        buffer[:, 2:] = sampled
        return buffer

    def _sample(self, size):
        idx = np.random.randint(self.n, size=size)
        sampled = self.locations[idx]
        i_mask = sampled[:, 0] == 0
        x_mask = sampled[:, 0] == 1

        for mask, shape in zip([i_mask, x_mask], [self.crop_shape, self.crop_shape_t]):
            high = np.floor(shape * 0.4)
            low = -high
            low[shape == 1] = 0
            high[shape == 1] = 1

            shift = np.random.randint(low=low, high=high, size=(mask.sum(), 3), dtype=np.int32)
            sampled[mask, 1:4] += shift
            sampled[mask, 4:] += shift

            sampled[mask, 1:4] = np.clip(sampled[mask, 1:4], 0, self.field.cube_shape - shape)
            sampled[mask, 4:7] = np.clip(sampled[mask, 4:7], 0 + shape, self.field.cube_shape)
        return sampled

    def __repr__(self):
        return f'<FaultSampler for {self.displayed_name}: '\
               f'crop_shape={tuple(self.crop_shape)}, threshold={self.threshold}>'

@njit
def spatial_check_points(points, matrix, crop_shape, i_mask, x_mask, threshold):
    """ Compute points, which would generate crops with more than `threshold` labeled pixels.
    For each point, we test two possible shapes (i-oriented and x-oriented) and check `matrix` to compute the
    number of present points. Therefore, each of the initial points can result in up to two points in the output.

    Used as one of the filters for points creation at sampler initialization.

    Parameters
    ----------
    points : np.ndarray
        Points in (i_start, x_start, h_start) format.
    matrix : np.ndarray
        Depth map in cube coordinates.
    crop_shape : tuple of two ints
        Spatial shape of crops to generate: (i_shape, x_shape).
    i_mask : np.ndarray
        For each point, whether to test i-oriented shape.
    x_mask : np.ndarray
        For each point, whether to test x-oriented shape.
    threshold : int
        Minimum amount of points in a generated crop.
    """
    shape_i, shape_x = crop_shape

    # Return inline, crossline, corrected_depth (mean across crop), and 0/1 as i/x flag
    buffer = np.empty((2 * len(points), 4), dtype=np.int32)
    counter = 0

    for (point_i, point_x, _), i_mask_, x_mask_ in zip(points, i_mask, x_mask):
        if i_mask_:
            sliced = matrix[point_i:point_i+shape_i, point_x:point_x+shape_x].ravel()
            present_mask = (sliced > 0)

            if present_mask.sum() >= threshold:
                h_mean = np.rint(sliced[present_mask].mean())
                buffer[counter, :] = point_i, point_x, np.int32(h_mean), np.int32(0)
                counter += 1

        if x_mask_:
            sliced = matrix[point_i:point_i+shape_x, point_x:point_x+shape_i].ravel()
            present_mask = (sliced > 0)

            if present_mask.sum() >= threshold:
                h_mean = np.rint(sliced[present_mask].mean())
                buffer[counter, :] = point_i, point_x, np.int32(h_mean), np.int32(1)
                counter += 1
    return buffer[:counter]



@njit
def spatial_check_sampled(locations, matrix, threshold):
    """ Remove points, which correspond to crops with less than `threshold` labeled pixels.
    Used as a final filter for already sampled locations: they can generate crops with
    smaller than `threshold` mask only due to the depth randomization.

    Parameters
    ----------
    locations : np.ndarray
        Locations in (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop) format.
    matrix : np.ndarray
        Depth map in cube coordinates.
    threshold : int
        Minimum amount of labeled pixels in a crop.

    Returns
    -------
    condition : np.ndarray
        Boolean mask for locations.
    """
    condition = np.ones(len(locations), dtype=np.bool_)

    for i, (_, i_start, x_start, h_start, i_stop,  x_stop,  h_stop) in enumerate(locations):
        sliced = matrix[i_start:i_stop, x_start:x_stop].ravel()
        present_mask = (h_start < sliced) & (sliced < h_stop)

        if present_mask.sum() < threshold:
            condition[i] = False
    return condition

@njit
def volumetric_check_sampled(locations, points, crop_shape, crop_shape_t, threshold):
    """ Remove points, which correspond to crops with less than `threshold` labeled pixels.
    Used as a final filter for already sampled locations: they can generate crops with
    smaller than `threshold`.

    Parameters
    ----------
    locations : np.ndarray
        Locations in (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop) format.
    points : points
        Fault points.
    crop_shape : np.ndarray
        Crop shape
    crop_shape_t : np.ndarray
        Tranposed crop shape
    threshold : int
        Minimum amount of labeled pixels in a crop.

    Returns
    -------
    condition : np.ndarray
        Boolean mask for locations.
    """
    condition = np.ones(len(locations), dtype=np.bool_)

    if threshold > 0:
        for i, (orientation, i_start, x_start, h_start, i_stop,  x_stop, h_stop) in enumerate(locations):
            shape = crop_shape if orientation == 0 else crop_shape_t
            mask_bbox = np.array([[i_start, i_stop], [x_start, x_stop], [h_start, h_stop]], dtype=np.int32)
            mask = np.zeros((shape[0], shape[1], shape[2]), dtype=np.int32)

            insert_fault_into_mask(mask, points, mask_bbox)
            if mask.sum() < threshold:
                condition[i] = False

    return condition

class SeismicSampler(Sampler):
    """ Mixture of samplers for multiple cubes with multiple labels.
    Used to sample crop locations in the format of
    (field_id, label_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).

    Parameters
    ----------
    labels : dict
        Dictionary where keys are cube names and values are lists of labels.
    proportions : sequence, optional
        Proportion of each cube in the resulting mixture.
    baseclass : type
        Class for initializing individual label samplers.
    crop_shape : tuple
        Shape of crop locations to generate.
    threshold : float
        Minimum proportion of labeled points in each sampled location.
    ranges : sequence, optional
        Sequence of three tuples of two ints or `None`s.
        If tuple of two ints, then defines ranges of sampling along this axis.
        If None, then field limits are used (no constraints).
        Note that we actually use only the first two elements, corresponding to spatial ranges.
    filtering_matrix : np.ndarray, optional
        Map of points to remove from potentially generated locations.
    shift_height : bool
        Whether to apply random shift to height locations of sampled horizon points or not.
    kwargs : dict
        Other parameters of initializing label samplers.
    """
    CLASS_TO_MODE = {
        GeometrySampler: ['geometry', 'cube', 'field'],
        HorizonSampler: ['horizon', 'surface'],
        FaultSampler: ['fault']
    }
    MODE_TO_CLASS = {mode : class_
                     for class_, mode_list in CLASS_TO_MODE.items()
                     for mode in mode_list}

    def __init__(self, labels, crop_shape, proportions=None, mode='geometry',
                 threshold=0.05, ranges=None, filtering_matrix=None, shift_height=True, **kwargs):
        baseclass = self.MODE_TO_CLASS[mode] if isinstance(mode, str) else mode

        names, field_names = {}, {}
        sampler = 0 & ConstantSampler(np.int32(0), dim=baseclass.dim)
        samplers = AugmentedDict({idx: [] for idx in labels.keys()})

        proportions = proportions or [1 / len(labels) for _ in labels]

        for (field_id, ((idx, list_labels), p)) in enumerate(zip(labels.items(), proportions)):
            list_labels = list_labels if isinstance(list_labels, (tuple, list)) else [list_labels]

            # Unpack parameters
            crop_shape_ = crop_shape[idx] if isinstance(crop_shape, dict) else crop_shape
            threshold_ = threshold[idx] if isinstance(threshold, dict) else threshold
            filtering_matrix_ = filtering_matrix[idx] if isinstance(filtering_matrix, dict) else filtering_matrix
            ranges_ = ranges[idx] if isinstance(ranges, dict) else ranges

            # Mixture for each cube
            cube_sampler = 0 & ConstantSampler(np.int32(0), dim=baseclass.dim)
            for label_id, label in enumerate(list_labels):
                label_sampler = baseclass(label, crop_shape=crop_shape_, threshold=threshold_,
                                          ranges=ranges_, filtering_matrix=filtering_matrix_,
                                          field_id=field_id, label_id=label_id, shift_height=shift_height,
                                          **kwargs)

                if label_sampler.n != 0:
                    cube_sampler = cube_sampler | label_sampler

                    samplers[idx].append(label_sampler)
                    names[(field_id, label_id)] = (idx, label.short_name)

            # Resulting mixture
            sampler = sampler | (p & cube_sampler)
            field_names[field_id] = idx

        self.sampler = sampler
        self.samplers = samplers
        self.names = names
        self.field_names = field_names

        self.crop_shape = crop_shape
        self.threshold = threshold
        self.proportions = proportions
        self.mode = mode
        self.baseclass = baseclass

    def create_weights(self, weights_mode='constant', weights_agg='max', **kwargs):
        """ Create weights by calling the corresponding method for each sampler in `self.samplers`.

        Parameters
        ----------
        weights_mode : AugmentedDict or str
            Weights mode for creating weights.
            If str, then `weights_mode` is applied for each sampler.
            If AugmentedDict, then for each sampler applied the corresponding mode.
        weights_agg : str
            Weights aggregation option. It is used to aggregate all weights on a crop to one location weight.
            It is provided directly to the :method:`get_locations_weights`.
        kwargs : dict
            Other parameters of creating samplers weights.

        Returns
        -------
        weights : AugmentedDict
        locations_weights : AugmentedDict
        """
        weights = AugmentedDict({idx: [] for idx in self.samplers})
        locations_weights = AugmentedDict({idx: [] for idx in self.samplers})

        if not isinstance(weights_mode, AugmentedDict):
            weights_mode_ = weights_mode
        if not isinstance(weights_agg, AugmentedDict):
            weights_agg_ = weights_agg

        for idx in self.samplers:
            for i, sampler in enumerate(self.samplers[idx]):
                if isinstance(weights_mode, AugmentedDict):
                    weights_mode_ = weights_mode[idx][i]
                if isinstance(weights_mode, AugmentedDict):
                    weights_agg_ = weights_agg[idx][i]

                # Create and save weights
                weights_, locations_weights_ = sampler.create_weights(weights_mode=weights_mode_,
                                                                      weights_agg=weights_agg_, **kwargs)
                weights[idx].append(weights_)
                locations_weights[idx].append(locations_weights_)

        return weights, locations_weights

    def set_weights(self, weights=None, weights_mode=None):
        """ Set weights of samplers with provided values.

        Under the hood, this method calls the `set_weights` method for each sampler in `self.samplers`.

        Parameters
        ----------
        weights : AugmentedDict, callable or np.ndarray
            Weights matrix for each sampler.
            Note that the `weights` parameter can be a weights matrix only in case there is
            only one sampler in `self.samplers`.
        weights_mode : AugmentedDict or str
            Weights mode for creating weights.
            If str, then `weights_mode` is applied for each sampler.
            If AugmentedDict, then for each sampler applied the corresponding mode.
        kwargs : dict
            Other parameters of setting samplers weights.
        """
        if not isinstance(weights, AugmentedDict):
            weights_ = weights
        if not isinstance(weights_mode, AugmentedDict):
            weights_mode_ = weights_mode

        for idx in self.samplers:
            for i, sampler in enumerate(self.samplers[idx]):
                if isinstance(weights, AugmentedDict):
                    weights_ = weights[idx][i]
                if isinstance(weights_mode, AugmentedDict):
                    weights_mode_ = weights_mode[idx][i]

                # Set weights
                sampler.set_weights(weights=weights_, weights_mode=weights_mode_)

    def sample(self, size):
        """ Generate exactly `size` locations. """
        return self.sampler.sample(size)

    def __call__(self, size):
        return self.sampler.sample(size)

    def to_names(self, id_array):
        """ Convert the first two columns of sampled locations into field and label string names. """
        return np.array([self.names[tuple(ids)] for ids in id_array])

    def __len__(self):
        return sum(len(sampler.locations) for sampler_list in self.samplers.values() for sampler in sampler_list)

    def __str__(self):
        msg = 'SeismicSampler:'
        for list_samplers, p in zip(self.samplers.values(), self.proportions):
            msg += f'\n    {list_samplers[0].field.short_name} @ {p}'
            for sampler in list_samplers:
                msg += f'\n        {sampler}'
        return msg

    def show_locations(self, **kwargs):
        """ Visualize on field map by using underlying `locations` structure. """
        data = []
        title = []
        xlabel = []
        ylabel = []

        for samplers_list in self.samplers.values():
            field = samplers_list[0].field

            data += [[sampler.orientation_matrix, field.zero_traces] for sampler in samplers_list]
            title += [f'{field.displayed_name}: {sampler.displayed_name}' for sampler in samplers_list]
            xlabel += [field.index_headers[0]] * len(samplers_list)
            ylabel += [field.index_headers[1]] * len(samplers_list)

        kwargs = {
            'cmap': [['Sampler', 'black']] * len(data),
            'alpha': [[1.0, 0.4]] * len(data),
            # 'ncols': ncols,
            # 'nrows': nrows,
            'title': title,
            'vmin': [[1, 0]] * len(data),
            'vmax': [[3, 1]] * len(data),
            'xlabel': xlabel,
            'ylabel': ylabel,
            'shapes': 1, # this parameter toggles additional subplot axes creation for further legend display
            'return_figure': True,
            **kwargs
        }

        fig = plot_image(data, **kwargs)

        legend_params = {
            'ax': fig.axes[len(data)],
            'color': ('purple','blue','red', 'white', 'gray'),
            'label': ('ILINES and CROSSLINES', 'only ILINES', 'only CROSSLINES', 'restricted', 'dead traces'),
            'size': 20,
            'loc': 10,
            'facecolor': 'silver',
        }

        MatplotlibPlotter.add_legend(**legend_params)

    def show_sampled(self, n=10000, binary=False, **kwargs):
        """ Visualize on field map by sampling `n` crop locations. """
        sampled = self.sample(n)

        data = []
        title = []
        for field_id in np.unique(sampled[:, 0]):
            field = self.samplers[field_id][0].field
            matrix = np.zeros_like(field.zero_traces, dtype=np.int32)

            sampled_ = sampled[sampled[:, 0] == field_id]
            for (_, _, _, point_i_start, point_x_start, _,
                          point_i_stop,  point_x_stop,  _) in sampled_:
                matrix[point_i_start:point_i_stop, point_x_start:point_x_stop] += 1
            if binary:
                matrix[matrix > 0] = 1
                kwargs['bad_values'] = ()

            field_data = [matrix, field.zero_traces]
            data.append(field_data)

            field_title = f'{field.displayed_name}: {len(sampled_)} points'
            title.append(field_title)

        kwargs = {
            'matrix_name': 'Sampled slices',
            'cmap': [['Reds', 'black']] * len(data),
            'alpha': [[1.0, 0.4]] * len(data),
            'title': title,
            'interpolation': 'bilinear',
            'xlabel': field.index_headers[0],
            'ylabel': field.index_headers[1],
            'shapes': 1, # this parameter toggles additional subplot axes creation for further legend display
            'return_figure': True,
            **kwargs
        }

        fig = plot_image(data, **kwargs)

        legend_params = {
            'ax': fig.axes[len(data)],
            'color': ('beige', 'salmon', 'grey'),
            'label': ('alive traces', 'sampled locations', 'dead traces'),
            'size': 20,
            'loc': 10,
            'facecolor': 'silver',
        }

        MatplotlibPlotter.add_legend(**legend_params)

    def show_weights(self, **kwargs):
        """ Visualize on field map by using underlying `weights` structure (if exists). """
        data = []
        title = []
        xlabel = []
        ylabel = []
        vmin = []
        vmax = []

        for samplers_list in self.samplers.values():
            field = samplers_list[0].field

            data_ = []
            for sampler in samplers_list:
                if hasattr(sampler, 'weights'):
                    if sampler.weights is not None:
                        data_.append(sampler.weights)
            if data_:
                data += data_

                title += [f'{field.displayed_name}: {sampler.displayed_name}, weights mode: {sampler.weights_mode}'
                        for sampler in samplers_list]
                xlabel += [field.index_headers[0]] * len(samplers_list)
                ylabel += [field.index_headers[1]] * len(samplers_list)

                vmin += [np.nanmin(sampler.weights) for sampler in samplers_list]
                vmax += [np.nanmax(sampler.weights) for sampler in samplers_list]

        kwargs = {
            'cmap': 'viridis',
            'alpha': 1.,
            'title': title,
            'vmin': vmin,
            'vmax': vmax,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'separate': True,
            'return_figure': False,
            'colorbar':True,
            **kwargs
        }

        if data:
            plot_image(data, **kwargs)

    def show_ptps(self, **kwargs):
        """ Visualize histogram by using underlying `locations_ptps` and `sampled_locations_ptps`
        structures (if exists).
        """
        data = []
        title = []
        xlabel = []
        ylabel = []

        for samplers_list in self.samplers.values():
            field = samplers_list[0].field

            data_ = []
            title_ = []
            xlabel_ = []
            y_label_ = []
            for sampler in samplers_list:
                title_prefix = f"{field.displayed_name}: {sampler.displayed_name}"

                if hasattr(sampler, 'locations_ptps'):
                    if hasattr(sampler, 'sampled_locations_ptps'):
                        data_.extend([sampler.locations_ptps, sampler.sampled_locations_ptps])
                        title_.extend([f"{title_prefix}\nLocations ptps", f"{title_prefix}\nSampled locations ptps"])
                        xlabel_ = ["ptp"] * 2
                        y_label_ = ["frequency"] * 2
                    else:
                        data_.append(sampler.locations_ptps)
                        title_.append(f"{title_prefix}\nLocations ptps")
                        xlabel_ = ["ptp"]
                        y_label_ = ["frequency"]

            if data_:
                data += data_
                title += title_

                xlabel += xlabel_ * len(samplers_list)
                ylabel += y_label_ * len(samplers_list)


        kwargs = {
            'color': 'firebrick',
            'mode': 'hist',
            'title': title,
            'grid': True,
            'separate': True,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'return_figure': False,
            **kwargs
        }

        if data:
            plot_image(data, **kwargs)



class BaseGrid:
    """ Determenistic generator of crop locations. """
    def __init__(self, crop_shape=None, batch_size=64,
                 locations=None, orientation=None, origin=None, endpoint=None, field=None, label_name='unknown'):
        self._iterator = None
        self.crop_shape = np.array(crop_shape)
        self.batch_size = batch_size

        if locations is None:
            self._make_locations()
        else:
            self.locations = locations
            self.orientation = orientation
            self.origin = origin
            self.endpoint = endpoint
            self.shape = endpoint - origin
            self.field = field
            self.field_name = field.short_name
            self.label_name = label_name

    def _make_locations(self):
        raise NotImplementedError('Must be implemented in sub-classes')

    def to_names(self, id_array):
        """ Convert the first two columns of sampled locations into field and label string names. """
        return np.array([(self.field_name, self.label_name) for ids in id_array])

    # Iteration protocol
    @property
    def iterator(self):
        """ Iterator that generates batches of locations. """
        if self._iterator is None:
            self._iterator = self.make_iterator()
        return self._iterator

    def make_iterator(self):
        """ Iterator that generates batches of locations. """
        return (self.locations[i:i+self.batch_size] for i in range(0, len(self), self.batch_size))

    def __call__(self, batch_size=None):
        _ = batch_size
        return next(self.iterator)

    def next_batch(self, batch_size=None):
        """ Yield the next batch of locations. """
        _ = batch_size
        return next(self.iterator)

    def __len__(self):
        """ Total number of locations to be generated. """
        return len(self.locations)

    @property
    def length(self):
        """ Total number of locations to be generated. """
        return len(self.locations)

    @property
    def n_iters(self):
        """ Total number of iterations. """
        return np.ceil(len(self) / self.batch_size).astype(np.int32)

    # Concatenate multiple grids into one
    def join(self, other):
        """ Update locations of a current grid with locations from other instance of BaseGrid. """
        if not isinstance(other, BaseGrid):
            raise TypeError('Other should be an instance of `BaseGrid`')
        if self.field_name != other.field_name:
            raise ValueError('Grids should be for the same field!')

        locations = np.concatenate([self.locations, other.locations], axis=0)
        locations = np.unique(locations, axis=0)
        batch_size = min(self.batch_size, other.batch_size)

        if self.orientation == other.orientation:
            orientation = self.orientation
        else:
            orientation = 2

        self_origin = self.origin if isinstance(self, RegularGrid) else self.actual_origin
        other_origin = other.origin if isinstance(other, RegularGrid) else other.actual_origin
        origin = np.minimum(self_origin, other_origin)

        self_endpoint = self.endpoint if isinstance(self, RegularGrid) else self.actual_endpoint
        other_endpoint = other.endpoint if isinstance(other, RegularGrid) else other.actual_endpoint
        endpoint = np.maximum(self_endpoint, other_endpoint)

        label_name = other.label_name if isinstance(other, ExtensionGrid) else self.label_name

        return BaseGrid(locations=locations, batch_size=batch_size, orientation=orientation,
                        origin=origin, endpoint=endpoint, field=self.field, label_name=label_name)

    def __add__(self, other):
        return self.join(other)

    def __and__(self, other):
        return self.join(other)


    # Useful info
    def __repr__(self):
        return f'<BaseGrid for {self.field_name}: origin={tuple(self.origin)}, endpoint={tuple(self.endpoint)}>'

    @property
    def original_crop_shape(self):
        """ Original crop shape. """
        return self.crop_shape if self.orientation == 0 else self.crop_shape[[1, 0, 2]]

    @property
    def actual_origin(self):
        """ The upper leftmost point of all locations. """
        return self.locations[:, [3, 4, 5]].min(axis=0).astype(np.int32)

    @property
    def actual_endpoint(self):
        """ The lower rightmost point of all locations. """
        return self.locations[:, [6, 7, 8]].max(axis=0).astype(np.int32)

    @property
    def actual_shape(self):
        """ Shape of the covered by the grid locations. """
        return self.endpoint - self.origin

    @property
    def actual_ranges(self):
        """ Ranges of covered by the grid locations. """
        return np.array(tuple(zip(self.origin, self.endpoint)))

    def show(self, grid=True, markers=False, n_patches=None, **kwargs):
        """ Display the grid over field overlay.

        Parameters
        ----------
        grid : bool
            Whether to show grid lines.
        markers : bool
            Whether to show markers at location origins.
        n_patches : int
            Number of locations to display with overlayed mask.
        kwargs : dict
            Other parameters to pass to the plotting function.
        """
        n_patches = n_patches or int(np.sqrt(len(self))) // 5
        fig = self.field.geometry.show('zero_traces', cmap='Gray', colorbar=False, return_figure=True, **kwargs)
        ax = fig.axes

        if grid:
            spatial = self.locations[:, [3, 4]]
            for i in np.unique(spatial[:, 0]):
                sliced = spatial[spatial[:, 0] == i][:, 1]
                ax[0].vlines(i, sliced.min(), sliced.max(), colors='pink')

            spatial = self.locations[:, [3, 4]]
            for x in np.unique(spatial[:, 1]):
                sliced = spatial[spatial[:, 1] == x][:, 0]
                ax[0].hlines(x, sliced.min(), sliced.max(), colors='pink')

        if markers:
            ax[0].scatter(self.locations[:, 3], self.locations[:, 3], marker='x', linewidth=0.1, color='r')

        overlay = np.zeros_like(self.field.zero_traces)
        for n in range(0, len(self), len(self)//n_patches - 1):
            slc = tuple(slice(o, e) for o, e in zip(self.locations[n, [3, 4]], self.locations[n, [6, 7]]))
            overlay[slc] = 1
            ax[0].scatter(*self.locations[n, [3, 4]], marker='x', linewidth=3, color='g')

        kwargs = {
            'cmap': 'green',
            'alpha': 0.3,
            'colorbar': False,
            'matrix_name': 'Grid visualization',
            'ax': ax[0],
            **kwargs,
        }
        self.field.geometry.show(overlay, **kwargs)



class RegularGrid(BaseGrid):
    """ Regular grid over the selected `ranges` of cube, covering it with overlapping locations.
    Filters locations with less than `threshold` meaningful traces.

    Parameters
    ----------
    field : Field
        Field to create grid for.
    ranges : sequence
        Nested sequence, where each element is either None or sequence of two ints.
        Defines ranges to create grid for: iline, crossline, heights.
    crop_shape : sequence
        Shape of crop locations to generate.
    orientation : int
        Either 0 or 1. Defines orientation of a grid. Used in `locations` directly.
    threshold : number
        Minimum amount of non-dead traces in a crop to keep it in locations.
        If number in 0 to 1 range, then used as percentage.
    strides : sequence, optional
        Strides between consecutive crops. Only one of `strides`, `overlap` or `overlap_factor` should be specified.
    overlap : sequence, optional
        Overlaps between consecutive crops. Only one of `strides`, `overlap` or `overlap_factor` should be specified.
    overlap_factor : sequence, optional
        Ratio of overlap between cosecutive crops.
        Only one of `strides`, `overlap` or `overlap_factor` should be specified.
    batch_size : int
        Number of batches to generate on demand.
    field_id, label_id : int
        Used as the first two columns of sampled values.
    label_name : str, optional
        Name of the inferred label.
    locations : np.array, optional
        Pre-defined locations. If provided, then directly stored and used as the grid coordinates.
    """
    def __init__(self, field, ranges, crop_shape, orientation=0, strides=None, overlap=None, overlap_factor=None,
                 threshold=0, batch_size=64, field_id=-1, label_id=-1, label_name='unknown', locations=None):
        # Make correct crop shape
        orientation = field.geometry.parse_axis(orientation)
        crop_shape = np.array(crop_shape)
        crop_shape = crop_shape if orientation == 0 else crop_shape[[1, 0, 2]]

        if strides is not None:
            strides = np.array(strides)
            strides = strides if orientation == 0 else strides[[1, 0, 2]]

        # Make ranges
        ranges = [item if item is not None else [0, c] for item, c in zip(ranges, field.shape)]
        ranges = np.array(ranges)
        self.ranges = ranges

        # Infer from `ranges`
        self.origin = ranges[:, 0]
        self.endpoint = ranges[:, 1]
        self.shape = ranges[:, 1] - ranges[:, 0]

        # Make `strides`
        if (strides is not None) + (overlap is not None) + (overlap_factor is not None) > 1:
            raise ValueError('Only one of `strides`, `overlap` or `overlap_factor` should be specified!')
        overlap_factor = [overlap_factor] * 3 if isinstance(overlap_factor, (int, float)) else overlap_factor

        if strides is None:
            if overlap is not None:
                strides = [c - o for c, o in zip(crop_shape, overlap)]
            elif overlap_factor is not None:
                strides = [max(1, c // f) for c, f in zip(crop_shape, overlap_factor)]
            else:
                strides = crop_shape
        self.strides = np.array(strides)

        # Update threshold: minimum amount of non-empty traces
        if 0 < threshold < 1:
            threshold = int(threshold * crop_shape[0] * crop_shape[1])
        self.threshold = threshold

        self.field_id = field_id
        self.label_id = label_id
        self.orientation = orientation
        self.field = field
        self.field_name = field.short_name
        self.label_name = label_name
        self.unfiltered_length = None
        super().__init__(crop_shape=crop_shape, batch_size=batch_size, locations=locations, field=field,
                         orientation=self.orientation, origin=self.origin, endpoint=self.endpoint)

    @staticmethod
    def _arange(start, stop, stride, limit):
        grid = np.arange(start, stop, stride, dtype=np.int32)
        grid = np.unique(np.clip(grid, 0, limit))
        return np.sort(grid)

    def _make_locations(self):
        # Ranges for each axis
        i_args, x_args, h_args = tuple(zip(self.ranges[:, 0],
                                           self.ranges[:, 1],
                                           self.strides,
                                           self.field.cube_shape - self.crop_shape))
        i_grid = self._arange(*i_args)
        x_grid = self._arange(*x_args)
        h_grid = self._arange(*h_args)
        self.unfiltered_length = len(i_grid) * len(x_grid) * len(h_grid)

        # Create points: origins for each crop
        points = []
        for i, x in product(i_grid, x_grid):
            sliced = self.field.zero_traces[i:i+self.crop_shape[0],
                                            x:x+self.crop_shape[1]]
            # Number of non-dead traces
            if (sliced.size - sliced.sum()) > self.threshold:
                for h in h_grid:
                    points.append((i, x, h))
        points = np.array(points, dtype=np.int32)

        # Buffer: (cube_id, i_start, x_start, h_start, i_stop, x_stop, h_stop)
        buffer = np.empty((len(points), 9), dtype=np.int32)
        buffer[:, 0] = self.field_id
        buffer[:, 1] = self.label_id
        buffer[:, 2] = self.orientation
        buffer[:, [3, 4, 5]] = points
        buffer[:, [6, 7, 8]] = points
        buffer[:, [6, 7, 8]] += self.crop_shape
        self.locations = buffer


    def to_chunks(self, size, overlap=0.05):
        """ Split the current grid into chunks along `orientation` axis.

        Parameters
        ----------
        size : int
            Length of one chunk along the splitting axis.
        overlap : number
            If integer, then number of slices for overlapping between consecutive chunks.
            If float, then proportion of `size` to overlap between consecutive chunks.

        Returns
        -------
        iterator with instances of `RegularGrid`.
        """
        return RegularGridChunksIterator(grid=self, size=size, overlap=overlap)


    def __repr__(self):
        return f'<RegularGrid for {self.field.short_name}: '\
               f'origin={tuple(self.origin)}, endpoint={tuple(self.endpoint)}, crop_shape={tuple(self.crop_shape)}, '\
               f'orientation={self.orientation}>'



class RegularGridChunksIterator:
    """ Split regular grid into chunks along `orientation` axis. Supposed to be iterated over.

    Parameters
    ----------
    grid : RegularGrid
        Regular grid to split into chunks.
    size : int
        Length of one chunk along the splitting axis.
    overlap : number
        If integer, then number of slices for overlapping between consecutive chunks.
        If float, then proportion of `size` to overlap between consecutive chunks.
    """
    def __init__(self, grid, size, overlap):
        self.grid = grid
        self.size = size
        self.overlap = overlap

        self.step = int(size * (1 - overlap)) if isinstance(overlap, (float, np.float)) else size - overlap

    def __iter__(self):
        grid = self.grid

        for start in range(*grid.ranges[grid.orientation], self.step):
            stop = min(start + self.size, grid.field.shape[grid.orientation])

            chunk_ranges = grid.ranges.copy()
            chunk_ranges[grid.orientation] = [start, stop]

            # Filter points beyound chunk ranges along `orientation` axis
            mask = ((grid.locations[:, 3 + grid.orientation] >= start) &
                    (grid.locations[:, 6 + grid.orientation] <= stop))
            chunk_locations = grid.locations[mask]

            yield RegularGrid(locations=chunk_locations, ranges=chunk_ranges, strides=grid.strides,
                              orientation=grid.orientation, threshold=grid.threshold, field=grid.field,
                              crop_shape=grid.original_crop_shape, batch_size=grid.batch_size)

    def __len__(self):
        return len(range(*self.grid.ranges[self.grid.orientation], self.step))



class ExtensionGrid(BaseGrid):
    """ Generate locations to enlarge horizon from its boundaries both inwards and outwards.

    For each point on the boundary of a horizon, we test 4 possible directions and pick `top` best of them.
    Each location is created so that the original point is `stride` units away from the left/right edge of a crop.
    Only the locaitons that would potentially add more than `threshold` pixels remain.

    Refer to `_make_locations` method and comments for more info about inner workings.

    Parameters
    ----------
    horizon : Horizon
        Surface to extend.
    crop_shape : sequence
        Shape of crop locations to generate. Note that both iline and crossline orientations are used.
    stride : int
        Overlap with already known horizon for each location.
    top : int
        Number of the best locations to keep for each point.
    threshold : int
        Minimum amount of potentially added pixels for each locaiton.
    randomize : bool
        Whether to randomize the loop for computing the potential of each location.
    batch_size : int
        Number of batches to generate on demand.
    """
    def __init__(self, horizon, crop_shape, stride=16, batch_size=64, top=1, threshold=4, randomize=True):
        self.top = top
        self.stride = stride
        self.threshold = threshold
        self.randomize = randomize

        self.horizon = horizon
        self.field = horizon.field
        self.field_name = horizon.field.short_name
        self.label_name = horizon.short_name

        self.uncovered_before = None

        super().__init__(crop_shape=crop_shape, batch_size=batch_size)


    def _make_locations(self):
        # Get border points (N, 3)
        # Create locations for all four possible directions, stack into (4*N, 6)
        # Compute potential added area for each of the locations, while also updating coverage matrix
        # For each point, keep `top` of the best (potentially add more points) locations
        # Keep only those locations that potentially add more than `threshold` points
        #pylint: disable=too-many-statements

        crop_shape = self.crop_shape
        crop_shape_t = crop_shape[[1, 0, 2]]

        # True where dead trace / already covered
        coverage_matrix = self.field.zero_traces.copy().astype(np.bool_)
        coverage_matrix[self.horizon.full_matrix > 0] = True
        self.uncovered_before = coverage_matrix.size - coverage_matrix.sum()

        # Compute boundary points of horizon: both inner and outer borders
        border_points = np.stack(np.where(self.horizon.boundaries_matrix), axis=-1)
        heights = self.horizon.matrix[border_points[:, 0], border_points[:, 1]]

        # Shift heights up
        border_points += (self.horizon.i_min, self.horizon.x_min)
        heights -= crop_shape[2] // 2

        # Buffer for locations
        buffer = np.empty((len(border_points), 7), dtype=np.int32)
        buffer[:, 0] = 0
        buffer[:, [1, 2]] = border_points
        buffer[:, 3] = heights
        buffer[:, [4, 5]] = border_points
        buffer[:, 6] = heights

        # Repeat the same data along new 0-th axis: shift origins/endpoints
        buffer = np.repeat(buffer[np.newaxis, ...], 4, axis=0)

        # Crops with fixed INLINE, moving CROSSLINE: [-stride:-stride + shape]
        buffer[0, :, [2, 5]] -= self.stride
        np.clip(buffer[0, :, 2], 0, self.field.shape[1], out=buffer[0, :, 2])
        np.clip(buffer[0, :, 5], 0, self.field.shape[1], out=buffer[0, :, 5])
        buffer[0, :, [4, 5, 6]] += crop_shape.reshape(-1, 1)

        # Crops with fixed INLINE, moving CROSSLINE: [-shape + stride:+stride]
        buffer[1, :, [2, 5]] -= (crop_shape[1] - self.stride)
        np.clip(buffer[1, :, 2], 0, self.field.shape[1] - crop_shape[1], out=buffer[1, :, 2])
        np.clip(buffer[1, :, 5], 0, self.field.shape[1] - crop_shape[1], out=buffer[1, :, 5])
        buffer[1, :, [4, 5, 6]] += crop_shape.reshape(-1, 1)

        # Crops with fixed CROSSLINE, moving INLINE: [-stride:-stride + shape]
        buffer[2, :, [1, 4]] -= self.stride
        np.clip(buffer[2, :, 1], 0, self.field.shape[0], out=buffer[2, :, 1])
        np.clip(buffer[2, :, 4], 0, self.field.shape[0], out=buffer[2, :, 4])
        buffer[2, :,  [4, 5, 6]] += crop_shape_t.reshape(-1, 1)
        buffer[2, :, 0] = 1

        # Crops with fixed CROSSLINE, moving INLINE: [-shape + stride:+stride]
        buffer[3, :, [1, 4]] -= (crop_shape[1] - self.stride)
        np.clip(buffer[3, :, 1], 0, self.field.shape[0] - crop_shape[1], out=buffer[3, :, 1])
        np.clip(buffer[3, :, 4], 0, self.field.shape[0] - crop_shape[1], out=buffer[3, :, 4])
        buffer[3, :,  [4, 5, 6]] += crop_shape_t.reshape(-1, 1)
        buffer[3, :, 0] = 1

        if self.randomize:
            buffer = buffer[np.random.permutation(4)]
        # Array with locations for each of the directions
        # Each 4 consecutive rows are location variants for each point on the boundary
        buffer = buffer.transpose((1, 0, 2)).reshape(-1, 7)

        # Compute potential addition for each location
        potential = compute_potential(buffer, coverage_matrix, crop_shape)
        self.uncovered_best = coverage_matrix.size - coverage_matrix.sum()

        # Get argsort for each group of four
        argsort = potential.reshape(-1, 4).argsort(axis=-1)[:, -self.top:].reshape(-1)

        # Shift argsorts to original indices
        shifts = np.repeat(np.arange(0, len(buffer), 4, dtype=np.int32), self.top)
        indices = argsort + shifts

        # Keep only top locations; remove too locations with small potential
        potential = potential[indices]
        buffer = buffer[indices, :]

        mask = potential > self.threshold
        buffer = buffer[mask]

        # Correct the height
        np.clip(buffer[:, 3], 0, self.field.depth - crop_shape[2], out=buffer[:, 3])
        np.clip(buffer[:, 6], 0 + crop_shape[2], self.field.depth, out=buffer[:, 6])

        locations = np.empty((len(buffer), 9), dtype=np.int32)
        locations[:, [0, 1]] = -1
        locations[:, 2:9] = buffer
        self.locations = locations


    @property
    def uncovered_after(self):
        """ Number of points not covered in the horizon, if all of the locations would
        add their maximum potential amount of pixels to the labeling.
        """
        coverage_matrix = self.field.zero_traces.copy().astype(np.bool_)
        coverage_matrix[self.horizon.full_matrix > 0] = True

        for (i_start, x_start, _, i_stop, x_stop, _) in self.locations[:, 3:]:
            coverage_matrix[i_start:i_stop, x_start:x_stop] = True
        return coverage_matrix.size - coverage_matrix.sum()

    def show(self, markers=False, overlay=True, frequency=1, **kwargs):
        """ Display the grid over horizon overlay.

        Parameters
        ----------
        markers : bool
            Whether to show markers at location origins.
        overlay : bool
            Whether to show overlayed mask for locations.
        frequency : int
            Frequency of shown overlayed masks.
        kwargs : dict
            Other parameters to pass to the plotting function.
        """
        hm = self.horizon.full_matrix.astype(np.float32)
        hm[hm < 0] = np.nan
        fig = self.field.geometry.show(hm, cmap='Depths', colorbar=False, return_figure=True, **kwargs)
        ax = fig.axes

        self.field.geometry.show('zero_traces', ax=ax[0], cmap='Grey', colorbar=False, **kwargs)

        if markers:
            ax[0].scatter(self.locations[:, 3], self.locations[:, 4], marker='x', linewidth=0.1, color='r')

        if overlay:
            overlay = np.zeros_like(self.field.zero_traces)
            for n in range(0, len(self), frequency):
                slc = tuple(slice(o, e) for o, e in zip(self.locations[n, [3, 4]], self.locations[n, [6, 7]]))
                overlay[slc] = 1

            kwargs = {
                'cmap': 'blue',
                'alpha': 0.3,
                'colorbar': False,
                'title': f'Extension Grid on `{self.label_name}`',
                'ax': ax[0],
                **kwargs,
            }
            self.field.geometry.show(overlay, **kwargs)

@njit
def compute_potential(locations, coverage_matrix, shape):
    """ For each location, compute the amount of points it would potentially add to the labeling.
    If the shape of a location is not the same, as requested at grid initialization, we place `-1` value instead:
    that is filtered out later.
    """
    area = shape[0] * shape[1]
    buffer = np.empty((len(locations)), dtype=np.int32)

    for i, (_, i_start, x_start, _, i_stop, x_stop, _) in enumerate(locations):
        sliced = coverage_matrix[i_start:i_stop, x_start:x_stop].ravel()

        if len(sliced) == area:
            covered = sliced.sum()
            buffer[i] = area - covered
            coverage_matrix[i_start:i_stop, x_start:x_stop] = True
        else:
            buffer[i] = -1

    return buffer
