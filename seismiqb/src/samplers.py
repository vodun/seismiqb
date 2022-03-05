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

import numpy as np
from numba import njit

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


class GeometrySampler(BaseSampler):
    """ Generator of crop locations, based on a field. Not intended to be used directly, see `SeismicSampler`.
    Makes locations that:
        - start from the non-dead trace on a field, excluding those marked by `filtering_matrix`
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



class HorizonSampler(BaseSampler):
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
        self.displayed_name = horizon.short_name

        if shift_height:
            shift_height = shift_height if isinstance(shift_height, tuple) else (0.9, 0.1)
        self.shift_height = shift_height

        self.spatial_shift = spatial_shift
        super().__init__()

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
        idx = np.random.randint(self.n, size=size)
        sampled = self.locations[idx] # (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop)

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
               f'crop_shape={tuple(self.crop_shape)}, threshold={self.threshold}, '\
               f'shift_height={self.shift_height}, spatial_shift={self.spatial_shift}>'

    @property
    def orientation_matrix(self):
        orientation_matrix = super().orientation_matrix
        if self.horizon.is_carcass:
            orientation_matrix = self.horizon.matrix_enlarge(orientation_matrix, 9)
        return orientation_matrix



class FaultSampler(BaseSampler):
    """ Generator of crop locations, based on a single fault. Not intended to be used directly, see `SeismicSampler`.
    Makes locations that:
        - start from the labeled point on fault
        - don't go beyond cube limits

    Locations are produced as np.ndarray of (size, 9) shape with following columns:
        (field_id, label_id, orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).
    Location is randomized in (-0.4*shape, 0.4*shape) range.

    For sampling, we randomly choose `size` rows from `locations`. If some of the sampled locations does not fit the
    `threshold` constraint or it is impossible to make crop of defined shape, resample until we get exactly
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



class BaseGrid:
    """ Deterministic generator of crop locations. """
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
            self.ranges = np.array([origin, endpoint]).T
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


    def to_chunks(self, size, overlap=0.05, orientation=None):
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
        if orientation is None:
            if self.orientation != 2:
                orientation = self.orientation

        if not isinstance(size, (tuple, list)):
            size = [size, size]

            if orientation is not None:
                orientation = self.field.geometry.parse_axis(orientation)
                size[1 - orientation] = None

        if not isinstance(overlap, (tuple, list)):
            overlap = [overlap, overlap]
        return RegularGridChunksIterator(grid=self, size=size, overlap=overlap)


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
        Ratio of overlap between consecutive crops.
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

    def __repr__(self):
        return f'<RegularGrid for {self.field.short_name}: '\
               f'origin={tuple(self.origin)}, endpoint={tuple(self.endpoint)}, crop_shape={tuple(self.crop_shape)}, '\
               f'orientation={self.orientation}>'



class RegularGridChunksIterator:
    """ Split regular grid into chunks along `orientation` axis. Supposed to be iterated over.

    Parameters
    ----------
    grid : BaseGrid
        Grid to split into chunks.
    size : int or None or tuple of two ints or Nones
        Length of chunks along corresponding axes. `None` indicates that there is no chunking along the axis.
    overlap : number
        If integer, then number of slices for overlapping between consecutive chunks.
        If float, then proportion of `size` to overlap between consecutive chunks.
    """
    def __init__(self, grid, size, overlap):
        self.grid = grid

        size_i, size_x = size
        overlap_i, overlap_x = overlap

        if size_i is not None:
            step_i = int(size_i * (1 - overlap_i)) if isinstance(overlap_i, (float, np.float)) else size_i - overlap_i
        else:
            step_i = size_i = self.grid.shape[0]

        if size_x is not None:
            step_x = int(size_x * (1 - overlap_x)) if isinstance(overlap_x, (float, np.float)) else size_x - overlap_x
        else:
            step_x = size_x = self.grid.shape[1]

        self.size_i, self.size_x = size_i, size_x
        self.step_i, self.step_x = step_i, step_x

        self._iterator = None

    @property
    def iterator(self):
        """ Cached sequence of chunks. """
        # pylint: disable=protected-access
        if self._iterator is None:
            iterator = []
            grid = self.grid

            grid_i = RegularGrid._arange(*grid.ranges[0], self.step_i, max(0, grid.endpoint[0] - self.size_i))
            grid_x = RegularGrid._arange(*grid.ranges[1], self.step_x, max(0, grid.endpoint[1] - self.size_x))

            for start_i in grid_i:
                stop_i = start_i + self.size_i
                for start_x in grid_x:
                    stop_x = start_x + self.size_x

                    chunk_origin = np.array([start_i, start_x, grid.origin[2]])
                    chunk_endpoint = np.array([stop_i, stop_x, grid.endpoint[2]])

                    # Filter points beyond chunk ranges along `orientation` axis
                    mask = ((grid.locations[:, 3] >= start_i) &
                            (grid.locations[:, 6] <= stop_i)  &
                            (grid.locations[:, 4] >= start_x) &
                            (grid.locations[:, 7] <= stop_x))
                    chunk_locations = grid.locations[mask]

                    if len(chunk_locations):
                        chunk_grid = BaseGrid(field=grid.field, locations=chunk_locations,
                                              origin=chunk_origin, endpoint=chunk_endpoint, batch_size=grid.batch_size)
                        iterator.append(chunk_grid)
            self._iterator = iterator
        return self._iterator

    def __iter__(self):
        for chunk_grid in self.iterator:
            yield chunk_grid

    def __len__(self):
        return len(self.iterator)


class ExtensionGrid(BaseGrid):
    """ Generate locations to enlarge horizon from its boundaries both inwards and outwards.

    For each point on the boundary of a horizon, we test 4 possible directions and pick `top` best of them.
    Each location is created so that the original point is `stride` units away from the left/right edge of a crop.
    Only the locations that would potentially add more than `threshold` pixels remain.

    Refer to `_make_locations` method and comments for more info about inner workings.

    Parameters
    ----------
    horizon : Horizon
        Surface to extend.
    crop_shape : sequence
        Shape of crop locations to generate. Note that both iline and crossline orientations are used.
    stride : int
        Overlap with already known horizon for each location.
    threshold : int
        Minimum amount of potentially added pixels for each location.
    randomize : bool
        Whether to randomize the loop for computing the potential of each location.
    batch_size : int
        Number of batches to generate on demand.
    mode : {'best_for_each_independent', 'up', 'down', 'left', 'right',
            'vertical', 'horizontal', 'best_for_all', 'best_for_each'}
        Mode for directions of locations to generate.
        If mode is 'up', 'down', 'left' or 'right', then use only that direction.
        If 'vertical' ('horizontal'), then use up and down (right and left) directions.

        If 'best_for_all',  then select one direction for all points, based on total potentially added points.

        If 'best_for_each', then select direction for each point individually, based on total potentially added points.
        The potential of locations is computed sequentially: if one of the previous locations already covers area,
        it is considered to covered for all of the next potentials.

        If 'best_for_each_independent', then select direction for each point individually, based on total potentially
        added points. The potential of locations is computed independently of other locations.

    top : int
        Number of the best directions to keep for each point. Relevant only in `best_*` modes.
    """
    def __init__(self, horizon, crop_shape, stride=16, batch_size=64,
                 top=1, threshold=4, prior_threshold=8, randomize=True, mode='best_for_each'):
        self.top = top
        self.stride = stride
        self.threshold = threshold
        self.prior_threshold = prior_threshold
        self.randomize = randomize
        self.mode = mode

        self.horizon = horizon
        self.field = horizon.field
        self.field_name = horizon.field.short_name
        self.label_name = horizon.short_name

        self.uncovered_before = None
        self.locations_stats = {}

        allowed_directions = ['up', 'down', 'left', 'right']

        if self.mode in ['best_for_all', 'best_for_each', 'best_for_each_independent']:
            self.directions = allowed_directions
        elif self.mode in allowed_directions:
            self.directions = [self.mode]
        elif self.mode == 'vertical':
            self.directions = ['up', 'down']
        elif self.mode == 'horizontal':
            self.directions = ['left', 'right']
        else:
            raise ValueError('Provided wrong `mode` argument, for possible options look at the docstring.')

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

        # Buffer for locations (orientation, i_start, x_start, h_start, i_stop, x_stop, h_stop).
        buffer = np.empty((len(border_points), 7), dtype=np.int32)
        buffer[:, 0] = 0
        buffer[:, [1, 2]] = border_points
        buffer[:, 3] = heights
        buffer[:, [4, 5]] = border_points
        buffer[:, 6] = heights

        # Repeat the same data along new 0-th axis: shift origins/endpoints
        n_directions = len(self.directions)
        buffer = np.repeat(buffer[np.newaxis, ...], n_directions, axis=0)
        directions_iterator = 0

        if 'up' in self.directions:
            # Crops with fixed INLINE, moving CROSSLINE: [-stride:-stride + shape]
            buffer[directions_iterator, :, [2, 5]] -= self.stride

            np.clip(buffer[directions_iterator, :, 2], 0, self.field.shape[1],
                    out=buffer[directions_iterator, :, 2])
            np.clip(buffer[directions_iterator, :, 5], 0, self.field.shape[1],
                    out=buffer[directions_iterator, :, 5])

            buffer[directions_iterator, :, [4, 5, 6]] += crop_shape.reshape(-1, 1)
            directions_iterator += 1

        if 'down' in self.directions:
            # Crops with fixed INLINE, moving CROSSLINE: [-shape + stride:+stride]
            buffer[directions_iterator, :, [2, 5]] -= (crop_shape[1] - self.stride)

            np.clip(buffer[directions_iterator, :, 2], 0, self.field.shape[1] - crop_shape[1],
                    out=buffer[directions_iterator, :, 2])
            np.clip(buffer[directions_iterator, :, 5], 0, self.field.shape[1] - crop_shape[1],
                    out=buffer[directions_iterator, :, 5])

            buffer[directions_iterator, :, [4, 5, 6]] += crop_shape.reshape(-1, 1)
            directions_iterator += 1

        if 'left' in self.directions:
            # Crops with fixed CROSSLINE, moving INLINE: [-stride:-stride + shape]
            buffer[directions_iterator, :, [1, 4]] -= self.stride

            np.clip(buffer[directions_iterator, :, 1], 0, self.field.shape[0],
                    out=buffer[directions_iterator, :, 1])
            np.clip(buffer[directions_iterator, :, 4], 0, self.field.shape[0],
                    out=buffer[directions_iterator, :, 4])

            buffer[directions_iterator, :,  [4, 5, 6]] += crop_shape_t.reshape(-1, 1)
            buffer[directions_iterator, :, 0] = 1
            directions_iterator += 1

        if 'right' in self.directions:
            # Crops with fixed CROSSLINE, moving INLINE: [-shape + stride:+stride]
            buffer[directions_iterator, :, [1, 4]] -= (crop_shape[1] - self.stride)

            np.clip(buffer[directions_iterator, :, 1], 0, self.field.shape[0] - crop_shape[1],
                    out=buffer[directions_iterator, :, 1])
            np.clip(buffer[directions_iterator, :, 4], 0, self.field.shape[0] - crop_shape[1],
                    out=buffer[directions_iterator, :, 4])

            buffer[directions_iterator, :,  [4, 5, 6]] += crop_shape_t.reshape(-1, 1)
            buffer[directions_iterator, :, 0] = 1
            directions_iterator += 1

        update_coverage_matrix = self.mode not in ['best_for_all', 'best_for_each_independent']
        if self.randomize and update_coverage_matrix:
            buffer = buffer[np.random.permutation(n_directions)]

        # Array with locations for each of the directions
        # Each 4 consecutive rows are location variants for each point on the boundary
        buffer = buffer.transpose((1, 0, 2)).reshape(-1, 7)
        self.locations_stats['possible'] = buffer.shape[0]

        # Compute potential addition for each location
        # for 'best_for_all' and 'best_for_each_independent' modes potential calculated independently
        potential = compute_potential(locations=buffer, coverage_matrix=coverage_matrix,
                                      shape=crop_shape, stride=self.stride, prior_threshold=self.prior_threshold,
                                      update_coverage_matrix=update_coverage_matrix)

        if self.mode in ['best_for_each', 'best_for_each_independent']:
            # For each trace get the most potential direction index
            # Get argsort for each group of four
            argsort = potential.reshape(-1, n_directions).argsort(axis=-1)[:, -self.top:].reshape(-1)

            # Shift argsorts to original indices
            shifts = np.repeat(np.arange(0, len(buffer), n_directions, dtype=np.int32), self.top)
            indices = argsort + shifts

        elif self.mode == 'best_for_all':
            # Get indices of locations corresponding to the best direction
            # The best direction is a direction ('left', 'right', 'up' or 'down') with maximal potentially added traces
            positive_potential = potential.copy()
            positive_potential[positive_potential < 0] = 0

            best_direction_idx = np.argmax(positive_potential.reshape(-1, n_directions).sum(axis=0))
            indices = range(best_direction_idx, len(buffer), n_directions)

        else:
            indices = slice(None)

        # Keep only top locations; remove locations with too small potential if needed
        potential = potential[indices]
        buffer = buffer[indices, :]

        # Drop locations duplicates
        buffer, unique_locations_indices = np.unique(buffer, axis=0, return_index=True)
        potential = potential[unique_locations_indices]

        self.locations_stats['top_locations'] = buffer.shape[0]

        mask = potential > self.threshold
        buffer = buffer[mask]
        potential = potential[mask]
        self.locations_stats['selected'] = buffer.shape[0]

        # Correct the height
        np.clip(buffer[:, 3], 0, self.field.depth - crop_shape[2], out=buffer[:, 3])
        np.clip(buffer[:, 6], 0 + crop_shape[2], self.field.depth, out=buffer[:, 6])

        locations = np.empty((len(buffer), 9), dtype=np.int32)
        locations[:, [0, 1]] = -1
        locations[:, 2:9] = buffer
        self.locations = locations
        self.potential = potential.reshape(-1, 1)

        if update_coverage_matrix:
            self.uncovered_best = coverage_matrix.size - coverage_matrix.sum()
        else:
            # In the 'best_for_all' and 'best_for_each_independent' we don't update the `coverage_matrix``
            self.uncovered_best = self.uncovered_after


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
def compute_potential(locations, coverage_matrix, shape, stride, prior_threshold, update_coverage_matrix=True):
    """ For each location, compute the amount of points it would potentially add to the labeling.
    If the shape of a location is not the same, as requested at grid initialization, we place `-1` value instead:
    that is filtered out later. That is used to filter locations out of field bounds.

    For each location, we also check whether one of its sides (left/right/up/down) contains more
    than `prior_threshold` covered points.
    """
    area = shape[0] * shape[1]
    buffer = np.empty((len(locations)), dtype=np.int32)

    for i, (orientation, i_start, x_start, _, i_stop, x_stop, _) in enumerate(locations):
        sliced = coverage_matrix[i_start:i_stop, x_start:x_stop]

        if sliced.size == area:

            if orientation == 0:
                left, right = sliced[:stride, :].sum(), sliced[:-stride, :].sum()
                prior = max(left, right)
            elif orientation == 1:
                up, down = sliced[:, :stride].sum(), sliced[:, :-stride].sum()
                prior = max(up, down)

            if prior >= prior_threshold:
                covered = sliced.sum()
                buffer[i] = area - covered

                if update_coverage_matrix:
                    coverage_matrix[i_start:i_stop, x_start:x_stop] = True
            else:
                buffer[i] = -1
        else:
            buffer[i] = -1

    return buffer

class LocationsPotentialContainer:
    """ Container for saving history of `ExtensionGrid`.

    It saves locations and their potential from each grid provided in the method `update_grid`.
    Also, it removes repetitions from the grid locations and potentials.
    """
    def __init__(self, locations=None, potential=None):
        if locations is None:
            locations = np.empty(shape=(0, 9), dtype=np.int32)
        if potential is None:
            potential = np.empty(shape=(0, 1), dtype=np.int32)

        ncols = locations.shape[1]

        self.initial_dtype = locations.dtype
        self.locations_dtype = {'names': [f'col_{i}' for i in range(ncols)],
                                          'formats': ncols * [self.initial_dtype]}

        self.locations = locations.view(self.locations_dtype)
        self.potential = potential

        self.stats = {
            'repeated_locations': [],
            'total_repeated_locations': 0
        }

    def update_grid(self, grid):
        """ Update grid and container locations and potential.

        For the container, we update potentials for existing locations and safe new locations and their potentials.
        For the grid, we remove locations and potentials pairs that are saved in the container. It helps reduce
        locations amount and avoid repetitive locations processing such as model inference on these locations.
        """
        # Choose locations and potential pairs that are not in the container
        grid_locations = grid.locations.view(self.locations_dtype)

        repeated_locations = np.in1d(grid_locations, self.locations)
        repeated_potential = np.in1d(grid.potential, self.potential)
        repeated_locations_potential = repeated_locations & repeated_potential

        new_locations = grid_locations[~repeated_locations_potential]
        new_potential = grid.potential[~repeated_locations_potential]

        # Safe stats
        repeated_locations = len(grid_locations) - len(new_locations)
        self.stats['repeated_locations'].append(repeated_locations)
        self.stats['total_repeated_locations'] += repeated_locations

        # Update container: save new potentials for old locations and save new locations with their potential
        if len(new_locations) > 0:
            if len(self.locations) > 0:
                repeated_locations_history = np.in1d(self.locations, new_locations)
                locations = self.locations[~repeated_locations_history]
                potential = self.potential[~repeated_locations_history]

                self.locations = np.vstack([locations, new_locations])
                self.potential = np.vstack([potential, new_potential])
            else:
                self.locations = new_locations
                self.potential = new_potential

            new_locations = new_locations.view(self.initial_dtype).reshape(-1, grid.locations.shape[1])
        else:
            new_locations = np.empty(shape=(0, grid.locations.shape[1]))
            new_potential = np.empty(shape=(0, 1))

        # Update grid: set locations and grid with values that are not in the container
        grid.locations, grid.potential = new_locations, new_potential
