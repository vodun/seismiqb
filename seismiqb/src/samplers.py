""" !!. """
import numpy as np
from numba import njit

from .utils import _filtering_function
from .utility_classes import IndexedDict
from ..batchflow import Sampler, ConstantSampler



class HorizonSampler(Sampler):
    """ Generator of crop locations, based on a single horizon.
    Makes locations that:
        - start from the labeled point on horizon, exluding those marked by `filtering_matrix`
        - contain more than `threshold` labeled pixels inside
        - don't go beyond cube limits

    Locations are produced as np.ndarray of (size, 7) shape with following columns:
        (cube_id, i_start, x_start, h_start, i_stop, x_stop, h_stop).
    Depth location is randomized in (0.1*shape, 0.9*shape) range.

    Under the hood, we prepare `locations` attribute, that contains all already-filtered possible locations,
    and then randomly choose `size` rows for sampling. If some of the sampled locations does not fit the `threshold`
    constraint, resample until we get exactly `size` locations.

    Parameters
    ----------
    horizon : Horizon
        Horizon to base sampler on.
    shape : tuple
        Shape of crop locations to generate.
    threshold : float
        Minimum proportion of labeled points in each sampled location.
    filtering_matrix : np.ndarray, optional
        Map of points to remove from potentially generated locations.
    """
    dim = 7 # dimensionality of sampled points

    def __init__(self, horizon, shape, threshold=0.05, filtering_matrix=None, **kwargs):

        geometry = horizon.geometry

        shape = np.array(shape)
        shape_t = shape[[1, 0, 2]]
        n_threshold = np.int32(shape[0] * shape[1] * threshold)

        points = horizon.points.copy()
        full_matrix = horizon.full_matrix

        # Keep only points, that can be a starting point for a crop of given shape
        i_mask = ((points[:, :2] + shape[:2]) < geometry.cube_shape[:2]).all(axis=1)
        x_mask = ((points[:, :2] + shape_t[:2]) < geometry.cube_shape[:2]).all(axis=1)
        mask = i_mask | x_mask

        points = points[mask]
        i_mask = i_mask[mask]
        x_mask = x_mask[mask]

        # Apply filtration
        if filtering_matrix is not None:
            points = _filtering_function(points, filtering_matrix)

        # Keep only points, that produce crops with horizon larger than threshold
        points = check_points(points, full_matrix, shape[:2], i_mask, x_mask, n_threshold)

        # Transform points to (i_start, x_start, h_start, i_stop, x_stop, h_stop)
        buffer = np.empty((len(points), 6), dtype=np.int32)
        buffer[:, :3] = points[:, :3]
        buffer[points[:, 3] == 0, 3:] = shape
        buffer[points[:, 3] == 1, 3:] = shape_t
        buffer[:, 3:] += points[:, :3]

        # Store attributes
        self.locations = buffer
        self.n = len(buffer)

        self.shape = shape
        self.shape_t = shape_t
        self.height = shape[2]
        self.threshold = threshold
        self.n_threshold = n_threshold
        self.kwargs = kwargs

        self.horizon = horizon
        self.geometry = horizon.geometry
        self.full_matrix = full_matrix
        self.name = horizon.geometry.short_name
        super().__init__()

    def sample(self, size):
        """ Get exactly `size` locations. """
        if self.threshold == 0.0:
            sampled = self._sample(size)
        else:
            accumulated = 0
            sampled_list = []

            while accumulated < size:
                sampled = self._sample(size*2)
                condition = check_sampled(sampled, self.full_matrix, self.n_threshold)

                sampled_list.append(sampled[condition])
                accumulated += condition.sum()
            sampled = np.concatenate(sampled_list)[:size]
            self._counter = len(sampled_list)

        buffer = np.empty((size, 7), dtype=object)
        buffer[:, 0] = self.name
        buffer[:, 1:] = sampled
        return buffer

    def _sample(self, size):
        idx = np.random.randint(self.n, size=size)
        sampled = self.locations[idx]

        shift = np.random.randint(low=-int(self.height*0.9), high=-int(self.height*0.1),
                                  size=(size, 1), dtype=np.int32)
        sampled[:, [2, 5]] += shift
        return sampled

    def __repr__(self):
        return f'<HorizonSampler for {self.horizon.short_name}: '\
               f'shape={tuple(self.shape)}, threshold={self.threshold}>'

    @property
    def flags(self):
        """ False for iline-locations, True for xline-locations. """
        shapes = self.locations[:, 3:] - self.locations[:, :3]
        return (shapes == self.shape_t).all(axis=1)

    @property
    def matrix(self):
        """ Possible locations, mapped on geometry.
            - 0 where no locations can be sampled.
            - 1 where only iline-oriented crops can be sampled.
            - 2 where only xline-oriented crops can be sampled.
            - 3 where both types of crop orientations can be sampled.
        """
        flags = self.flags
        i_matrix = np.full_like(self.horizon.full_matrix, 0.0)
        i_points = self.locations[~flags]
        i_matrix[i_points[:, 0], i_points[:, 1]] = 1

        x_matrix = np.full_like(self.horizon.full_matrix, 0.0)
        x_points = self.locations[flags]
        x_matrix[x_points[:, 0], x_points[:, 1]] = 2

        matrix = i_matrix + x_matrix
        matrix[matrix == 0.0] = np.nan
        return matrix



class SeismicSampler(Sampler):
    """ Mixture of samplers on multiple cubes with label-samplers.
    Used to sample crop locations in (cube_id, i_start, x_start, h_start, i_stop, x_stop, h_stop) format,
    potentially with additional colums.

    Parameters
    ----------
    labels : dict
        Dictionary with nested lists that contains labels.
    shape : tuple
        Shape of crop locations to generate.
    proportions : sequence, optional
        Proportion of each cube in the resulting mixture.
    baseclass : type
        Class for label samplers.
    threshold : float
        Minimum proportion of labeled points in each sampled location.
    filtering_matrix : np.ndarray, optional
        Map of points to remove from potentially generated locations.
    kwargs : dict
        Other parameters of initializing label samplers.
    """
    def __init__(self, labels, shape, proportions=None, baseclass=HorizonSampler,
                 threshold=0.05, filtering_matrix=None, **kwargs):

        sampler = 0 & ConstantSampler(0, dim=baseclass.dim)
        samplers = IndexedDict({idx: [] for idx in labels.keys()})
        proportions = proportions or [1 / len(labels) for _ in labels]

        for (idx, list_labels), p in zip(labels.items(), proportions):
            cube_sampler = 0 & ConstantSampler(0, dim=baseclass.dim)
            shape_ = shape[idx] if isinstance(shape, dict) else shape
            threshold_ = threshold[idx] if isinstance(threshold, dict) else threshold
            filtering_matrix_ = filtering_matrix[idx] if isinstance(filtering_matrix, dict) else filtering_matrix

            for label in list_labels:
                label_sampler = baseclass(label, shape=shape_, threshold=threshold_,
                                          filtering_matrix=filtering_matrix_, **kwargs)
                cube_sampler = cube_sampler | label_sampler
                samplers[idx].append(label_sampler)

            sampler = sampler | (p & cube_sampler)

        self.sampler = sampler
        self.samplers = samplers
        self.proportions = proportions

        self.shape = shape
        self.threshold = threshold

    def sample(self, size):
        """ Generate exactly `size` locations. """
        return self.sampler.sample(size)

    def __call__(self, size):
        return self.sampler.sample(size)

    def __repr__(self):
        msg = 'SeismicSampler:'
        for list_samplers, p in zip(self.samplers.values(), self.proportions):
            msg += f'\n    {list_samplers[0].geometry.short_name} @ {p}'
            for sampler in list_samplers:
                msg += f'\n        {sampler}'
        return msg

    def show(self, ncols=2, **kwargs):
        """ Visualize on geometry map by using underlying `points` structure. """
        for idx, samplers in self.samplers.items():
            geometry = samplers[0].geometry
            images = [sampler.horizon.enlarge_carcass_image(sampler.matrix, 9)
                      if sampler.horizon.is_carcass else sampler.matrix
                      for sampler in samplers]
            images = [[image, geometry.zero_traces] for image in images]

            ncols_ = min(ncols, len(samplers))
            nrows = len(samplers) // ncols_ or 1

            kwargs = {
                'cmap': [['Sampler', 'black']] * len(samplers),
                'alpha': [[1.0, 0.4]] * len(samplers),
                'ncols': ncols_, 'nrows': nrows,
                'figsize': (16, 5*nrows),
                'title': [sampler.horizon.short_name for sampler in samplers],
                'suptitle_label': idx if len(samplers)>1 else '',
                'constrained_layout': True,
                'colorbar': False,
                'legend_label': ['ILINES and CROSSLINES', 'only ILINES', 'only CROSSLINES',
                                 'restricted', 'dead traces'],
                'legend_cmap': ['purple','blue','red', 'white', 'gray'],
                'legend_loc': 10,
                **kwargs
            }
            geometry.show(images, **kwargs)

    def visualize(self, n=10000, binary=False, **kwargs):
        """ Visualize on geometry map by sampling `n` crop locations. """
        sampled = self.sample(n)

        names = np.unique(sampled[:, 0])
        for name in names:
            geometry = self.samplers[name][0].geometry
            matrix = np.zeros_like(geometry.zero_traces, dtype=np.int32)

            sampled_ = sampled[sampled[:, 0] == name]
            for (_, point_i_start, point_x_start, _,
                    point_i_stop, point_x_stop, _) in sampled_:
                matrix[point_i_start:point_i_stop, point_x_start:point_x_stop] += 1
            if binary:
                matrix[matrix > 0] = 1
                kwargs['bad_values'] = ()

            kwargs = {
                'matrix_name': 'Sampled slices',
                'cmap': ['Reds', 'black'],
                'alpha': [1.0, 0.4],
                'figsize': (16, 8),
                'title': f'{name}: {len(sampled_)} points',
                'interpolation': 'bilinear',
                **kwargs
            }
            geometry.show((matrix, geometry.zero_traces), **kwargs)


@njit
def check_points(points, matrix, shape, i_mask, x_mask, threshold):
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
    shape : tuple of two ints
        Spatial shape of crops to generate: (i_shape, x_shape).
    i_mask : np.ndarray
        For each point, whether to test i-oriented shape.
    x_mask : np.ndarray
        For each point, whether to test x-oriented shape.
    threshold : int
        Minimum amount of points in a generated crop.
    """
    shape_i, shape_x = shape

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
def check_sampled(points, matrix, threshold):
    """ Remove points, which correspond to crops with less than `threshold` labeled pixels.
    Used as a final filter for already sampled points: they can generate crops with
    smaller than `threshold` mask due to depth randomization.

    Parameters
    ----------
    points : np.ndarray
        Points in (i_start, x_start, h_start, i_stop, x_stop, h_stop) format.
    matrix : np.ndarray
        Depth map in cube coordinates.
    threshold : int
        Minimum amount of labeled pixels in a crop.
    """
    condition = np.ones(len(points), dtype=np.bool_)

    for i, (point_i_start, point_x_start, point_h_start,
            point_i_stop,  point_x_stop,  point_h_stop) in enumerate(points):
        sliced = matrix[point_i_start:point_i_stop, point_x_start:point_x_stop].ravel()
        present_mask = (point_h_start < sliced) & (sliced < point_h_stop)

        if present_mask.sum() < threshold:
            condition[i] = False
    return condition


class BaseGrid:
    """ !!. """
    def __init__(self, crop_shape, batch_size=64):
        self.crop_shape = np.array(crop_shape)
        self.batch_size = batch_size

        self._make_locations()

    def _make_locations(self):
        raise NotImplementedError('Must be implemented in sub-classes')

    def __len__(self):
        return len(self.locations)

    @property
    def origin(self):
        """ !!. """
        return self.locations[:, [1, 2, 3]].min(axis=0)

    @property
    def endpoint(self):
        """ !!. """
        return self.locations[:, [1, 2, 3]].max(axis=0) + self.crop_shape

    @property
    def actual_shape(self):
        """ !!. """
        return self.endpoint - self.origin

    @property
    def actual_ranges(self):
        """ !!. """
        return np.array(tuple(zip(self.origin, self.endpoint)))

    @property
    def n_iters(self):
        """ !!. """
        return np.ceil(len(self) / self.batch_size).astype(np.int32)



from itertools import product

class RegularGrid(BaseGrid):
    """ !!. """
    def __init__(self, geometry, ranges, crop_shape, threshold=0,
                 strides=None, overlap=None, overlap_factor=None, batch_size=64):
        # Make ranges
        ranges = [item if item is not None else [0, c]
                  for item, c in zip(ranges, geometry.cube_shape)]
        ranges = np.array(ranges)

        if (ranges[:, 0] < 0).any():
            raise ValueError('Grid ranges must contain in the geometry!')
        if (ranges[:, 1] > geometry.cube_shape).any():
            raise ValueError('Grid ranges must contain in the geometry!')
        self.ranges = ranges
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

        self.geometry = geometry
        self.name = geometry.short_name
        self.unfiltered_length = None
        super().__init__(crop_shape=crop_shape, batch_size=batch_size)

    @staticmethod
    def _arange(start, stop, stride, limit):
        grid = np.arange(start, stop, stride)
        grid = np.unique(np.clip(grid, 0, limit))
        return np.sort(grid)

    def _make_locations(self):
        i_args, x_args, h_args = tuple(zip(self.ranges[:, 0],
                                           self.ranges[:, 1],
                                           self.strides,
                                           self.geometry.cube_shape - self.crop_shape))
        i_grid = self._arange(*i_args)
        x_grid = self._arange(*x_args)
        h_grid = self._arange(*h_args)
        self.unfiltered_length = len(i_grid) * len(x_grid) * len(h_grid)

        # Create points: origins for each crop
        points = []
        for i, x in product(i_grid, x_grid):
            sliced = self.geometry.zero_traces[i:i+self.crop_shape[0],
                                               x:x+self.crop_shape[1]]
            # Number of non-dead traces
            if (np.prod(sliced.shape) - sliced.sum()) > self.threshold: 
                for h in h_grid:
                    points.append((i, x, h))
        points = np.array(points, dtype=np.int32)

        # Buffer: (cube_id, i_start, x_start, h_start, i_stop, x_stop, h_stop)
        buffer = np.empty((len(points), 7), dtype=object)
        buffer[:, 0] = self.name
        buffer[:, [1, 2, 3]] = points
        buffer[:, [4, 5, 6]] = points
        buffer[:, [4, 5, 6]] += self.crop_shape
        self.locations = buffer

        if len(buffer) > 0:
            iterator = (buffer[i:i+self.batch_size]
                        for i in range(0, len(buffer), self.batch_size))
        else:
            iterator = iter(())
        self.iterator = iterator

    def __call__(self):
        """ !!. """
        return next(self.iterator)

    def next_batch(self):
        """ !!. """
        return next(self.iterator)


    def show(self, grid=True, markers=False, n_patches=None, **kwargs):
        """ !!. """
        n_patches = n_patches or int(np.sqrt(len(self))) // 5
        fig = self.geometry.show('zero_traces', cmap='Gray', colorbar=False, return_figure=True, **kwargs)
        ax = fig.axes

        if grid:
            spatial = self.locations[:, [1, 2]]
            for i in np.unique(spatial[:, 0]):
                sliced = spatial[spatial[:, 0] == i][:, 1]
                ax[0].vlines(i, sliced.min(), sliced.max(), colors='pink')

            spatial = self.locations[:, [1, 2]]
            for x in np.unique(spatial[:, 1]):
                sliced = spatial[spatial[:, 1] == x][:, 0]
                ax[0].hlines(x, sliced.min(), sliced.max(), colors='pink')

        if markers:
            ax[0].scatter(self.locations[:, 1], self.locations[:, 2], marker='x', linewidth=0.1, color='r')

        overlay = np.zeros_like(self.geometry.zero_traces)
        for n in range(0, len(self), len(self)//n_patches - 1):
            slc = tuple(slice(o, e) for o, e in zip(self.locations[n, [1, 2]], self.locations[n, [4, 5]]))
            overlay[slc] = 1
            ax[0].scatter(*self.locations[n, [1, 2]], marker='x', linewidth=3, color='g')

        kwargs = {
            'cmap': 'green',
            'alpha': 0.3,
            'colorbar': False,
            'matrix_name': 'Grid visualization',
            'ax': ax[0],
        }
        self.geometry.show(overlay, **kwargs)
