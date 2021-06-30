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
        mask_i = ((points[:, :2] + shape[:2]) < geometry.cube_shape[:2]).all(axis=1)
        mask_x = ((points[:, :2] + shape_t[:2]) < geometry.cube_shape[:2]).all(axis=1)
        mask = mask_i | mask_x

        points = points[mask]
        mask_i = mask_i[mask]
        mask_x = mask_x[mask]

        # Apply filtration
        if filtering_matrix is not None:
            points = _filtering_function(points, filtering_matrix)

        # Keep only points, that produce crops with horizon larger than threshold
        points = check_points(points, full_matrix, shape[:2], mask_i, mask_x, n_threshold)

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
def check_points(points, matrix, shape, mask_i, mask_x, threshold):
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
    mask_i : np.ndarray
        For each point, whether to test i-oriented shape.
    mask_x : np.ndarray
        For each point, whether to test x-oriented shape.
    threshold : int
        Minimum amount of points in a generated crop.
    """
    shape_i, shape_x = shape

    # Return inline, crossline, corrected_depth (mean across crop), and 0/1 as i/x flag
    buffer = np.empty((2 * len(points), 4), dtype=np.int32)
    counter = 0

    for (point_i, point_x, _), mask_i_, mask_x_ in zip(points, mask_i, mask_x):
        if mask_i_:
            sliced = matrix[point_i:point_i+shape_i, point_x:point_x+shape_x].ravel()
            present_mask = (sliced > 0)

            if present_mask.sum() >= threshold:
                h_mean = np.rint(sliced[present_mask].mean())
                buffer[counter, :] = point_i, point_x, np.int32(h_mean), np.int32(0)
                counter += 1

        if mask_x_:
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
