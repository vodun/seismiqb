""" !!. """
import numpy as np
from numba import njit

from .utils import _filtering_function
from .utility_classes import IndexedDict
from ..batchflow import Sampler, ConstantSampler



class HorizonSampler(Sampler):
    """ !!. """
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

        self.points = buffer
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
        """ !!. """
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

        buffer = np.empty((size, 7), dtype=object)
        buffer[:, 0] = self.name
        buffer[:, 1:] = sampled
        return buffer

    def _sample(self, size):
        idx = np.random.randint(self.n, size=size)
        sampled = self.points[idx]

        shift = np.random.randint(low=-int(self.height*0.9), high=-int(self.height*0.1),
                                  size=(size, 1), dtype=np.int32)
        sampled[:, [2, 5]] += shift
        return sampled

    @property
    def flags(self):
        """ !!. """
        shapes = self.points[:, 3:] - self.points[:, :3]
        return (shapes == self.shape_t).all(axis=1)

    @property
    def matrix(self):
        """ !!. """
        flags = self.flags
        i_matrix = np.full_like(self.horizon.full_matrix, 0.0)
        i_points = self.points[~flags]
        i_matrix[i_points[:, 0], i_points[:, 1]] = 1

        x_matrix = np.full_like(self.horizon.full_matrix, 0.0)
        x_points = self.points[flags]
        x_matrix[x_points[:, 0], x_points[:, 1]] = 2

        matrix = i_matrix + x_matrix
        matrix[matrix == 0.0] = np.nan
        return matrix



class SeismicSampler(Sampler):
    """ !!. """
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

    def sample(self, size):
        """ !!. """
        return self.sampler.sample(size)

    def __call__(self, size):
        return self.sampler.sample(size)


    def show(self, ncols=2, **kwargs):
        """ !!. """
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
        """ !!. """
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
    """ !!. """
    # shape is (i, x) only
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
    """ !!. """
    condition = np.ones(len(points), dtype=np.bool_)

    for i, (point_i_start, point_x_start, point_h_start,
            point_i_stop,  point_x_stop,  point_h_stop) in enumerate(points):
        sliced = matrix[point_i_start:point_i_stop, point_x_start:point_x_stop].ravel()
        present_mask = (point_h_start < sliced) & (sliced < point_h_stop)

        if present_mask.sum() < threshold:
            condition[i] = False
    return condition
