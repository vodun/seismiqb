""" Functions for generation of 2d and 3d synthetic seismic arrays.
"""
#pylint: disable=not-an-iterable
from collections import defaultdict

import numpy as np
from numba import njit, prange

import cv2
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from scipy.signal import ricker, convolve, resample

from ..plotters import MatplotlibPlotter, plot_image


@njit(parallel=True)
def compute_velocity_model(buffer, velocity_vector, horizon_matrices):
    """ !!. """
    i_range, x_range, depth = buffer.shape

    for i in prange(i_range):
        for j in range(x_range):
            indices = horizon_matrices[:, i, j]

            for k, velocity_value in enumerate(velocity_vector[:-1]):
                start, stop = indices[k], indices[k+1]
                buffer[i, j, start : stop] = velocity_value

            final = indices[-1]
            if final < depth:
                buffer[i, j, final:] = velocity_vector[-1]

    return buffer

@njit(parallel=True)
def compute_reflectivity_model(buffer, impedance_model):
    """ !!. """
    i_range, x_range, depth = buffer.shape

    for i in prange(i_range):
        for j in range(x_range):
            eps = 2 * impedance_model[i, j, depth // 2]
            for k in range(1, depth):
                previous_element, current_element = impedance_model[i, j, k-1], impedance_model[i, j, k]
                buffer[i, j, k] = ((current_element - previous_element) /
                                   (current_element + previous_element + eps))

            buffer[i, j, 0] = buffer[i, j, 1]
    return buffer

@njit
def inplace_add(matrix, indices_1, indices_2):
    """ !!. """
    for i_1, i_2 in zip(indices_1, indices_2):
        matrix[i_1, i_2] += 1
    return matrix


@njit
def find_depths(array, idx_x, depths):
    """ !!. """
    array_depth = array.shape[-1]
    output = np.empty_like(idx_x)

    for i in range(len(output)):
        idx_x_, depth = idx_x[i], depths[i]
        closest_idx, closest_diff = 0, array_depth

        for k in range(0, array_depth):
            idx = array[idx_x_, k]
            diff = abs(idx - depth)
            if diff < 1.:
                closest_idx = k
                break
            if diff <= closest_diff:
                closest_idx = k
                closest_diff = diff

        output[i] = closest_idx
    return output



class NewSyntheticGenerator:
    """ !!. """
    def __init__(self, rng=None, seed=None, **kwargs):
        # Random number generator. Should be used exclusively throughout the class for randomization
        self.rng = rng or np.random.default_rng(seed)

        self.velocity_vector = None
        self.num_horizons = None
        self.amplified_horizon_indices = None

        self.shape = None
        self.depth = None
        self.depth_intervals = None
        self.horizon_matrices = None
        self.fault_id = 0
        self.fault_point_clouds = defaultdict(list) # mapping from inline coordinate to list of point clouds
        self.fault_paths = {}

        # Models: arrays of `shape` with different geological attributes of the seismic
        self.velocity_model = None
        self.impedance_model = None
        self.density_model = None
        self.reflectivity_model = None
        self.synthetic = None

        # Properties
        self._horizon_mask = None
        self._fault_mask = None

        for key, value in kwargs.items():
            setattr(self, key, value)


    # Make velocity model: base for seismic generation
    def make_velocity_vector(self, num_horizons=10, limits=(5_000, 10_000), amplify=None,
                             randomization='uniform', randomization_scale=0.3):
        """ !!. """
        # TODO: maybe, add scale back to initial `limits` range?
        # Base velocity vector
        velocity_vector, delta = np.linspace(*limits, num=num_horizons, retstep=True, dtype=np.float32)

        # Generate and apply perturbation. Note the dtype
        if randomization == 'normal':
            perturbation = self.rng.standard_normal(size=num_horizons, dtype=np.float32)
        elif randomization == 'uniform':
            perturbation = 2 * self.rng.random(size=num_horizons, dtype=np.float32) - 1
        else:
            perturbation = 0

        velocity_vector += randomization_scale * delta * perturbation

        # Amplify some of the horizons
        amplified_horizon_indices = []
        if amplify:
            for depth, multiplier in amplify:
                index = round(depth * num_horizons)
                velocity_vector[index:] += delta * multiplier
                amplified_horizon_indices.append(index)

        # Store in the instance
        self.num_horizons = num_horizons
        self.velocity_vector = velocity_vector
        self.amplified_horizon_indices = amplified_horizon_indices
        return self

    def make_horizons(self, shape, num_horizons=None,
                      horizon_intervals='uniform', interval_randomization=None, interval_randomization_scale=0.1,
                      horizon_randomization1=True, randomization1_scale=0.25, num_nodes=10, interpolation_kind='cubic',
                      horizon_randomization2=False, randomization2_scale=0.1, digitize=True, n_bins=10,
                      locs_n_range=(2, 10), locs_scale_range=(5, 15), sample_size=None, blur_size=9, blur_sigma=2.):
        """ !!. """
        # TODO: maybe, reverse the direction?
        # Parse parameters
        shape = shape if len(shape) == 3 else (1, *shape)
        *spatial_shape, depth = shape
        num_horizons = num_horizons or self.num_horizons

        # Prepare intervals between horizons
        if horizon_intervals == 'uniform':
            depth_intervals = np.ones(num_horizons - 1, dtype=np.float32) / num_horizons
        elif isinstance(horizon_intervals, np.ndarray):
            depth_intervals = horizon_intervals

        # Slightly perturb intervals between horizons
        if interval_randomization == 'normal':
            perturbation = self.rng.standard_normal(size=num_horizons - 1, dtype=np.float32)
        elif interval_randomization == 'uniform':
            perturbation = 2 * self.rng.random(size=num_horizons - 1, dtype=np.float32) - 1
        else:
            perturbation = 0
        depth_intervals += (interval_randomization_scale / num_horizons) * perturbation

        depth_intervals = depth_intervals / depth_intervals.sum()

        # Make horizon matrices: starting from a zero-plane, move to the next `depth_interval` and apply randomizations
        horizon_matrices = [np.zeros(spatial_shape, dtype=np.float32)]

        for depth_interval in depth_intervals:
            previous_matrix = horizon_matrices[-1]
            next_matrix = previous_matrix + depth_interval

            if horizon_randomization1:
                perturbation_matrix = self.make_randomization1_matrix(shape=spatial_shape, num_nodes=num_nodes,
                                                                      interpolation_kind=interpolation_kind,
                                                                      rng=self.rng)
                next_matrix += randomization1_scale * depth_interval * perturbation_matrix

            if horizon_randomization2:
                perturbation_matrix = self.make_randomization2_matrix(shape=spatial_shape,
                                                                      locs_n_range=locs_n_range,
                                                                      locs_scale_range=locs_scale_range,
                                                                      sample_size=sample_size,
                                                                      blur_size=blur_size, blur_sigma=blur_sigma,
                                                                      digitize=digitize, n_bins=n_bins)
                next_matrix += randomization2_scale * depth_interval * perturbation_matrix

            horizon_matrices.append(next_matrix)

        horizon_matrices = np.array(horizon_matrices) * depth
        horizon_matrices = np.round(horizon_matrices).astype(np.int32)

        self.shape = shape
        self.depth = shape[-1]
        self.depth_intervals = depth_intervals
        self.horizon_matrices = horizon_matrices
        return self

    def make_velocity_model(self):
        """ !!. """
        buffer = np.empty(self.shape, dtype=np.float32)
        velocity_model = compute_velocity_model(buffer, self.velocity_vector, self.horizon_matrices)
        self.velocity_model = velocity_model
        return self


    # Modifications of velocity model
    def make_fault_2d(self, coordinates, max_shift=20, width=3, fault_id=0,
                      shift_vector=None, shift_sign=+1, mode='sin2', num_zeros=0.1,
                      perturb_peak=True, perturb_values=True,
                      ):
        """ !!. """
        # Parse parameters
        point_1, point_2 = coordinates
        point_1 = point_1 if len(point_1) == 3 else (0, *point_1)
        point_2 = point_2 if len(point_2) == 3 else (0, *point_2)

        point_1 = tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                        for c, s in zip(point_1, self.shape))
        point_2 = tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                        for c, s in zip(point_2, self.shape))

        (i1, x1, d1), (i2, x2, d2) = point_1, point_2
        # Make sure that i1 == i2!

        # Define crop of the slide that we work with: ranges along both axis
        x_low, x_high = 0, self.shape[1]
        d_low, d_high = min(d1, d2), max(d1, d2)
        x_len, d_len = x_high - x_low, d_high - d_low

        # Axis meshes: both (x_len, d_len) shape
        xmesh, dmesh = np.meshgrid(np.arange(x_low, x_high), np.arange(0, d_high - d_low), indexing='ij')

        # Compute the direction vector of a plane
        d1_, d2_ = d1 - d_low, d2 - d_low

        k = (x2 - x1) / (d2_ - d1_)
        b = (x1 * d2_ - x2 * d1_) / (d2_ - d1_)
        kx, kd = (k, 1)

        # Define distance to plane, use it to restrict coordinate shifts to an area
        # TODO: add curving to the plane (e.g. sine wave)
        distance = xmesh - kx * dmesh - b                     # range (-inf, + inf)
        indicator = np.clip(distance / width, 0, 1)           # range [0, 1]: fractions used to imitate transition

        # Make shift vector
        if shift_vector is not None:
            if len(shift_vector) != d_len:
                shift_vector = resample(shift_vector, d_len)
        else:
            shift_vector = self.make_shift_vector(d_len, mode=mode, num_zeros=num_zeros, rng=self.rng,
                                                  perturb_peak=perturb_peak, perturb_values=perturb_values)
        shift_vector *= shift_sign

        # Final shifts: (x_len, d_len) shaped matrix with shifts for each pixel
        shifts = max_shift * indicator * shift_vector.reshape(1, -1)

        # Transform velocity model: actually, optional
        # TODO: test if creating velocity model after adding faults is better
        if self.velocity_model is not None:
            map_coordinates(input=self.velocity_model[i1, x_low:x_high, d_low:d_high],
                            coordinates=(xmesh + kx * shifts, dmesh + kd * shifts),
                            output=self.velocity_model[i1, x_low:x_high, d_low:d_high],
                            mode='nearest')

        # Transform horizon matrices: compute indices after shift
        indices_array = (np.zeros_like(self.velocity_model) +
                         np.arange(0, self.depth, dtype=np.float32).reshape(1, 1, -1))

        map_coordinates(input=indices_array[i1, x_low:x_high, d_low:d_high],
                        coordinates=(xmesh + kx * shifts, dmesh + kd * shifts),
                        output=indices_array[i1, x_low:x_high, d_low:d_high],
                        mode='nearest')

        horizon_matrices_shifted = self.horizon_matrices.copy()
        for i, horizon_matrix in enumerate(self.horizon_matrices):
            # Find the position of original `depth` in the transformed indices
            idx_x = np.nonzero(horizon_matrix[i1])[0]
            depths = horizon_matrix[i1][idx_x]

            mask = (0 <= depths) & (depths < self.depth)
            idx_x, depths = idx_x[mask], depths[mask]

            horizon_matrices_shifted[i][i1][idx_x] = find_depths(indices_array[i1], idx_x, depths)

        # Transform fault point clouds on the same slide
        shifts_array = np.zeros(self.shape[1:], dtype=np.float32)
        shifts_array[x_low:x_high, d_low:d_high] = shifts

        updated_point_clouds = []
        for fault_id, (idx_x, depths) in self.fault_point_clouds[i1]:
            shifts_ = shifts_array[idx_x, depths]
            idx_x, depths = idx_x + kx * shifts_, depths + kd * shifts_
            idx_x, depths = np.round(idx_x).astype(np.int32), np.round(depths).astype(np.int32)

            updated_point_clouds.append((fault_id, (idx_x, depths)))
        self.fault_point_clouds[i1] = updated_point_clouds

        # Store the added fault point cloud
        idx_x, depths = np.nonzero((0 <= distance) & (distance / width < 1))
        idx_x, depths = idx_x.astype(np.int32), depths.astype(np.int32)
        depths += d_low
        mask = (d_low <= depths) & (depths < d_high)
        idx_x, depths = idx_x[mask], depths[mask]
        self.fault_point_clouds[i1].append((fault_id, (idx_x, depths)))

        self.horizon_matrices = horizon_matrices_shifted
        self.indices_array = indices_array
        return self


    def make_fault_3d(self, upper_points, lower_points, max_shift=20, width=3,
                      shift_sign=+1, mode='sin2', num_zeros=0.1, perturb_peak=True, perturb_values=True,):
        """ !!. """
        # Prepare points
        upper_points = tuple(tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                                   for c, s in zip(point, self.shape))
                             for point in upper_points)
        lower_points = tuple(tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                                   for c, s in zip(point, self.shape))
                             for point in lower_points)

        # Prepare paths
        upper_path = []
        for i in range(len(upper_points) - 1):
            path = self.make_path(upper_points[i], upper_points[i + 1])
            upper_path.extend(path)
        upper_path = sorted(set(upper_path), key=lambda item: item[0])

        lower_path = []
        for i in range(len(lower_points) - 1):
            path = self.make_path(lower_points[i], lower_points[i + 1])
            lower_path.extend(path)
        lower_path = sorted(set(lower_path), key=lambda item: item[0])

        # Prepare one common shift vector: resampled in each slice
        shift_vector = self.make_shift_vector(self.depth, mode=mode, num_zeros=num_zeros,
                                              perturb_peak=perturb_peak, perturb_values=perturb_values, rng=self.rng)

        for upper_point, lower_point in zip(upper_path, lower_path):
            self.make_fault_2d((upper_point, lower_point), max_shift=max_shift, width=width, fault_id=self.fault_id,
                               shift_vector=shift_vector, shift_sign=shift_sign)

        self.fault_paths[self.fault_id] = (upper_path, lower_path)
        self.fault_id += 1
        return self

    @staticmethod
    def make_path(point_1, point_2):
        """ !!. """
        (i1, x1, d1), (i2, x2, d2) = point_1, point_2

        n = i2 - i1
        t = np.arange(0, n + 1, dtype=np.int32, )

        i_array = i1 +                   t
        x_array = x1 + ((x2 - x1) / n) * t
        d_array = d1 + ((d2 - d1) / n) * t

        x_array = np.round(x_array).astype(np.int32)
        d_array = np.round(d_array).astype(np.int32)

        return list(zip(i_array, x_array, d_array))


    # Generate synthetic seismic, based on velocities
    def make_density_model(self, scale=0.01,
                           randomization='uniform', randomization_limits=(0.97, 1.03), randomization_scale=0.1):
        """ !!. """
        if randomization == 'uniform':
            a, b = randomization_limits
            perturbation = (scale * (b - a)) * self.rng.random(size=self.shape, dtype=np.float32) + (scale * a)
        elif randomization == 'normal':
            perturbation = scale * randomization_scale * self.rng.standard_normal(size=self.shape, dtype=np.float32)
        else:
            perturbation = scale * 1.0

        self.density_model = self.velocity_model * perturbation
        return self

    def make_impedance_model(self):
        """ !!. """
        self.impedance_model = self.velocity_model * self.density_model
        return self

    def make_reflectivity_model(self):
        """ !!.
        reflectivity = ((resistance[..., 1:] - resistance[..., :-1]) /
                        (resistance[..., 1:] + resistance[..., :-1]))
        """
        buffer = np.empty_like(self.impedance_model)
        reflectivity_model = compute_reflectivity_model(buffer, self.impedance_model)

        self.reflectivity_model = reflectivity_model
        return self

    def make_synthetic(self, ricker_width=5, ricker_points=50):
        """ !!. """
        wavelet = ricker(ricker_points, ricker_width)
        wavelet = wavelet.astype(np.float32).reshape(1, ricker_points)

        synthetic = np.empty_like(self.reflectivity_model)
        for i in range(self.shape[0]):
            cv2.filter2D(src=self.reflectivity_model[i], ddepth=-1, kernel=wavelet,
                         dst=synthetic[i], borderType=cv2.BORDER_CONSTANT)

        self.synthetic = synthetic
        return self

    def postprocess_synthetic(self, sigma=1., kernel_size=9, noise_mul=None):
        """ !!. """
        if sigma is not None:
            self.synthetic = self.apply_gaussian_filter_3d(self.synthetic, kernel_size=kernel_size, sigma=sigma)
        if noise_mul is not None:
            perturbation = self.rng.random(size=self.synthetic.shape, dtype=np.float32)
            self.synthetic += noise_mul * self.synthetic.std() * perturbation
        return self


    # Extraction
    @property
    def increasing_impedance_model(self):
        """ !!. """
        return ...

    def extract_horizons(self, indices='all', format='mask', width=3):
        """ !!. """
        #pylint: disable=redefined-builtin
        # Select appropriate horizons
        if isinstance(indices, (slice, list)):
            pass
        elif indices == 'all':
            indices = slice(1, None)
        elif indices == 'amplified':
            indices = self.amplified_horizon_indices
        elif 'top' in indices:
            k = int(indices[3:].strip())
            velocity_deltas = np.abs(np.diff(self.velocity_vector))
            indices = np.argsort(velocity_deltas)[::-1][:k]
        else:
            raise ValueError(f'Unsupported `indices={indices}`!')

        horizon_matrices = self.horizon_matrices[indices]

        #
        if 'matrix' in format:
            result = horizon_matrices
        elif 'instance' in format:
            result = ... #TODO
        elif 'mask' in format:
            indices = np.nonzero((0 <= horizon_matrices) & (horizon_matrices < self.depth))
            result = np.zeros(self.shape, dtype=np.float32)
            result[(*indices[1:], horizon_matrices[indices])] = 1

            if width is not None:
                kernel = np.ones(width, dtype=np.float32).reshape(1, width)

                for i in range(self.shape[0]):
                    cv2.filter2D(src=result[i], ddepth=-1, kernel=kernel,
                                 dst=result[i], borderType=cv2.BORDER_CONSTANT)
            result = np.clip(result, 0, 1)
        else:
            raise ValueError(f'Unsupported `format={format}`!')
        return result

    @property
    def horizon_instances(self):
        """ !!. """
        return self.extract_horizons(indices='all', format='instances')

    @property
    def horizon_mask(self):
        """ !!. """
        if self._horizon_mask is None:
            self._horizon_mask = self.extract_horizons(indices='all', format='mask')
        return self._horizon_mask


    def extract_faults(self, format='mask', width=5):
        """ !!. """
        if 'cloud' in format:
            ...
        elif 'instance' in format:
            ...
        elif 'mask' in format:
            result = np.zeros(self.shape, dtype=np.float32)

            for i in self.fault_point_clouds.keys():
                for _, (idx_x, depths) in self.fault_point_clouds[i]:
                    result[i][idx_x, depths] = 1

            if width is not None:
                kernel = np.ones(width, dtype=np.float32).reshape(width, 1)

                for i in range(self.shape[0]):
                    cv2.filter2D(src=result[i], ddepth=-1, kernel=kernel,
                                 dst=result[i], borderType=cv2.BORDER_CONSTANT)
            result = np.clip(result, 0, 1)

        else:
            raise ValueError(f'Unsupported `format={format}`!')
        return result

    @property
    def fault_mask(self):
        """ !!. """
        if self._fault_mask is None:
            self._fault_mask = self.extract_faults(format='mask')
        return self._fault_mask


    # Visualization
    def show_slide(self, loc=None, axis=0, velocity_cmap='jet', return_figure=False,
                   horizon_width=5, fault_width=7, **kwargs):
        """ !!. """
        #TODO: add the same functionality, as in `SeismicCropBatch.plot_roll`
        #TODO: use the same v_min/v_max for all locations
        try:
            loc = loc or self.shape[axis] // 2
            self._horizon_mask_ = self.extract_horizons(width=horizon_width)
            self._fault_mask_ = self.extract_faults(width=fault_width)

            # Non-overlapping data
            attributes = ['velocity_model', 'reflectivity_model', 'synthetic', '_horizon_mask_', '_fault_mask_']
            default_cmaps = [velocity_cmap, 'gray', 'gray', 'gray', 'gray']

            data, titles, cmaps = [], [], []
            for attribute, cmap in zip(attributes, default_cmaps):
                try:
                    image = np.take(getattr(self, attribute), indices=loc, axis=axis)
                    data.append([image])
                    titles.append(f'`{attribute.strip("_")}`')
                    cmaps.append([cmap])
                except AttributeError:
                    pass

            # Overlapping data
            horizon_mask = np.take(self._horizon_mask_, indices=loc, axis=axis)
            fault_mask = np.take(self._fault_mask_, indices=loc, axis=axis)

            attributes = ['velocity_model', 'synthetic']
            default_cmaps = [velocity_cmap, 'gray']
            for attribute, cmap in zip(attributes, default_cmaps):
                try:
                    image = np.take(getattr(self, attribute), indices=loc, axis=axis)
                    data.append([image, horizon_mask, fault_mask])
                    titles.append(f'`{attribute} overlayed`')
                    cmaps.append([cmap, 'red', 'purple'])
                except AttributeError:
                    pass

            # Reorder
            if len(data) == 7:
                order = [0, 5, 2, 6, 1, 3, 4]
                data = [data[item] for item in order]
                titles = [titles[item] for item in order]
                cmaps = [cmaps[item] for item in order]

            # Display images
            plot_params = {
                'suptitle': f'SyntheticGenerator slide: loc={loc}, axis={axis}',
                'title': titles,
                'cmap': cmaps,
                'colorbar': True,
                'ncols': 4,
                'scale': 0.5,
                'shapes': 1, # this parameter toggles additional subplot axes creation for further legend display
                'return_figure': True,
                **kwargs
            }
            fig = plot_image(data, **plot_params)

            # Display textual information on the same figure
            msg = f'shape = {self.shape}\nnum_horizons = {self.num_horizons}'
            msg += f'\nmin_interval = {self.depth_intervals.min() * self.depth:4.0f}'
            msg += f'\nmax_interval = {self.depth_intervals.max() * self.depth:4.0f}'
            msg += f'\nmean_interval = {self.depth_intervals.mean() * self.depth:4.0f}'
            legend_params = {
                'color': 'pink',
                'label': msg,
                'size': 14, 'loc': 10,
                'facecolor': 'pink',
            }
            MatplotlibPlotter.add_legend(ax=fig.axes[len(data)], **legend_params)

        finally:
            self._horizon_mask_, self._fault_mask_ = None, None

        if return_figure:
            return fig
        return None

    def show_stats(self, ):
        """ !!. """
        # d = np.arange(1, 1500 + 1)
        # plot_image(generator.reflectivity_model.mean(axis=(0, 1)), mode='curve')


    # Utilities and faster versions of common operations
    @staticmethod
    def make_gaussian_kernel_1d(kernel_size, sigma):
        """ !!. """
        kernel_1d = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size, dtype=np.float32)
        kernel_1d = np.exp(-0.5 * np.square(kernel_1d) / np.square(sigma))
        return kernel_1d / kernel_1d.sum()

    @staticmethod
    def make_randomization1_matrix(shape, num_nodes=10, interpolation_kind='cubic', rng=None):
        """ !!. """
        # Parse parameters
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        squeezed_shape = tuple(s for s in shape if s != 1)

        if len(squeezed_shape) == 1:
            num_nodes = (num_nodes,) if isinstance(num_nodes, int) else num_nodes
            interpolator_constructor = interp1d
        else:
            num_nodes = (num_nodes, num_nodes) if isinstance(num_nodes, int) else num_nodes
            interpolator_constructor = interp2d

        # Create interpolator on nodes
        nodes_grid =  [np.linspace(0, 1, num_nodes_, dtype=np.float32) for num_nodes_ in num_nodes]
        nodes_matrix = 2 * rng.random(size=num_nodes, dtype=np.float32).T - 1

        interpolator = interpolator_constructor(*nodes_grid, nodes_matrix, kind=interpolation_kind)

        # Apply interpolator on actual shape
        spatial_grid = [np.linspace(0, 1, s, dtype=np.float32) for s in squeezed_shape]
        spatial_matrix = interpolator(*spatial_grid).T
        return spatial_matrix.astype(np.float32).reshape(shape)

    @staticmethod
    def make_randomization2_matrix(shape, locs_n_range=(2, 10), locs_scale_range=(5, 15), sample_size=None,
                                   blur_size=9, blur_sigma=2., digitize=True, n_bins=10, rng=None):
        # Parse parameters
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        sample_size = sample_size or max(10000, np.prod(shape))

        # Generate locations for gaussians
        locs_n = np.random.randint(*locs_n_range)
        locs_low = [0 for _ in range(2*locs_n)]
        locs_high = [shape[i % 2] for i in range(2*locs_n)]

        locs = rng.uniform(low=locs_low, high=locs_high)

        # Sample points from all gaussians
        locs_scales = rng.uniform(*locs_scale_range, size=2*locs_n)
        sampled = rng.normal(loc=locs, scale=locs_scales, size=(sample_size, 2*locs_n))
        sampled = np.round(sampled).astype(np.int32)

        # Prepare indices
        indices_1, indices_2 = sampled[:, 0::2].reshape(-1), sampled[:, 1::2].reshape(-1)
        mask_1 = (0 <= indices_1) & (indices_1 < shape[0])
        mask_2 = (0 <= indices_2) & (indices_2 < shape[1])
        mask = mask_1 & mask_2

        indices_1 = indices_1[mask]
        indices_2 = indices_2[mask]

        # Create matrix: add ones at `indices`
        matrix = np.zeros(shape, dtype=np.float32)
        matrix = inplace_add(matrix, indices_1, indices_2)

        # Final blur and digitize
        kernel_1d = NewSyntheticGenerator.make_gaussian_kernel_1d(kernel_size=blur_size, sigma=blur_sigma)
        cv2.sepFilter2D(src=matrix, ddepth=-1, kernelX=kernel_1d.reshape(1, -1), kernelY=kernel_1d.reshape(-1, 1),
                        dst=matrix, borderType=cv2.BORDER_CONSTANT)
        matrix /= matrix.max()

        if digitize:
            bins = np.linspace(0, 1, n_bins + 1, dtype=np.float32)
            matrix = np.digitize(matrix, bins).astype(np.float32)
            matrix /= n_bins

        return matrix

    @staticmethod
    def make_shift_vector(num_points, mode='sin2', num_zeros=0.2, perturb_peak=True, perturb_values=True, rng=None):
        """ !!. """
        # TODO: dtypes? do we need them here (output of this function used for map_coordinates only)
        # Parse parameters
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        num_zeros = int(num_points * num_zeros) if isinstance(num_zeros, (float, np.floating)) else num_zeros
        num_nonzeros = num_points - num_zeros

        # Compute shifts for non-zeros
        if mode == 'sin2':
            x = np.arange(0, num_nonzeros)
            values = np.sin(np.pi * x / num_nonzeros) ** 2

        if perturb_values:
            step = 1 / num_nonzeros
            values += rng.uniform(-step, +step, num_nonzeros)

        # Define ranges of zeroes / actual values
        left = num_zeros // 2
        right = num_zeros - left

        if perturb_peak:
            peak_shift = rng.integers(-left // 2, right // 2)
            left += peak_shift
            right -= peak_shift

        # Make vector
        vector = np.zeros(num_points, dtype=np.float32)
        vector[left : -right] = values
        return vector

    @staticmethod
    def apply_gaussian_filter_3d(array, kernel_size=9, sigma=1.):
        """ !!. """
        kernel_1d = NewSyntheticGenerator.make_gaussian_kernel_1d(kernel_size=kernel_size, sigma=sigma)

        for i in range(array.shape[0]):
            cv2.sepFilter2D(src=array[i], ddepth=-1, kernelX=kernel_1d.reshape(1, -1), kernelY=kernel_1d.reshape(-1, 1),
                            dst=array[i], borderType=cv2.BORDER_CONSTANT)

        if array.shape[0] >= 3 * sigma * kernel_size:
            for j in range(array.shape[1]):
                cv2.filter2D(src=array[:, j], ddepth=-1, kernel=kernel_1d.reshape(-1, 1),
                             dst=array[:, j], borderType=cv2.BORDER_CONSTANT)
        return array



@njit
def _make_velocity_model_2d(velocities, surfaces, shape):
    """ Make 2d-velocity model.
    """
    array = np.zeros(shape=shape)
    for i in range(array.shape[0]):
        vec = array[i, :]
        for j, color in enumerate(velocities):
            low = np.minimum(surfaces[j][i], array.shape[-1])
            vec[low : ] = color
    return array


@njit
def _make_velocity_model_3d(velocities, surfaces, shape):
    """ Make 3d-velocity model.
    """
    array = np.zeros(shape=shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            vec = array[i, j, :]
            for k, color in enumerate(velocities):
                low = np.minimum(surfaces[k][i, j], array.shape[-1])
                vec[low : ] = color
    return array


class SyntheticGenerator():
    """ Class for generation of syhthetic velocity and density models and synthetic seismic - 2D/3D.
    """
    def __init__(self, rng=None, seed=None):
        """ Class for generation of syhthetic velocity and density models and synthetic seismic - 2D/3D.
        Can generate synthetic seismic with faults. Horizons and faults can be stored in instances of the
        class.

        Parameters
        ----------
        rng : np.random.Generator or None
            Generator of random numbers.
        seed : int or None
            Seed used for creation of random generator (check out `np.random.default_rng`).
        """
        self.rng = rng or np.random.default_rng(seed)
        self.velocities = None
        self.velocity_model = None
        self.density_model = None
        self.reflectivity_coefficients = None
        self.synthetic = None
        self.num_reflections = None
        self.reflection_surfaces = None
        self.horizon_heights = ()
        self.faults_coordinates = ()
        self.mask = None

    def make_velocities(self, num_reflections=200, velocity_limits=(900, 5400), horizon_heights=(1/4, 1/2, 2/3),
                        horizon_multipliers=(7, 5, 4)):
        """ Generate and store array of velocities. Roughly speaking, seismic slide is a stack of layers of constant
        velocities. This method generates the array of velocity-values, that are to be used later for making of
        velocity model.

        Parameters
        ----------
        num_reflections : int
            The number of reflective surfaces.
        velocity_limits : sequence
            Contains two floats. Velocities of layers in velocity model gradually change from the
            lower limit (first number) to the upper limit (second number) with some noise added.
        horizon_heights : sequence
            Each element is a float in [0, 1] interval that defines the depth (at unit coordinates) at which a
            horizon should be located.
        horizon_multipliers : sequence
            Each element is float mutiplier >> 1 (or << -1) controling the magnitide of gradients in velocity.
            The larger the gradients, the more prominient are the horizons. The argument should have the same length
            as `horizon_heights`-arg.
        """
        # Form array of velocities
        low, high = velocity_limits
        velocities = np.linspace(low, high, num_reflections + 1)

        # Add random perturbations
        velocity_delta = velocities[1] - velocities[0]
        velocities += self.rng.uniform(low=-velocity_delta, high=velocity_delta, size=(num_reflections + 1, ))

        # Add velocity gradients of large magnitide
        # to model horizons
        indices = (np.array(horizon_heights) * (num_reflections + 1)).astype(np.int32)
        for ix, multiplier in zip(indices, horizon_multipliers):
            velocities[ix:] += velocity_delta * multiplier

        self.horizon_heights = horizon_heights
        self.velocities = velocities
        self.num_reflections = num_reflections
        return self

    def make_upward_velocities(self):
        """ Make array of upward velocities (with only positive diffs) out of existing array of velocities.
        """
        self.upward_velocities = np.cumsum(np.abs(np.diff(self.velocities, prepend=0)))
        return self

    def _make_surfaces(self, num_surfaces, grid_shape, shape, kind='cubic', perturbation_share=0.25, shares=None):
        """ Make arrays representing heights of surfaces in a 3d/2d-array.

        Parameters
        ----------
        num_surfaces : int
            The number of resulting surfaces.
        grid_shape : tuple
            Shape of a grid of points used for interpolating surfaces.
        shape : tuple
            Shape of a 3d/2d array inside which the surfaces are created.
        kind : str
            Surfaces are interpolated from values on the grid of points. This is the type of interpolation
            to use (see `scipy.interpolate.intepr1d` for all possible options).
        perturbation_share : float
            Maximum allowed surface-perturbation w.r.t. the distance between subsequent surfaces.
        shares : np.ndarray
            Array representing height-distances between subsequent surfaces as shares of unit-interval.

        Returns
        -------
        np.ndarray
            Array of size num_surfaces X shape[:2] representing resulting surfaces-heights.
        """
        # Check shapes and select interpolation-method
        grid_shape = (grid_shape, ) if isinstance(grid_shape, int) else grid_shape
        if len(shape) != len(grid_shape) + 1:
            raise ValueError("`(len(shape) - 1)` should be equal to `len(grid_shape)`.")

        if len(shape) == 2:
            interp = interp1d
        elif len(shape) == 3:
            interp = interp2d
        else:
            raise ValueError('The function only supports the generation of 1d and 2d-surfaces.')

        # Make the grid
        grid = [np.linspace(0, 1, num_points) for num_points in grid_shape]

        # Make the first surface
        surfaces = [np.zeros(grid_shape)]
        shares = shares if shares is not None else np.ones((num_surfaces, ))
        shares = np.array(shares) / np.sum(shares)
        for delta_h in shares:
            epsilon = perturbation_share * delta_h

            # Make each surface in unit-terms
            surfaces.append(surfaces[-1] + delta_h * np.ones_like(surfaces[0])
                            + self.rng.uniform(low=-epsilon, high=epsilon, size=surfaces[0].shape))

        # Interpolate and scale each surface to cube-shape
        results = []
        for surface in surfaces:
            func = interp(*grid, surface, kind=kind)
            results.append((func(*[np.arange(num_points) / num_points for num_points in shape[:-1]])
                            * shape[-1]).astype(np.int).T)
        return np.array(results)

    @classmethod
    def make_velocity_model_(cls, velocities, surfaces, shape):
        """ Make 2d or 3d velocity model given shape, vector of velocities and level-surfaces.
        """
        if len(shape) in (2, 3):
            dim = len(shape)
        else:
            raise ValueError('Only supports the generation of 2d and 3d synthetic seismic.')

        _make_velocity_model = _make_velocity_model_2d if dim == 2 else _make_velocity_model_3d
        return _make_velocity_model(velocities, surfaces, shape)

    def make_velocity_model(self, shape=(50, 400, 800), grid_shape=(10, 10), perturbation_share=.2):
        """ Make 2d or 3d velocity model out of the array of velocities and store it in the class-instance.

        Parameters
        ----------
        shape : tuple
            [n_ilines X n_xlines X n_samples].
        grid_shape : tuple
            Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
        perturbation_share : float
            Sets the limit of random perturbation for surfaces' creation. The limit is set relative to the depth
            of a layer of constant velocity. The larger the value, more 'curved' are the horizons.
        """
        # Make and store surfaces-list to later use them as horizons
        surfaces = self._make_surfaces(self.num_reflections, grid_shape, perturbation_share=perturbation_share,
                                       shape=shape)
        self.reflection_surfaces = surfaces

        # Make and store velocity-model
        self.velocity_model = self.make_velocity_model_(self.velocities, surfaces, shape)
        return self

    def make_upward_velocity_model(self):
        """ Build velocity-model from array of upward velocities.
        """
        self.upward_velocity_model = self.make_velocity_model_(self.upward_velocities, self.reflection_surfaces,
                                                               self.velocity_model.shape)
        return self

    def _make_elastic_distortion(self, xs, n_points=10, zeros_share=0.2, kind='cubic',
                                 perturb_values=True, perturb_peak=True, random_invert=True):
        """ Generate a hump-shaped distortion [0, 1] -> [0, 1] and apply it to a set of points.
        The transformation has form f(x) = x + distortion * mul. It represents an
        elastic transform of an image represented by `xs`. The left and the right tails of the
        distortion can be filled with zeros. Also, the peak of the hump can be randomly shifted to
        left or right. In addition, when needed, the distortion itself can be randomly inverted.
        """
        points = np.linspace(0, 1, n_points)

        # Compute length of prefix of zeros
        n_zeros = int(n_points * zeros_share)
        if perturb_peak:
            delta_position = self.rng.integers(-n_zeros // 4, n_zeros // 4 + 1)
        else:
            delta_position = 0
        prefix_length = n_zeros // 2 + delta_position

        # Form the values-hump and perturb it if needed
        values = np.zeros((n_points, ))
        n_values = n_points - n_zeros
        half_hump = np.linspace(0, 1, n_values // 2 + 1 + n_values % 2)[1:]
        hump = np.concatenate([half_hump, half_hump[::-1][n_values % 2:]])
        if perturb_values:
            step = 2 / n_values
            hump += self.rng.uniform(-step / 2, step / 2, (n_values, ))
        values[prefix_length: prefix_length + len(hump)] = hump
        spline = interp1d(points, values, kind=kind)

        # Possibly invert the elastic transform
        if random_invert:
            if self.rng.choice([True, False]):
                return -spline(xs)
        return spline(xs)

    def _add_fault(self, fault_coordinates, num_points, max_shift, zeros_share, kind,
                   perturb_values, perturb_peak, random_invert, fetch_and_update_mask):
        """ Add fault to a velocity model.
        """
        x0, x1 = fault_coordinates[0][0], fault_coordinates[1][0]
        y0, y1 = fault_coordinates[0][1], fault_coordinates[1][1]
        x_low, x_high, y_low, y_high = (0, self.velocity_model.shape[0], min(y0, y1), max(y0, y1))

        # y-axis coordinate shift
        y0, y1 = y0 - y_low, y1 - y_low

        # Coeffs of the line equation x = ky + b
        k = (x1 - x0) / (y1 - y0)
        b = (x0 * y1 - x1 * y0) / (y1 - y0)
        kx, ky = (k, 1)

        # Make preparations for coordinate-map (i.e. elastic transform)
        xs, ys = np.meshgrid(np.arange(x_low, x_high), np.arange(0, y_high - y_low))

        # 0 to the left of the fault, 1 to the right
        indicator = (np.sign(xs - k * ys - b) + 1) / 2

        # Compute measure of closeness of a point to the fault-center
        closeness = self._make_elastic_distortion(ys / (y_high - y_low), n_points=num_points,
                                                  perturb_peak=perturb_peak, perturb_values=perturb_values,
                                                  kind=kind, zeros_share=zeros_share, random_invert=random_invert)

        # Compute vector field for a coordinate-map and apply the map to the seismic
        delta_xs, delta_ys = (max_shift * kx * indicator * closeness,
                              max_shift * ky * indicator * closeness)
        crop = self.velocity_model[x_low:x_high, y_low:y_high]
        crop_elastic = map_coordinates(crop.astype(np.float32),
                                       (xs + delta_xs, ys + delta_ys),
                                       mode='nearest').T
        self.velocity_model[x_low:x_high, y_low:y_high] = crop_elastic

        # Adjust mask if needed
        if fetch_and_update_mask is not None:
            if isinstance(fetch_and_update_mask, str):
                fetch_and_update_mask = {'mode': fetch_and_update_mask}
            fetch_and_update_mask['horizon_format'] = 'mask'

            # Make mask and apply the same coordinate-map to it
            mask = self.fetch_horizons(**fetch_and_update_mask)
            crop = mask[x_low:x_high, y_low:y_high]
            crop_elastic = map_coordinates(crop.astype(np.int32),
                                           (xs + delta_xs, ys + delta_ys),
                                           mode='nearest').T

            # Update the mask
            mask[x_low:x_high, y_low:y_high] = crop_elastic
            self.mask = mask

    def add_faults(self, faults_coordinates=None, num_points=10, max_shift=10,
                   zeros_share=0.6, kind='cubic', perturb_values=True,
                   perturb_peak=False, random_invert=False, fetch_and_update_mask='horizons'):
        """ Add faults to the velocity model. Faults are basically elastic transforms of patches of
        generated seismic images. Elastic transforms are performed through coordinates-transformation
        in depth-projection. Those are smooth maps [0, 1] -> [0, 1] described as f(x) = x + distortion.
        In current version, distortions are always hump-shaped. Almost all parameters of the function
        are used to define properties of the hump-shaped distortion.

        Parameters
        ----------
        faults_coordinates : sequence
            Iterable containing faults-coordinates in form ((x0, y0), (x1, y1)).
        num_points : int
            Number of points used for making coordinate-shifts for faults.
        max_shift : int
            Maximum vertical shift resulting from the fault.
        zeros_share : float
            Left and right tails of humps are set to zeros. This is needed to make
            transformations that are identical on the tails. The parameter controls the share
            of zero-values for tails.
        kind : str
            Kind of interpolation used for building coordinate-shifts.
        perturb_values : bool
            Whether to add random perturbations to a coordinate-shift hump.
        perturb_peak : bool
            If set True, the position of hump's peak is randomly moved.
        random_invert : bool
            If True, the coordinate-shift is defined as x - "hump" rather than x + "hump".
        fetch_and_update_mask : dict or None
            If not None or False, horizons-mask is also updated when faulting. If does not exist yet,
            will be created. Represents kwargs-dict for creating/fetching the mask.
        """
        if self.velocity_model is None:
            raise ValueError("You need to create velocity model first to add faults later.")

        faults_coordinates = faults_coordinates or tuple()
        self.faults_coordinates = faults_coordinates
        for fault in faults_coordinates:
            self._add_fault(fault, num_points, max_shift, zeros_share, kind, perturb_values,
                            perturb_peak, random_invert, fetch_and_update_mask)
        return self

    def make_density_model(self, density_noise_lims=(0.97, 1.3)):
        """ Make density model out of velocity model and store it in the class-instance.

        Parameters
        ----------
        density_noise_lims : tuple or None
            Density-model is given by (velocity model * noise). The param sets the limits for noise.
            If set to None, density-model is equal to velocity-model.
        """
        if density_noise_lims is not None:
            self.density_model = self.velocity_model * self.rng.uniform(*density_noise_lims,
                                                                        size=self.velocity_model.shape)
        else:
            self.density_model = self.velocity_model
        return self

    def make_reflectivity(self):
        """ Make reflectivity coefficients given velocity and density models.
        Velocities and reflectivity coefficients can be either 2d or 3d.
        """
        reflectivity = np.zeros_like(self.velocity_model)
        v_rho = self.velocity_model * self.density_model

        # Deal with all heights except h=0
        reflectivity[..., 1:] = (v_rho[..., 1:] - v_rho[..., :-1]) / (v_rho[..., 1:] + v_rho[..., :-1])

        # Deal with h=0
        reflectivity[..., 0] = reflectivity[..., 1]

        self.reflectivity_coefficients = reflectivity
        return self

    def make_synthetic(self, ricker_width=5, ricker_points=50):
        """ Generate and store 2d or 3d synthetic seismic. Synthetic seismic generation relies
        on generated velocity and density models. Hence, can be run only after `generate_velocities`,
        `generate_velocity_model` and `generate_density_model` methods.

        Parameters
        ----------
        ricker_width : float
            Width of the ricker-wave - `a`-parameter of `scipy.signal.ricker`.
        ricker_points : int
            Number of points in the ricker-wave - `points`-parameter of `scipy.signal.ricker`.
        """
        wavelet = ricker(ricker_points, ricker_width)

        # Colvolve reflection coefficients with wavelet to obtain synthetic
        # NOTE: convolution is performed only along depth-axis, hence additional axes in kernel
        kernel = wavelet.reshape((1, ) * (self.velocity_model.ndim - 1) + (-1, ))
        self.synthetic = convolve(self.reflectivity_coefficients, kernel, mode='same')
        return self

    def postprocess_synthetic(self, sigma=1.1, noise_mul=0.5):
        """ Simple postprocessing function for a seismic seismic, containing blur and noise.

        Parameters
        ----------
        sigma : float or None
            Sigma used for gaussian blur of the synthetic seismic.
        noise_mul : float or None
            If not None, gaussian noise scale by this number is applied to the synthetic.
        """
        if sigma is not None:
            self.synthetic = gaussian_filter(self.synthetic, sigma=sigma)
        if noise_mul is not None:
            self.synthetic += noise_mul * self.rng.random(self.synthetic.shape) * self.synthetic.std()
        return self

    def _enumerated_to_heights(self, mask):
        """ Convert enumerated mask to heights.
        """
        surfaces = []
        n_levels = np.max(mask)
        for i in range(1, n_levels + 1):
            heights = np.where(mask == i)[-1].reshape(self.reflection_surfaces[0].shape)
            surfaces.append(heights)
        return surfaces

    @staticmethod
    def _add_surface_to_mask(surface, mask, alpha=1):
        """ Add horizon-surface to mask.
        """
        indices = np.array((surface >= 0) & (surface < mask.shape[-1])).nonzero()
        mask[(*indices, surface[indices])] = alpha

    def _make_enumerated_mask(self, surfaces):
        """ Make enumerated mask from a sequence of surfaces. Each surfaces is marked by its ordinal
        number from `range(1, len(surfaces) + 1)` on a resulting mask.
        """
        mask = np.zeros_like(self.velocity_model)
        for i, horizon in enumerate(surfaces):
            self._add_surface_to_mask(horizon, mask, alpha=i + 1)
        return mask

    def fetch_horizons(self, mode='horizons', horizon_format='heights', width=5):
        """ Fetch some (or all) reflective surfaces.

        Parameters
        ----------
        mode : str
            Can be either 'horizons', 'all' ot 'top{K}'. When 'horizons', only horizon-surfaces
            (option `horizon_heights`) are returned. Choosing 'all' allows to return all of
            the reflections, while 'top{K}' option leads to fetching K surfaces correpsonding
            to K largest jumps in velocities-array.
        horizon_format : str
            Can be either 'heights' or 'mask'.
        width : int
            Width of horizons on resulting masks.

        Returns
        -------
        np.ndarray
            If format set to 'heights', array of shape n_horizons X n_ilines X n_xlines
            containing horizon-heights of selected horizons. If format set to 'mask',
            returns horizons-mask.
        """
        if isinstance(mode, (slice, list)):
            indices = mode
        elif mode == 'all':
            indices = slice(0, None)
        elif mode == 'horizons':
            indices = [int(self.reflection_surfaces.shape[0] * height_share)
                       for height_share in self.horizon_heights]
        elif 'top' in mode:
            top_k = int(mode.replace('top', ''))
            indices = np.argsort(np.abs(np.diff(self.velocities)))[::-1][:top_k]
        else:
            raise ValueError('Mode can be one of `horizons`, `all` or `top[k]`')
        surfaces = self.reflection_surfaces[indices]

        if horizon_format == 'heights':
            return surfaces
        if horizon_format == 'mask':
            if self.mask is not None:
                return self.mask
            mask = np.zeros_like(self.velocity_model)
            for surface in surfaces:
                self._add_surface_to_mask(surface, mask)

            # Add width to horizon-mask if needed
            if width is not None:
                if width > 1:
                    dim = len(mask.shape)
                    slc = (width // 2, ) * (dim - 1) + (slice(None, None), )
                    kernel = np.zeros((width, ) * dim)
                    kernel[slc] = 1
                    mask = binary_dilation(mask, kernel)
            return mask
        raise ValueError('Format can be either `heights` or `mask`')

    def fetch_faults(self, faults_format='mask', width=5):
        """ Fetch faults in N X 3 - format (cloud of points).

        Parameters
        ----------
        faults_format : str
            Can be either `point_cloud` or `mask`.
        width : int
            Width of faults on resulting mask - used when faults_format `mask`  is chosen.

        Returns
        -------
        list
            List containing arrays of shape N_points_in_fault X 3.
        """
        # Convert each fault to the point-cloud format
        point_clouds = []
        for fault in self.faults_coordinates:
            x0, x1 = fault[0][0], fault[1][0]
            y0, y1 = fault[0][1], fault[1][1]
            y_low, y_high = min(y0, y1), max(y0, y1)

            # Coeffs of the line equation x = ky + b
            k = (x1 - x0) / (y1 - y0)
            b = (x0 * y1 - x1 * y0) / (y1 - y0)

            heights = np.arange(y_low, y_high)
            ilines, xlines = np.zeros_like(heights), (np.rint(k * heights + b)).astype(np.int)
            point_cloud = np.stack([ilines, xlines, heights], axis=1)
            point_clouds.append(point_cloud)

        # Form masks out of point clouds if needed
        if faults_format == 'mask':
            mask = np.zeros_like(self.velocity_model)
            for point_cloud in point_clouds:
                xlines, heights = point_cloud[:, 1], point_cloud[:, 2]
                mask[xlines, heights] = 1

            # Add width to faults-mask if needed
            if width is not None:
                if width > 1:
                    dim = len(mask.shape)
                    slc = (slice(None, None), ) * (dim - 1) + (width // 2, )
                    kernel = np.zeros((width, ) * dim)
                    kernel[slc] = 1
                    mask = binary_dilation(mask, kernel)
            return mask
        if faults_format == 'point_cloud':
            return point_clouds
        raise ValueError('Format can be either `point_cloud` or `mask`')


def generate_synthetic(shape=(50, 400, 800), num_reflections=200, velocity_limits=(900, 5400), #pylint: disable=too-many-arguments
                       horizon_heights=(1/4, 1/2, 2/3), horizon_multipliers=(7, 5, 4), grid_shape=(10, 10),
                       perturbation_share=.2, density_noise_lims=(0.97, 1.3),
                       ricker_width=5, ricker_points=50, sigma=1.1, noise_mul=0.5,
                       faults_coordinates=None,
                       num_points_faults=10, max_shift=10, zeros_share_faults=0.6, fault_shift_interpolation='cubic',
                       perturb_values=True, perturb_peak=False, random_invert=False,
                       fetch_surfaces='horizons', geobodies_format=('mask', 'mask'),
                       geobodies_width=(5, 5), rng=None, seed=None):
    """ Generate synthetic 3d-cube and most prominent reflective surfaces ("horizons").

    Parameters
    ----------
    shape : tuple
        [n_ilines X n_xlines X n_samples].
    num_reflections : int
        The number of reflective surfaces.
    velocity_limits : tuple
        Contains two floats. Velocities of layers in velocity model gradually change from the
        lower limit (first number) to the upper limit (second number) with some noise added.
    horizon_heights : tuple
        Some reflections are sharper than the others - they represent seismic horizons. The tuple contains
        heights (in [0, 1]-interval) of sharp reflections.
    horizon_multipliers : tuple
        Mutipliers controling the magnitide of sharp jumps. Should have the same length as `horizon_heights`-arg.
    grid_shapes : tuple
        Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
    perturbation_share : float
        Sets the limit of random perturbation for surfaces' creation. The limit is set relative to the depth
        of a layer of constant velocity. The larger the value, more 'curved' are the horizons.
    density_noise_lims : tuple or None
        Density-model is given by (velocity model * noise). The param sets the limits for noise.
        If set to None, density-model is equal to velocity-model.
    ricker_width : float
        Width of the ricker-wave - `a`-parameter of `scipy.signal.ricker`.
    ricker_points : int
        Number of points in the ricker-wave - `points`-parameter of `scipy.signal.ricker`.
    sigma : float or None
        Sigma used for gaussian blur of the synthetic seismic.
    noise_mul : float or None
        If not None, gaussian noise scale by this number is applied to the synthetic.
    faults_coordinates : tuple or list
        Iterable containing faults-coordinates in form ((x0, y0), (x1, y1)).
    num_points_faults : int
        Number of points used for making coordinate-shifts for faults.
    max_shift : int
        Maximum vertical shift resulting from the fault.
    zeros_share_faults : float
        Left and right tails of humps are set to zeros. This is needed to make
        transformations that are identical on the tails. The parameter controls the share
        of zero-values for tails.
    fault_shift_interpolation : str
        Kind of interpolation used for building coordinate-shifts.
    perturb_values : bool
        Add random perturbations to a coordinate-shift hump.
    perturb_peak : bool
        If set True, the position of hump's peak is randomly moved.
    random_invert : bool
        If True, the coordinate-shift is defined as x - "hump" rather than x + "hump".
    fetch_surfaces : str
        Can be either 'horizons', 'all' or None. When 'horizons', only horizon-surfaces
        (option `horizon_heights`) are returned. Choosing 'all' allows to return all of
        the reflections, while 'topK' option leads to fetching K surfaces correpsonding
        to K largest jumps in velocities-array.
    rng : np.random.Generator or None
        Generator of random numbers.
    seed : int or None
        Seed used for creation of random generator (check out `np.random.default_rng`).
    geobodies_format : tuple or list
        Sequence containing return-format of horizons and faults. See docstrings
        of `SyntheticGenerator.fetch_horizons` and `SyntheticGenerator.fetch_faults`.
    geobodies_width : tuple or list
        Sequence containing width of horizons and faults on returned-masks. See docstrings
        of `SyntheticGenerator.fetch_horizons` and `SyntheticGenerator.fetch_faults`.

    Returns
    -------
    tuple
        Tuple (cube, horizons, faults); horizons can be None if `fetch_surfaces` is set to None.
    """
    if len(shape) in (2, 3):
        dim = len(shape)
    else:
        raise ValueError('The function only supports the generation of 2d and 3d synthetic seismic.')

    gen = (SyntheticGenerator(rng, seed)
           .make_velocities(num_reflections, velocity_limits, horizon_heights, horizon_multipliers)
           .make_velocity_model(shape, grid_shape, perturbation_share))

    # Add faults if needed and possible
    if faults_coordinates is not None:
        if len(faults_coordinates) > 0:
            if dim == 2:
                fetch_and_update = {'mode': fetch_surfaces, 'horizon_format': geobodies_format[0],
                                    'width': geobodies_width[0]}
                gen.add_faults(faults_coordinates, num_points_faults, max_shift, zeros_share_faults,
                               fault_shift_interpolation, perturb_values, perturb_peak, random_invert,
                               fetch_and_update)
            else:
                raise ValueError("For now, faults are only supported for dim = 2.")

    gen = (gen.make_density_model(density_noise_lims)
              .make_reflectivity()
              .make_synthetic(ricker_width, ricker_points)
              .postprocess_synthetic(sigma, noise_mul))

    return (gen.synthetic,
            gen.fetch_horizons(fetch_surfaces, horizon_format=geobodies_format[0], width=geobodies_width[0]),
            gen.fetch_faults(faults_format=geobodies_format[1], width=geobodies_width[1]))


def surface_to_points(surface):
    """ Make points-array by adding ilines-xlines columns and flattening the surface-column.
    No offset is added: ilines and xlines are assumed to be simple ranges 0..ilines_len.

    Parameters
    ----------
    surface : np.ndarray
        Array of heights representing the reflective surface in a generated cube.
    """
    n_ilines, n_xlines = surface.shape
    mesh = np.meshgrid(range(n_ilines), range(n_xlines), indexing='ij')
    points = np.stack([mesh[0].reshape(-1), mesh[1].reshape(-1),
                       surface.reshape(-1)], axis=1).astype(np.int)
    return points
