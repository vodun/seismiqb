""" Functions for generation of 2d and 3d synthetic seismic arrays.
"""
#pylint: disable=not-an-iterable, too-many-arguments, too-many-statements, redefined-builtin
from collections import defaultdict
from locale import DAY_1

import numpy as np
from numba import njit, prange

import cv2
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from scipy.signal import ricker, convolve, resample

from ..plotters import MatplotlibPlotter, plot_image



class SyntheticGenerator:
    """ !!. """
    def __init__(self, rng=None, seed=None, **kwargs):
        # Random number generator. Should be used exclusively throughout the class for randomization
        self.rng = rng or np.random.default_rng(seed)

        self.velocity_vector = None
        self.num_horizons = None
        self.amplified_horizon_indices = None

        self.shape = None
        self.depth = None
        self.padding = None
        self.shape_padded = None
        self.depth_padded = None
        self.finalized = False

        self.depth_intervals = None
        self.horizon_matrices = None

        self.fault_id = 0
        self.fault_coordinates = []
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
    def make_velocity_vector(self, num_horizons=10, limits=(5_000, 10_000),
                             randomization='uniform', randomization_scale=0.3,
                             amplify_probability=0.2, amplify_range=(2.0, 4.0), amplify_sign_probability=0.8):
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
        a, b = amplify_range[0], amplify_range[1] - amplify_range[0]

        amplified_horizon_indices = []
        for index in range(1, num_horizons):
            if self.rng.random() <= amplify_probability:
                multiplier = b*self.rng.random(dtype=np.float32) + a
                sign = +1 if self.rng.random() <= amplify_sign_probability else -1

                velocity_vector[index:] += delta * sign * multiplier
                amplified_horizon_indices.append(index)

        # Store in the instance
        self.num_horizons = num_horizons
        self.velocity_vector = velocity_vector
        self.amplified_horizon_indices = amplified_horizon_indices
        return self

    def make_horizons(self, shape, padding=(16, 32), num_horizons=None, horizon_intervals='uniform',
                      interval_randomization=None, interval_randomization_scale=0.1, interval_min=0.5,
                      randomization1_scale=0.25, num_nodes=10, interpolation_kind='cubic',
                      randomization2_scale=0.1, digitize=True, n_bins=10,
                      locs_n_range=(2, 10), locs_scale_range=(5, 15), sample_size=None, blur_size=9, blur_sigma=2.):
        """ !!. """
        # TODO: maybe, reverse the direction?
        # Parse parameters
        shape = shape if len(shape) == 3 else (1, *shape)
        padding = padding if isinstance(padding, (tuple, list)) else (padding, padding)
        padding = padding if len(padding) == 3 else (1, *padding)
        padding = tuple(p if s != 1 else 0 for s, p in zip(shape, padding))
        shape_padded = tuple(s + p for s, p in zip(shape, padding))

        *spatial_shape, depth = shape_padded
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
        depth_intervals = np.clip(depth_intervals, (interval_min / num_horizons), 1.0)
        depth_intervals = depth_intervals / depth_intervals.sum()

        # Make horizon matrices: starting from a zero-plane, move to the next `depth_interval` and apply randomizations
        horizon_matrices = [np.zeros(spatial_shape, dtype=np.float32)]

        for depth_interval in depth_intervals:
            previous_matrix = horizon_matrices[-1]
            next_matrix = previous_matrix + depth_interval

            if randomization1_scale > 0:
                perturbation_matrix = self.make_randomization1_matrix(shape=spatial_shape, num_nodes=num_nodes,
                                                                      interpolation_kind=interpolation_kind,
                                                                      rng=self.rng)
                next_matrix += randomization1_scale * depth_interval * perturbation_matrix

            if randomization2_scale:
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
        self.padding = padding
        self.shape_padded = shape_padded
        self.depth_padded = depth
        self.depth_intervals = depth_intervals
        self.horizon_matrices = horizon_matrices
        return self

    def make_velocity_model(self):
        """ !!. """
        buffer = np.empty(self.shape_padded, dtype=np.float32)
        velocity_model = compute_velocity_model(buffer, self.velocity_vector, self.horizon_matrices)
        self.velocity_model = velocity_model
        return self


    # Modifications of velocity model
    def make_fault_2d(self, coordinates, max_shift=20, width=3, fault_id=0,
                      shift_vector=None, shift_sign=+1, mode='sin2', num_zeros=0.1,
                      perturb_peak=True, perturb_values=True):
        """ !!. """
        # Parse parameters
        if coordinates == 'random':
            coordinates = self.make_fault_coordinates_2d()

        point_1, point_2 = coordinates
        point_1 = point_1 if len(point_1) == 3 else (0, *point_1)
        point_2 = point_2 if len(point_2) == 3 else (0, *point_2)

        point_1 = tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                        for c, s in zip(point_1, self.shape_padded))
        point_2 = tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                        for c, s in zip(point_2, self.shape_padded))

        (i1, x1, d1), (i2, x2, d2) = point_1, point_2
        if i1 != i2:
            raise ValueError(f'Points should be on the same iline! {point_1}, {point_2}')

        # Define crop of the slide that we work with: ranges along both axis
        x_low, x_high = 0, self.shape_padded[1]
        d_low, d_high = min(d1, d2), max(d1, d2)
        d_len = d_high - d_low

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
                         np.arange(0, self.depth_padded, dtype=np.float32).reshape(1, 1, -1))

        map_coordinates(input=indices_array[i1, x_low:x_high, d_low:d_high],
                        coordinates=(xmesh + kx * shifts, dmesh + kd * shifts),
                        output=indices_array[i1, x_low:x_high, d_low:d_high],
                        mode='nearest')

        for i, horizon_matrix in enumerate(self.horizon_matrices):
            # Find the position of original `depth` in the transformed indices
            idx_x = np.nonzero(horizon_matrix[i1])[0]
            depths = horizon_matrix[i1][idx_x]

            mask = (0 <= depths) & (depths < self.depth_padded)
            idx_x, depths = idx_x[mask], depths[mask]

            self.horizon_matrices[i][i1][idx_x] = find_depths(indices_array[i1], idx_x, depths)

        # Transform fault point clouds on the same slide
        shifts_array = np.zeros(self.shape_padded[1:], dtype=np.float32)
        shifts_array[x_low:x_high, d_low:d_high] = shifts

        updated_point_clouds = []
        for fault_id_, (idx_x, depths) in self.fault_point_clouds[i1]:
            shifts_ = shifts_array[idx_x, depths]
            idx_x, depths = idx_x + kx * shifts_, depths + kd * shifts_
            idx_x, depths = np.round(idx_x).astype(np.int32), np.round(depths).astype(np.int32)

            updated_point_clouds.append((fault_id_, (idx_x, depths)))
        self.fault_point_clouds[i1] = updated_point_clouds

        # Store the added fault point cloud
        idx_x, depths = np.nonzero((0 <= distance) & (distance / width < 1))
        idx_x, depths = idx_x.astype(np.int32), depths.astype(np.int32)
        depths += d_low
        mask = (d_low <= depths) & (depths < d_high)
        idx_x, depths = idx_x[mask], depths[mask]
        self.fault_point_clouds[i1].append((fault_id, (idx_x, depths)))

        self.fault_coordinates.append((point_1, point_2))
        return self


    def make_fault_3d(self, upper_points, lower_points, max_shift=20, width=3,
                      shift_sign=+1, mode='sin2', num_zeros=0.1, perturb_peak=True, perturb_values=True,):
        """ !!. """
        # Prepare points
        upper_points = tuple(tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                                   for c, s in zip(point, self.shape_padded))
                             for point in upper_points)
        lower_points = tuple(tuple(int(c * (s - 1)) if isinstance(c, (float, np.floating)) else c
                                   for c, s in zip(point, self.shape_padded))
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
        shift_vector = self.make_shift_vector(self.depth_padded, mode=mode, num_zeros=num_zeros,
                                              perturb_peak=perturb_peak, perturb_values=perturb_values, rng=self.rng)

        for upper_point, lower_point in zip(upper_path, lower_path):
            self.make_fault_2d((upper_point, lower_point), max_shift=max_shift, width=width, fault_id=self.fault_id,
                               shift_vector=shift_vector, shift_sign=shift_sign)

        self.fault_paths[self.fault_id] = (upper_path, lower_path)
        self.fault_id += 1
        return self


    def make_fault_coordinates_2d(self, margin=0.1, d1_range=(0.1, 0.25), d2_range=(0.75, 0.9)):
        """ !!. """
        x1 = self.rng.uniform(low=0+margin, high=1-margin)
        x2 = x1 + self.rng.uniform(low=-margin, high=+margin)

        d1 = self.rng.uniform(*d1_range)
        d2 = self.rng.uniform(*d2_range)
        return ((0, x1, d1), (0, x2, d2))

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
            perturbation = (scale * (b - a)) * self.rng.random(size=self.shape_padded, dtype=np.float32) + (scale * a)
        elif randomization == 'normal':
            perturbation = scale * randomization_scale * self.rng.standard_normal(size=self.shape_padded,
                                                                                  dtype=np.float32)
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
        for i in range(self.shape_padded[0]):
            cv2.filter2D(src=self.reflectivity_model[i], ddepth=-1, kernel=wavelet,
                         dst=synthetic[i], borderType=cv2.BORDER_CONSTANT)

        self.synthetic = synthetic
        return self

    def postprocess_synthetic(self, sigma=1., kernel_size=9, noise_mul=None):
        """ !!. """
        if sigma is not None:
            self.synthetic = self.apply_gaussian_filter_3d(self.synthetic, kernel_size=kernel_size, sigma=sigma)
        if noise_mul is not None:
            left, right = np.quantile(self.synthetic, (0.05, 0.95))
            mask = (left < self.synthetic) & (self.synthetic < right)

            perturbation = 2 * self.rng.random(size=mask.sum(), dtype=np.float32) - 1
            self.synthetic[mask] += noise_mul * perturbation

        return self

    def postprocess_synthetic_2d(self, ):
        """ TODO: Perlin noise. """


    def finalize_array(self, array, loc=None, axis=0, angle=None):
        """ !!. """
        if not self.finalized:
            slc = [slice(-s, None) for s in self.shape]
            array = array[slc]

        #TODO: rotation
        _ = angle

        if loc is not None:
            array = np.take(array, indices=loc, axis=axis)
        return array

    def finalize_attributes(self):
        """ !!. """
        slc = [slice(-s, None) for s in self.shape]
        padding_i, padding_x, padding_d = self.padding

        # Slice all of the arrays
        for attribute in ['velocity_model', 'reflectivity_model', 'synthetic']:
            view = getattr(self, attribute)[slc]
            setattr(self, attribute, view)

        # Shift horizon matrices
        self.horizon_matrices = self.horizon_matrices[:, slc[0], slc[1]]
        self.horizon_matrices -= padding_d

        # Shift fault point clouds
        updated_point_clouds = defaultdict(list)
        for i, point_cloud_list in self.fault_point_clouds.items():
            new_i = i - padding_i

            if new_i >= 0:
                for fault_id, (idx_x, depths) in point_cloud_list:
                    idx_x = idx_x - padding_x
                    depths = depths - padding_d

                    mask = (idx_x >=0) & (depths >=0)
                    idx_x, depths = idx_x[mask], depths[mask]

                    updated_point_clouds[new_i].append((fault_id, (idx_x, depths)))
        self.fault_point_clouds = updated_point_clouds

        self.finalized = True
        self.padding = (0, 0, 0)
        self.shape_padded = self.shape
        self.depth_padded = self.depth


    # Attribute getters
    def get_attribute(self, attribute='synthetic', loc=None, axis=0):
        """ !!. """
        result = getattr(self, attribute)
        result = self.finalize_array(result, loc=loc, axis=axis)
        return result

    def get_horizons(self, indices='all', format='mask', width=3, loc=None, axis=0):
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
            indices = np.nonzero((0 <= horizon_matrices) & (horizon_matrices < self.depth_padded))
            result = np.zeros(self.shape_padded, dtype=np.float32)
            result[(*indices[1:], horizon_matrices[indices])] = 1

            if width is not None:
                kernel = np.ones(width, dtype=np.float32).reshape(1, width)

                for i in range(self.shape_padded[0]):
                    cv2.filter2D(src=result[i], ddepth=-1, kernel=kernel,
                                 dst=result[i], borderType=cv2.BORDER_CONSTANT)
            result = np.clip(result, 0, 1)
            result = self.finalize_array(result, loc=loc, axis=axis)
        else:
            raise ValueError(f'Unsupported `format={format}`!')
        return result

    def get_faults(self, format='mask', width=5, loc=None, axis=0):
        """ !!. """
        if 'cloud' in format:
            ...
        elif 'instance' in format:
            ...
        elif 'mask' in format:
            result = np.zeros(self.shape_padded, dtype=np.float32)

            for i, point_cloud_list in self.fault_point_clouds.items():
                for _, (idx_x, depths) in point_cloud_list:
                    result[i][idx_x, depths] = 1

            if width is not None:
                kernel = np.ones((width, width), dtype=np.float32)#.reshape(width, 1)

                for i in range(self.shape_padded[0]):
                    cv2.filter2D(src=result[i], ddepth=-1, kernel=kernel,
                                 dst=result[i], borderType=cv2.BORDER_CONSTANT)
            result = np.clip(result, 0, 1)
            result = self.finalize_array(result, loc=loc, axis=axis)
        else:
            raise ValueError(f'Unsupported `format={format}`!')
        return result

    def get_increasing_impedance_model(self):
        """ !!. """
        return ...


    # Visualization
    def show_slide(self, loc=None, axis=0, velocity_cmap='jet', return_figure=False,
                   remove_padding=True, horizon_width=5, fault_width=7, **kwargs):
        """ !!. """
        #TODO: add the same functionality, as in `SeismicCropBatch.plot_roll`
        #TODO: use the same v_min/v_max for all locations
        loc = loc or self.shape[axis] // 2

        # Retrieve the data
        velocity = self.get_attribute(attribute='velocity_model', loc=loc, axis=axis)
        reflectivity = self.get_attribute(attribute='reflectivity_model', loc=loc, axis=axis)
        synthetic = self.get_attribute(attribute='synthetic', loc=loc, axis=axis)
        horizon_mask = self.get_horizons(width=horizon_width, loc=loc, axis=axis)
        fault_mask = self.get_faults(width=fault_width, loc=loc, axis=axis)

        # Arrange data
        data = [velocity, [velocity, horizon_mask, fault_mask],
                synthetic, [synthetic, horizon_mask, fault_mask],
                reflectivity, horizon_mask, fault_mask]
        titles = ['`velocity_model`', '`velocity model, overlayed`',
                  '`synthetic`', '`synthetic, overlayed`',
                  '`reflectivity model`', '`horizon mask`', '`fault mask`']
        cmaps = [[velocity_cmap], [velocity_cmap, 'red', 'purple'],
                 ['gray'], ['gray', 'red', 'purple'],
                 ['gray'], ['gray'], ['gray']]

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
        msg += f'\nmin_interval = {self.depth_intervals.min() * self.depth_padded:4.0f}'
        msg += f'\nmax_interval = {self.depth_intervals.max() * self.depth_padded:4.0f}'
        msg += f'\nmean_interval = {self.depth_intervals.mean() * self.depth_padded:4.0f}'
        legend_params = {
            'color': 'pink',
            'label': msg,
            'size': 14, 'loc': 10,
            'facecolor': 'pink',
        }
        MatplotlibPlotter.add_legend(ax=fig.axes[len(data)], **legend_params)

        if return_figure:
            return fig
        return None


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
        """ !!. """
        # Parse parameters
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        sample_size = sample_size or min(10000, 5*np.prod(shape))

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
        kernel_1d = SyntheticGenerator.make_gaussian_kernel_1d(kernel_size=blur_size, sigma=blur_sigma)
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
        kernel_1d = SyntheticGenerator.make_gaussian_kernel_1d(kernel_size=kernel_size, sigma=sigma)

        for i in range(array.shape[0]):
            cv2.sepFilter2D(src=array[i], ddepth=-1, kernelX=kernel_1d.reshape(1, -1), kernelY=kernel_1d.reshape(-1, 1),
                            dst=array[i], borderType=cv2.BORDER_CONSTANT)

        if array.shape[0] >= 3 * sigma * kernel_size:
            for j in range(array.shape[1]):
                cv2.filter2D(src=array[:, j], ddepth=-1, kernel=kernel_1d.reshape(-1, 1),
                             dst=array[:, j], borderType=cv2.BORDER_CONSTANT)
        return array




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
    #pylint: disable=consider-using-enumerate
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
