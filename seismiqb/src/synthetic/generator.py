""" Generation of synthetic seismic. """
#pylint: disable=not-an-iterable, too-many-arguments, too-many-statements, redefined-builtin
import os
from collections import defaultdict

import numpy as np
from numba import njit, prange

import cv2
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import map_coordinates
from scipy.signal import ricker, resample

from ..field import Field
from ..labels import Horizon, Fault
from ..geometry import array_to_sgy
from ..plotters import MatplotlibPlotter, plot_image



class SyntheticGenerator:
    """ A class for synthetic generation.

    The process is split into a number of methods, which are supposed to be chained.
    Each of the methods has a lot of parameters, which control the exact randomization that is used for generation.
    Here, we overview the overall process, explain key concepts and outline potential improvements.

    1. `make_velocity_vector` is used to create an increasing (almost everywhere) vector of velocities in desired range.
    This vector defines the textural patterns of the resulting seismic: colors and color differences of layers.
        - Randomization allows to create a non-linear sequence of velocities, as well as to add some sharper peaks
        (larger differences) to the vector: those would correspond to a better seen (amplified) horizons.
    This method defines the number of horizons in the resulting synthetic image.
    The result is `velocity_vector` attribute of (num_horizons,) shape.

    2. `make_horizons` creates a sequence of almost conforming surfaces.
    They define structural patterns of the resulting seismic: where the layers are and how they interact.
    In the simplest case, horizons are just uniformly spaced surfaces.
        - The first randomization is to change spacing between surfaces: this makes layers thicker / thinner.
        - The second randomization is to perturb the surfaces itself: the key here is to repeat this perturbation on all
        of subsequent surfaces. To this end, each next horizon matrix starts as a shifted version of the previous one.
        We support multiple types of such perturbations:
            - One uses a small number of spatial nodes to create uniformly distributed shifts, which are then
            interpolated to match the requested spatial shape: this results in smooth vertical changes.
            Number of nodes controls frequency of peaks; yet, the horizon lines would still be featureless.
            - The other uses a mixture of Gaussians with randomized locations and scales to add 'clusters' of shifts.
            This results in much more defined peaks, and also introduces some jiggle into the horizons.
            Parameters allow to control number of added peaks, their spatial/depth size, and the overall direction.
            Note that this kind of noise is highly desirable, but slow to compute.
            - TODO: Perlin noise.
            The hope is that it would produce similar results to the mixture of Gaussians, but much faster.
    This method defines the resulting shape of the produced synthetic image.
    The result is `horizon_matrices` attribute of (num_horizons, *spatial_shape) shape.

    3. `make_velocity_model` stretches the `velocity_vector` along `horizon_matrices` depth-wise.
        - TODO: add spatial randomization.
        Model small amplitude changes along horizons, as well as modify inter-horizon space.
    The result is `velocity_model` attribute of (*spatial_shape, depth) shape.

    4. `make_fault_2d` and `make_fault_3d` add elastic discontinueties on the velocity model.
    Modifies `horizon_matrices`, so all of the attributes are synchronized and correctly transformed.
    Can be used multiple times to add faults on the same image: each next fault will affect all of the previous ones.
    In 2D case, parametrized by a segment coordinates. In 3D case, fault is defined by upper and lower polylines,
    which are used to create segment coordinates for each 2D slide.

    5. `make_density_model`, `make_impedance_model`, `make_reflectivity_model` produce (*spatial_shape, depth) arrays.
        - `density_model` is a slightly perturbed `velocity_model`.
        - `impedance_model` is an element-wise product of `velocity_model` and `density_model`.
        - `reflectivity_model` is a ratio between difference and sum of `impedance_model` in subsequent layers.
        To condition this fraction, we add the doubled ~mean impedance to the denominator.

    6. `make_synthetic` convolves the `reflectivity_model` with Ricker wavelet.

    7. `postprocess_synthetic` and `postprocess_synthetic_2d` apply noise and blurring to make synthetic more realistic.

    That concludes the usual pipeline of synthetic generation. Additional features and methods:
        - `get_*` methods extract the actual data from the generator.
        Whether you need the synthetic image, horizon mask or something else, use those methods:
        never access instance attributes, as some array operations are applied only when the data is actually requested.

        - Shape padding. `make_horizons` allows to pad the `shape` to a bigger size, which makes all of the methods
        create slightly bigger arrays. The reason for padding is to avoid various border effects.
        At data access via `get_*` methods, the arrays are sliced to match the originally requested `shape`: that is
        one of the reasons to use `get_*` methods exclusively for data retrieval.

        - `finalize` method can be used to delete some of the unnecessary attributes (mostly, `*_model` arrays) and
        free up some memory. A usual generator uses 5 (*spatial_shape, depth)-shaped buffers:
            - `velocity_model`,
            - `density_model`,
            - `impedance_model`,
            - `reflectivity_model`,
            - `synthetic`,
        of which you may need only one or two. Reducing the memory footprint is always a good idea!

        - `show_slide` and `show_stats` can be used to visualize produced results and assess distributions.
    """
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

        for key, value in kwargs.items():
            setattr(self, key, value)

    # Make velocity model: base for seismic generation
    def make_velocity_vector(self, num_horizons=10, limits=(5_000, 10_000),
                             randomization='uniform', randomization_scale=0.3,
                             amplify_probability=0.2, amplify_range=(2.0, 4.0), amplify_sign_probability=0.8):
        """ Create an almost increasing vector of velocities, which defines textural patterns of resulting seismic.

        In the simplest case, is just a `num_horizons`-sized linspace of `limits`.
        `randomization_*` parameters control perturbations of this linspace and randomizes layer transitions.
        `amplify_*` parameters allow to make some of the horizons more visible. The first horizon can't be amplified.

        Parameters
        ----------
        num_horizons : int
            Number of desired horizons.
        limits : tuple of two int
            Minimum and maximum velocity. Not matched exactly and can be exceeded in both directions.
            # TODO: maybe, add scale back option to preserve `limits`?

        randomization : {None, 'normal', 'uniform'}
            Type of randomization to apply. Generates a zero-centered noise, which perturbs the original linspace.
        randomization_scale : number
            Scale of applied randomization with relation to (limits_diff / num_horizons).
            A scale of 1.0 means that perturbation can result in the same difference, as the original linspace.

        amplify_probability : number
            Probability of a horizon to be amplified.
        amplify_range : tuple of two numbers
            Range to uniformly sample the size of horizon amplification with relation to (limits_diff / num_horizons).
            The bigger it is, the sharper the contrast between layers.
        amplify_sign_probability : number
            Chance to invert the size of amplification. Due to this, velocity vector may not be always increasing.
            TODO: maybe, should be applied to all velocities?
        """
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

        # Resulting velocity vector may be outside of upper range. Split this difference across both ranges
        velocity_vector -= (velocity_vector.max() - limits[1]) / 2

        # Store in the instance
        self.num_horizons = num_horizons
        self.velocity_vector = velocity_vector
        self.amplified_horizon_indices = amplified_horizon_indices
        return self

    def make_horizons(self, shape, padding=(0, 16, 32), num_horizons=None,
                      interval_randomization=None, interval_randomization_scale=0.1, interval_min=0.5,
                      randomization1_scale=0.25, num_nodes=10, interpolation_kind='cubic',
                      randomization2_scale=0.1, locs_n_range=(2, 10), locs_scale_range=(5, 15), sample_size=None,
                      blur_size=9, blur_sigma=2.0, digitize=True, n_bins=20, output_range=(-0.2, 0.8)):
        """ Create a sequence of almost conforming horizons, which define structural features of the produced seismic.

        In the simplest case, a creates uniformly spaced surfaces.
        `interval_*` parameters changes distances between surfaces.
        `randomization1_*` changes surfaces in a smooth node-based way.
        `randomization2_*` uses mixture of Gaussians to add defined peaks to horizon surfaces.

        Optinally pads the `shape` so that all of the subsequent computations are performed on a slightly bigger shape.
        Used to avoid border effects and correctly processed by `get_*` methods.

        Parameters
        ----------
        shape : tuple of ints
            Desired shape of created seismic images.
        padding : tuple of ints
            Padding along each of the axis. If used, then all of the computations in this method and others would be
            performed with slightly bigger shape, and the final slicing is used in `get_*` methods.
            Note that axes of size 1 (in case of 2D seismic) are not padded.
        num_horizons : int, optional
            If provided, then the number of horizons to create. Default is the number of horizons in `velocity_vector`.

        interval_randomization : {None, 'uniform', 'normal'}
            Type of randomization to apply to intervals between horizons.
            Generates a zero-centered noise, which perturbs the original equally-spaced intervals.
        interval_randomization_scale : number
            Scale of applied interval randomization with relation to (depth / num_horizons).
            A scale of 1.0 means that perturbation can result in horizons running one to other.
        interval_min : number
            Smallest allowed interval after perturbation with relation to (depth / num_horizons).

        randomization1_scale : number
            Scale of surface perturbation of the first type with relation to its interval to the previous layer.
        num_nodes : int
            Number of nodes for creating grid. The bigger, the more oscillations.
        interpolation_kind : str or int, optional
            Kind of interpolation to use. Refer to `scipy.interpolate.interp2d` for further explanation.

        randomization2_scale : number
            Scale of surface perturbation of the second type with relation to its interval to the previous layer.
        locs_n_range : tuple of two ints
            Range for generating the number of Gaussians in the mixture.
        locs_scale_range : tuple of two numbers
            Range for generating the scale of Gaussians in the mixture.
        sample_size : number, optional
            If provided, then the number of sampled points. Default is the spatial shape size.
        blur_size, blur_sigma : number, number
            Parameters of Gaussian blur for post-processing of perturbation matrix.
        digitize, n_bins : bool, int
            Whether to binarize the perturbation matrix. Used to add sharpness and jiggle to horizons.
        output_range : tuple of two numbers
            Range to scale the perturbation, which defines the overall direction.
            For example, (0.0, 1.0) value would mean that perturbation shifts the surface only in the down direction.
        """
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
        depth_intervals = np.ones(num_horizons - 1, dtype=np.float32) / num_horizons

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
                                                                      digitize=digitize, n_bins=n_bins,
                                                                      output_range=output_range,
                                                                      rng=self.rng)
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
        """ Compute the velocity model by depth-wise stretching `velocity_vector` along `horizon_matrices`. """
        buffer = np.empty(self.shape_padded, dtype=np.float32)
        velocity_model = compute_velocity_model(buffer, self.velocity_vector, self.horizon_matrices)
        self.velocity_model = velocity_model
        return self

    # Modifications of velocity model
    def make_fault_2d(self, coordinates, max_shift=20, width=3, fault_id=0,
                      shift_vector=None, shift_sign=+1, mode='sin2', num_zeros=0.1,
                      perturb_peak=True, perturb_values=True, update_horizon_matrices=True):
        """ Add an elastic distortion to simulate fault on velocity model.
        Also transforms `horizon_matrices`, so that all of the attributes are synchronized.

        For a given segment, defined by `coordinates`, we do the following:
            - create a `shift_vector`, which defines the amplitude of produced fault and its patterns.
            - based on a segment, create a direction vector
            - compute distances to the segment for each point: it is used to scale the amount of distortion.
            The closer the point to the segment itself, the sharper its shift would be.
            - for each pixel in the velocity model, shift its coordinate to a new one, based on
            `shift_vector`, `direction_vector` and `distance`.
            - re-map the velocity model to new coordinates.

        Parameters
        ----------
        coordinates : str or tuple of tuples of two numbers
            If str 'random', then coordinates will be generated automatically by :meth:`.make_fault_coordinates_2d`.
            If tuple of tuple of two ints, then pixel coordinates of the segment: ((x1, d1), (x2, d2)).
            If tuple of tuple of two floats, then unit coordinates of the segment, which are scaled back to
            the integer pixel coordinates by multiplying provided numbers by `shape`.
        max_shift : number
            Maximum distance to shift pixels along segment.
        width : number
            Number of pixels for smooth transition from no shift to max shift. Can be used to model flexures.
        fault_id : int
            Technical parameter, should not be used.
        shift_vector : np.ndarray, optional
            If provided, used as `shift_vector`. May be resampled to match the length of fault segment.
            If provided, `mode`, `num_zero`, `pertub_peak` and `perturb_values` are not used.
        shift_sign : {-1, +1}
            +1 means that the shifted part would be moved down, -1 means moving up.
        mode : {'sin2'}
            Mode for making shift vector. Changes the pattern of shifts along chosen direction.
        num_zeros : number
            Proportion of zeros on the sides of created `shift_vector`.
            0.2 means that rougly 10% from each side are going to be filled with zeros.
        perturb_peak : bool
            Whether to randomly shift the peak of `shift_vector` to the sides.
        perturb_values : bool
            Whether to add small randomization to values in the `shift_vector`.
        update_horizon_matrices : bool
            Whether to update existing horizon matrices by produced elastic distortions.
            Can be set to False to speed up computations in case horizons are not needed.
        """
        # Parse parameters
        if coordinates == 'random':
            coordinates = self.make_fault_coordinates_2d(coordinates)

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
        """ Create a fault in 3D.
        Works by making coordinates along polylines of upper and lower points, and making 2D faults for each slide.
        Uses the same `shift_vector` on each slide, so that fault surface is continuous.

        Parameters
        ----------
        upper_points : tuple of tuples of three numbers
            Coordinates in space to connect by a continuous path. Used as the first coordinate in the 2D fault making.
        lower_points : tuple of tuples of three numbers
            Coordinates in space to connect by a continuous path. Used as the second coordinate in the 2D fault making.
        shift_sign, mode, num_zeros, perturb_peak, perturb_values : dict
            Used to create one common `shift_vector` for all slides.
        max_shift, width : dict
            Same as in :meth:`.make_fault_2d`.
        """
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

    def make_fault_coordinates_2d(self, mode='random', margin=0.1, d1_range=(0.1, 0.25), d2_range=(0.75, 0.9)):
        """ Sample a pair of points for 2D fault segment, avoiding the image edges.
        `mode` is not used currently, but passed from `make_fault_2d` and can be used for new randomizations.
        """
        _ = mode
        x1 = self.rng.uniform(low=0+margin, high=1-margin)
        x2 = x1 + self.rng.uniform(low=-margin, high=+margin)

        d1 = self.rng.uniform(*d1_range)
        d2 = self.rng.uniform(*d2_range)
        return ((0, x1, d1), (0, x2, d2))

    @staticmethod
    def make_path(point_1, point_2):
        """ Rasterize path between `point_1` and `point_2` in 3D space. """
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
        """ Make density model as velocity model with minor multiplicative perturbations.

        Parameters
        ----------
        scale : number
            Multiplier for output density model. Used to decay values.
        randomization : {None, 'uniform', 'normal'}
            Type of perturbations to apply.
        randomization_limits : tuple of two numbers
            Limits for generated perturbation. Used only in 'uniform' mode.
        randomization_scale : number
            Scale of generated perturbation. Used only in 'normal' mode.
        """
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
        """ Make impedance as the product of density and velocity models. """
        self.impedance_model = self.velocity_model * self.density_model
        return self

    def make_reflectivity_model(self):
        """ Compute reflectivity as the ratio between differences and sums for successive layers of impedance.
        To condition this fraction, we add the doubled ~mean impedance to the denominator.

        reflectivity = ((impedance[..., 1:] - impedance[..., :-1]) /
                        (impedance[..., 1:] + impedance[..., :-1]))
        """
        buffer = np.empty_like(self.impedance_model)
        reflectivity_model = compute_reflectivity_model(buffer, self.impedance_model)

        self.reflectivity_model = reflectivity_model
        return self

    def make_synthetic(self, ricker_width=5, ricker_points=50):
        """ Produce seismic image by convolving reflectivity with Ricker wavelet. """
        wavelet = ricker(ricker_points, ricker_width)
        wavelet = wavelet.astype(np.float32).reshape(1, ricker_points)
        wavelet *= 100

        synthetic = np.empty_like(self.reflectivity_model)
        for i in range(self.shape_padded[0]):
            cv2.filter2D(src=self.reflectivity_model[i], ddepth=-1, kernel=wavelet,
                         dst=synthetic[i], borderType=cv2.BORDER_CONSTANT)

        self.synthetic = synthetic
        return self

    def postprocess_synthetic(self, sigma=1., kernel_size=9, clip=True, noise_mode=None, noise_mul=0.2):
        """ Apply blur, clip and noise to the generated synthetic.
        Clipping is done by shifting values outside of (0.01, 0.99) quantiles closer to the distribution.

        Parameters
        ----------
        sigma : number
            Scale for gaussian blurring.
        kernel_size : int
            Size of the kernel for gaussian blurring.
        clip : bool
            Whether to clip amplitude values to (0.01, 0.99) quantiles.
        noise_mode : {None, 'uniform', 'normal'}
            Type of noise to add to seismic image.
        noise_mul : number
            SNR of added perturbation.
        """
        if sigma is not None:
            self.synthetic = self.apply_gaussian_filter_3d(self.synthetic, kernel_size=kernel_size, sigma=sigma)

        if clip:
            left, right = np.quantile(self.synthetic, [0.01, 0.99])
            self.synthetic = np.clip(self.synthetic, left, right)

        if noise_mode is None:
            perturbation = 0
        elif noise_mode == 'normal':
            perturbation = self.rng.standard_normal(size=self.shape_padded, dtype=np.float32)
        elif noise_mode == 'uniform':
            perturbation = 2 * self.rng.random(size=self.shape_padded, dtype=np.float32) - 1
        self.synthetic += (noise_mul * self.synthetic.std()) * perturbation

        return self

    def postprocess_synthetic_2d(self, ):
        """ TODO: Perlin noise. """


    # Finalization
    def finalize_array(self, array, loc=None, axis=0, angle=None):
        """ Slice the array to match requested `shape` (without padding), index along desired axis, if needed. """
        if not self.finalized:
            slc = [slice(-s, None) for s in self.shape]
            array = array[slc]

        #TODO: rotation
        _ = angle

        if loc is not None:
            array = np.take(array, indices=loc, axis=axis)
        return array

    def finalize(self):
        """ Slice all `*_model` attributes, as well as the `horizon_matrices` and `fault_point_clouds`
        to correctly account for effects of padding.
        """
        if self.finalized:
            return None

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
        self.depth_intervals *= (self.depth_padded / self.depth)
        self.shape_padded = self.shape
        self.depth_padded = self.depth
        return None

    def cleanup(self, delete=('density_model', 'impedance_model')):
        """ Delete some of the attributes. Useful in generation pipelines. """
        for attribute in delete:
            delattr(self, attribute)


    # Attribute getters
    def get_attribute(self, attribute='synthetic', loc=None, axis=0):
        """ Get a value of desired attribute while correctly accounting for padding of shapes. """
        result = getattr(self, attribute)
        result = self.finalize_array(result, loc=loc, axis=axis)
        return result

    def get_field(self):
        """ !!. """
        synthetic = self.finalize_array(self.synthetic)
        field = Field('array.dummyarray', geometry_kwargs={'array': synthetic})
        return field

    def get_horizons(self, indices='all', format='mask', width=3, loc=None, axis=0):
        """ Extract horizons as a mask, list of separate horizon matrices or instances.
        Correctly accounts for effects of padding.
        """
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
            self.finalize()
            result = horizon_matrices
        elif 'instance' in format:
            self.finalize()
            field = self.get_field()
            result = [Horizon(matrix, field=field, name=f'horizon_{i}')
                      for i, matrix in enumerate(horizon_matrices)]
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
        """ Extract faults as a mask, list of separate point clouds or instances.
        Correctly accounts for effects of padding.
        """
        if 'cloud' in format:
            # Collect point clouds with the same `fault_id`
            result = defaultdict(list)
            for i, pc_list in self.fault_point_clouds.items():
                for fault_id, (idx_x, depths) in pc_list:
                    point_cloud = np.stack([np.full_like(idx_x, fill_value=i), idx_x, depths]).T
                    result[fault_id].append(point_cloud)

            # Vstack points cloud for each `fault_id`
            for fault_id, pc_list in result.items():
                result[fault_id] = np.vstack(pc_list)
            result = list(result.values())
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
        """ TODO. """


    # Visualization
    def show_slide(self, loc=None, axis=0, velocity_cmap='jet', return_figure=False,
                   remove_padding=True, horizon_width=5, fault_width=7, **kwargs):
        """ Show a figure with pre-defined graphs:
            - velocity
            - velocity, overlayed with horizons and  faults
            - seismic image
            - seismic image, overlayed with horizons and faults
            - reflectivity
            - horizon mask
            - fault mask
        """
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

    def show_stats(self, return_figure=False, **kwargs):
        """ Show a figure with pre-defined graphs:
            - histogram of amplitudes
            - mean of amplitudes along depth
            - velocity vector
            - layer sizes
        """
        kwargs = {
            'nrows': 2,
            'shapes': 3,
            'return_figure': True,
            **kwargs
        }
        fig = plot_image(self.get_attribute(), mode='hist',
                         title='amplitude histogram', **kwargs)

        plot_image(self.synthetic.mean(axis=(0, 1)), mode='curve', ax=fig.axes[1],
                   title='amplitude / depth', xlabel='depth', ylabel='mean amplitude')

        plot_image(self.velocity_vector, mode='curve',
                   title='velocity vector', xlabel='horizon index', ylabel='velocity',
                   ax=fig.axes[2])

        plot_image(self.depth_intervals, mode='curve', ax=fig.axes[3],
                   title='layer size', xlabel='horizon index', ylabel='size')

        if return_figure:
            return fig
        return None


    # Storing to disk
    def dump(self, path, save_qblosc=True, save_carcasses=True, carcass_frequency=30, pbar='t'):
        """ Dump generated contents to a disk in a field format. Returns a re-loaded field with that data.
        Creates directory with a following structure:

        NAME/
        ├── NAME.sgy
        ├── NAME.meta
        ├── NAME.qblosc (optional)
        └──  INPUTS/
            ├── FAULTS/
            │   └── fault_{i}.npz
            └── HORIZONS/
                └── FULL/
                │   └── horizon_{i}.char
                └── CARCASS/ (optional)
                    └── carcass_of_horizon_{i}.char

        Name is the basename of `path`.

        Parameters
        ----------
        path : str
            Path to save the contents. Basename of the path is used as field name.
        save_qblosc : bool
            Whether to convert the SEG-Y to a QBLOSC.
        save_carcasses : bool
            Whether to save carcasses of horizons.
        carcass_frequency : int
            Frequency of carcass making.
        pbar : bool or str
            Identifier of progress bar to use. 't' is text-based progress bar. False disables progress bar.
        """
        # Prepare paths
        name = os.path.basename(path.strip('/'))

        for class_dir in ['FAULTS', 'HORIZONS']:
            new_dir = os.path.join(path, 'INPUTS', class_dir)
            os.makedirs(new_dir, exist_ok=True)

        cube_path_sgy = os.path.join(path, f'{name}.sgy')
        horizon_path = os.path.join(path, 'INPUTS/HORIZONS/FULL/$.char')
        carcass_path = os.path.join(path, 'INPUTS/HORIZONS/CARCASS/$.char')
        fault_path = os.path.join(path, 'INPUTS/FAULTS/$.npz')

        # Geometries: SEG-Y and QBLOSC
        self.finalize()
        array_to_sgy(self.synthetic, cube_path_sgy, zip_segy=False, pbar=pbar)
        field = Field(cube_path_sgy, geometry_kwargs={'pbar': pbar})
        if save_qblosc:
            geometry_converted = field.geometry.convert(format='qblosc', pbar=pbar)
            field = Field(geometry_converted)

        # Horizons: extract from matrices with correct names, save
        horizons = []
        for i, matrix in enumerate(self.horizon_matrices):
            name = f'horizon_{i}'
            if i in self.amplified_horizon_indices:
                name = 'amplified_' + name
            horizon = Horizon(matrix, field=field, name=name)
            horizons.append(horizon)
        field.load_labels({'horizons': horizons}, labels_class='horizon')
        field.horizons.dump(horizon_path)

        # Carcasses: make and save
        if save_carcasses:
            carcasses = field.horizons.make_carcass(frequencies=carcass_frequency, margin=5)
            field.load_labels({'carcasses': carcasses}, labels_class='horizon')
            carcasses.dump(carcass_path)

        # Faults
        faults = self.get_faults(format='point_clouds')
        faults = [Fault(point_cloud, field=field, name=f'fault_{i}')
                  for i, point_cloud in enumerate(faults)]
        field.load_labels({'faults': faults}, labels_class=Fault, pbar=False)
        field.faults.dump_points(fault_path)

        return field


    # Utilities and faster versions of common operations
    @staticmethod
    def make_gaussian_kernel_1d(kernel_size, sigma):
        """ Create a 1d gaussian kernel. """
        kernel_1d = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size, dtype=np.float32)
        kernel_1d = np.exp(-0.5 * np.square(kernel_1d) / np.square(sigma))
        return kernel_1d / kernel_1d.sum()

    @staticmethod
    def make_randomization1_matrix(shape, num_nodes=10, interpolation_kind='cubic', rng=None):
        """ Create 1d/2d perturbation matrix. Under the hood, uses sparse grid of nodes with uniform perturbations,
        which is then interpolated into desired `shape`.

        Parameters
        ----------
        num_nodes : int or tuple of int
            Number of nodes along each axis. If int, the same value is used for both axis.
            The more nodes, the more oscillations would be produced.
        interpolation_kind : str or int, optional
            Kind of interpolation to use. Refer to `scipy.interpolate.interp2d` for further explanation.
        rng : np.random.Generator or int, optional
            Random Number Generator or seed to create one.
        """
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
                                   blur_size=9, blur_sigma=2., digitize=True, n_bins=20, output_range=(0.0, 1.0),
                                   rng=None):
        """ Create 1d/2d perturbation matrix. Under the hood, we use a mixture of Gaussians with uniformly distributed
        locations and scales (in both directions) to sample shifts for each pixel.

        Parameters
        ----------
        locs_n_range : tuple of two ints
            Range for generating the number of Gaussians in the mixture.
        locs_scale_range : tuple of two numbers
            Range for generating the scale of Gaussians in the mixture.
        sample_size : number, optional
            If provided, then the number of sampled points. Default is the spatial shape size.
        blur_size, blur_sigma : number, number
            Parameters of Gaussian blur for post-processing of perturbation matrix.
        digitize, n_bins : bool, int
            Whether to binarize the perturbation matrix. Used to add sharpness and jiggle to horizons.
        output_range : tuple of two numbers
            Range to scale the perturbation, which defines the overall direction.
            For example, (0.0, 1.0) value would mean that perturbation shifts the surface only in the down direction.
        rng : np.random.Generator or int, optional
            Random Number Generator or seed to create one.
        """
        # Parse parameters
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        sample_size = sample_size or min(10000, 5*np.prod(shape))

        # Generate locations for gaussians
        locs_n = rng.integers(*locs_n_range, dtype=np.int32)
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
            matrix -= 1
            matrix /= n_bins

        matrix = (output_range[1] - output_range[0]) * matrix + output_range[0]
        return matrix

    @staticmethod
    def make_shift_vector(num_points, mode='sin2', num_zeros=0.2, perturb_peak=True, perturb_values=True, rng=None):
        """ Create a vector of pixel shifts.

        Parameters
        ----------
        mode : {'sin2'}
            Mode for making shift vector. Changes the pattern of shifts along chosen direction.
        num_zeros : number
            Proportion of zeros on the sides of created `shift_vector`.
            0.2 means that rougly 10% from each side are going to be filled with zeros.
        perturb_peak : bool
            Whether to randomly shift the peak of `shift_vector` to the sides.
        perturb_values : bool
            Whether to add small randomization to values in the `shift_vector`.
        """
        # TODO: dtypes? Do we even need them here (output of this function used for map_coordinates only)
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
        """ Apply gaussian filter in 3d in optimal way (with opencv convolutions). """
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
    """ Compute the velocity model by depth-wise stretching `velocity_vector` along `horizon_matrices`. """
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
    """ Compute reflectivity as the ratio between differences and sums for successive layers of impedance.
    To condition this fraction, we add the doubled ~mean impedance to the denominator.

    reflectivity = ((impedance[..., 1:] - impedance[..., :-1]) /
                    (impedance[..., 1:] + impedance[..., :-1]))
    """
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
    """ Inplace addition of ones into `matrix` at `indices_1, indices_2` position.
    The difference with numpy indexing `matrix[indices_1, indices_2] += 1` is in fact that numpy would not add twice
    at the same element, resulting in different outputs.
    """
    for i_1, i_2 in zip(indices_1, indices_2):
        matrix[i_1, i_2] += 1
    return matrix


@njit
def find_depths(array, idx_x, depths):
    """ For each pair of (x coordinate, depth) from `idx_x` and `depths`, find the closest value in `array`.
    Early breaks if found exactly that depth.
    """
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
