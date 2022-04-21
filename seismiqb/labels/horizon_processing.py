""" Mixin for horizon processing. """
from math import isnan
import numpy as np
from numba import njit, prange

from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes, binary_dilation, binary_erosion

from ..functional import make_gaussian_kernel
from ..utils import make_bezier_figure

class ProcessingMixin:
    """ Methods for horizon processing.

    This class contains methods which can be divided into the following groups:
        - Filtering methods to cut out some surface regions.
        - Horizon transformations such as smoothing and thinning.
        - Surface distortions such as holes or carcass creation.

    Note, almost all of these methods can change horizon surface inplace or create a new instance.
    In either case they return a filtered horizon instance.
    """
    # Filtering methods
    def filter(self, filtering_matrix=None, margin=0, inplace=True, **_):
        """ Remove points that correspond to 1's in `filtering_matrix` from the horizon surface.

        Note, this method can change horizon inplace or create a new instance.
        In either case it returns a filtered horizon instance.

        Parameters
        ----------
        filtering_matrix : np.ndarray
            Mask of points to cut out from the horizon.
            If None, then remove points corresponding to zero traces.
        margin : int
            Amount of traces to cut out near to boundaries considering `filtering_matrix` appliance.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if filtering_matrix is None:
            filtering_matrix = self.field.zero_traces

        if margin > 0:
            filtering_matrix = binary_dilation(filtering_matrix, structure=np.ones((margin, margin)))

            filtering_matrix[:margin, :] = 1
            filtering_matrix[:, :margin] = 1
            filtering_matrix[-margin:, :] = 1
            filtering_matrix[:, -margin:] = 1

        mask = filtering_matrix[self.points[:, 0], self.points[:, 1]]
        points = self.points[mask == 0]

        if inplace:
            self.points = points
            self.reset_storage('matrix')
            return self
        else:
            name = 'filtered_' + self.name if self.name is not None else None
            return type(self)(storage=points, field=self.field, name=name)

    def filter_spikes(self, spike_spatial_maxsize=7, spike_depth_minsize=5, close_depths_threshold=2,
                      dilation_iterations=0, inplace=True):
        """ Remove spikes from the horizon.

        Note, this method can change horizon inplace or create a new instance. By default works inplace.
        In either case it returns a filtered horizon instance.
        """
        spikes_mask = self.load_attribute('spikes', spike_spatial_maxsize=spike_spatial_maxsize,
                                          spike_depth_minsize=spike_depth_minsize,
                                          close_depths_threshold=close_depths_threshold,
                                          dilation_iterations=dilation_iterations)

        return self.filter(spikes_mask, inplace=inplace)

    despike = filter_spikes

    def filter_disconnected_regions(self, erosion_rate=0, inplace=True):
        """ Remove regions, not connected to the largest component of a horizon.

        Note, this method can change horizon inplace or create a new instance. By default works inplace.
        In either case it returns a filtered horizon instance.
        """
        if erosion_rate > 0:
            structure = np.ones((3, 3))
            matrix = binary_erosion(self.mask, structure, iterations=erosion_rate)
        else:
            matrix = self.mask

        labeled = label(matrix)
        values, counts = np.unique(labeled, return_counts=True)
        counts = counts[values != 0]
        values = values[values != 0]

        object_id = values[np.argmax(counts)]

        filtering_matrix = np.zeros_like(self.mask)
        filtering_matrix[labeled == object_id] = 1

        if erosion_rate > 0:
            filtering_matrix = binary_dilation(filtering_matrix, structure, iterations=erosion_rate)

        filtering_matrix = filtering_matrix == 0

        return self.filter(filtering_matrix, inplace=inplace)


    # Horizon surface transforms
    def thin_out(self, factor=1, threshold=256, inplace=True):
        """ Thin out the horizon by keeping only each `factor`-th line.

        Note, this method can change horizon inplace or create a new instance. By default works inplace.
        In either case it returns a filtered horizon instance.

        Parameters
        ----------
        factor : integer or sequence of two integers
            Frequency of lines to keep along ilines and xlines direction.
        threshold : integer
            Minimal amount of points in a line to keep.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if isinstance(factor, int):
            factor = (factor, factor)

        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        mask_i = np.isin(self.points[:, 0], uniques[counts > threshold][::factor[0]])

        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        mask_x = np.isin(self.points[:, 1], uniques[counts > threshold][::factor[1]])

        points = self.points[mask_i + mask_x]

        if inplace:
            self.points = points
            self.reset_storage('matrix')
            return self
        else:
            name = 'thinned_' + self.name if self.name is not None else None
            return type(self)(storage=points, field=self.field, name=name)

    def smooth_out(self, mode='convolve', kernel=None, kernel_size=3, iters=1, preserve_missings=True,
                   distance_threshold=5, sigma=0.8, inplace=True):
        """ Convolve the horizon with gaussian kernel with special treatment to absent points:
        if the point was present in the original horizon, then it is changed to a weighted sum of all
        present points nearby;
        if the point was absent in the original horizon and there is at least one non-fill point nearby,
        then it is changed to a weighted sum of all present points nearby.

        Note, this method can change horizon inplace or create a new instance. By default works inplace.
        In either case it returns a filtered horizon instance.

        Parameters
        ----------
        mode : str
            convolve or bilateral
        kernel : ndarray or None
            If passed, then ready-to-use kernel. Otherwise, gaussian kernel will be created.
        kernel_size : int
            Size of gaussian filter.
        iters : int
            Number of times to apply smoothing filter.
        preserve_missings : bool
            Whether or not to allow method label additional points.
        distance_threshold : number
            If the distance between anchor point and the point inside filter is bigger than the threshold,
            then the point is ignored in convolutions.
            Can be used for separate smoothening on sides of discontinuity.
        sigma : number
            Standard deviation (spread or “width”) for gaussian kernel.
            The lower, the more weight is put into the point itself.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        result = self.matrix_smooth_out(matrix=self.matrix, mode=mode, kernel=kernel, kernel_size=kernel_size,
                                        sigma=sigma, distance_threshold=distance_threshold, iters=iters,
                                        preserve_missings=preserve_missings)

        if inplace:
            self.matrix = result
            self.reset_storage('points')
            return self
        else:
            name = 'smoothed_' + self.name if self.name is not None else None
            return type(self)(storage=result, field=self.field, name=name)

    def interpolate(self, kernel=None, kernel_size=3, iters=1, min_neighbors=0, max_distance_threshold=None,
                    sigma=0.8, inplace=True):
        """ Interpolate horizon surface on the regions with missing traces.

        Under the hood, we fill missing traces with weighted neighbor values.

        Note, this method can change horizon inplace or create a new instance. By default works inplace.
        In either case it returns a filtered horizon instance.

        Parameters
        ----------
        kernel : ndarray or None
            Kernel to apply to missing points.
        kernel_size : int
            If the kernel is not provided, shape of the square gaussian kernel.
        iters : int
            Number of interpolation iterations to perform.
        min_neighbors: int or float
            Minimal of non-missing neighboring points in a window to interpolate a central point.
            If int, then it is an amount of points.
            If float, then it is a points ratio.
        max_distance_threshold : number
            A maximum distance between values in a squared window for which we apply interpolation.
        sigma : float
            Standard deviation for a gaussian kernel creation.
        inplace : bool
            Whether to apply operation inplace or return a new Horizon object.

        Returns
        -------
        :class:`~.Horizon`
            Processed horizon instance. A new instance if `inplace` is False, `self` otherwise.
        """
        if kernel is None:
            kernel = make_gaussian_kernel(kernel_size=kernel_size, sigma=sigma)

        if isinstance(min_neighbors, float):
            min_neighbors = round(min_neighbors * kernel.size)

        result = self.matrix.astype(np.float32)
        result[self.matrix == self.FILL_VALUE] = np.nan

        # Apply `_interpolate` multiple times. Note that there is no dtype conversion in between
        # Also the method returns a new object
        for _ in range(iters):
            result = _interpolate(src=result, kernel=kernel, min_neighbors=min_neighbors,
                                  max_distance_threshold=max_distance_threshold)

        result[np.isnan(result)] = self.FILL_VALUE
        result = np.rint(result).astype(np.int32)

        result[self.field.zero_traces[self.i_min:self.i_max + 1,
                                      self.x_min:self.x_max + 1] == 1] = self.FILL_VALUE

        if inplace:
            self.matrix = result
            self.reset_storage('points')
            return self
        else:
            name = 'interpolated_' + self.name if self.name is not None else None
            return type(self)(storage=result, field=self.field, name=name)

    # Horizon distortions
    def make_carcass(self, frequencies=100, regular=True, margin=50, apply_smoothing=False, add_prefix=True, **kwargs):
        """ Cut carcass out of a horizon. Returns a new instance.

        Parameters
        ----------
        frequencies : int or sequence of ints
            Frequencies of carcass lines.
        regular : bool
            Whether to make regular lines or base lines on geometry quality map.
        margin : int
            Margin from geometry edges to exclude from carcass.
        apply_smoothing : bool
            Whether to smooth out the result.
        kwargs : dict
            Other parameters for grid creation, see `:meth:~.SeismicGeometry.make_quality_grid`.
        """
        #pylint: disable=import-outside-toplevel
        frequencies = frequencies if isinstance(frequencies, (tuple, list)) else [frequencies]
        carcass = self.copy(add_prefix=add_prefix)
        carcass.name = carcass.name.replace('copy', 'carcass')

        if regular:
            from ..metrics import GeometryMetrics
            gm = GeometryMetrics(self.field.geometry)
            grid = gm.make_grid(1 - self.field.zero_traces, frequencies=frequencies, margin=margin, **kwargs)
        else:
            grid = self.field.geometry.make_quality_grid(frequencies, margin=margin, **kwargs)

        carcass.filter(filtering_matrix=1-grid)
        if apply_smoothing:
            carcass.smooth_out(preserve_missings=False)
        return carcass

    def generate_holes_matrix(self, n=10, scale=1.0, max_scale=.25,
                              max_angles_amount=4, max_sharpness=5.0, locations=None,
                              points_proportion=1e-5, points_shape=1,
                              noise_level=0, seed=None):
        """ Create matrix of random holes for horizon.

        Holes can be bezier-like figures or points-like.
        We can control bezier-like and points-like holes amount by `n` and `points_proportion` parameters respectively.
        We also do some noise amplifying with `noise_level` parameter.

        Parameters
        ----------
        n : int
            Amount of bezier-like holes on horizon.
        points_proportion : float
            Proportion of point-like holes on the horizon. A number between 0 and 1.
        points_shape : int or sequence of int
            Shape of point-like holes.
        noise_level : int
            Radius of noise scattering near the borders of holes.
        scale : float or sequence of float
            If float, each bezier-like hole will have a random scale from exponential distribution with parameter scale.
            If sequence, each bezier-like hole will have a provided scale.
        max_scale : float
            Maximum bezier-like hole scale.
        max_angles_amount : int
            Maximum amount of angles in each bezier-like hole.
        max_sharpness : float
            Maximum value of bezier-like holes sharpness.
        locations : ndarray
            If provided, an array of desired locations of bezier-like holes.
        seed : int, optional
            Seed the random numbers generator.
        """
        rng = np.random.default_rng(seed)
        filtering_matrix = np.zeros_like(self.full_matrix)

        # Generate bezier-like holes
        # Generate figures scales
        if isinstance(scale, float):
            scales = []
            sampling_scale = int(
                np.ceil(1.0 / (1 - np.exp(-scale * max_scale)))
            ) # inverse probability of scales < max_scales
            while len(scales) < n:
                new_scales = rng.exponential(scale, size=sampling_scale*(n - len(scales)))
                new_scales = new_scales[new_scales <= max_scale]
                scales.extend(new_scales)
            scales = scales[:n]
        else:
            scales = scale

        # Generate figures-like holes locations
        if locations is None:
            idxs = rng.choice(len(self), size=n)
            locations = self.points[idxs, :2]

        coordinates = [] # container for all types of holes, represented by their coordinates

        # Generate figures inside the field
        for location, figure_scale in zip(locations, scales):
            n_key_points = rng.integers(2, max_angles_amount + 1)
            radius = rng.random()
            sharpness = rng.random() * rng.integers(1, max_sharpness)

            figure_coordinates = make_bezier_figure(n=n_key_points, radius=radius, sharpness=sharpness,
                                                    scale=figure_scale, shape=self.shape, seed=seed)
            figure_coordinates += location

            # Shift figures if they are out of field bounds
            negative_coords_shift = np.min(np.vstack([figure_coordinates, [0, 0]]), axis=0)
            huge_coords_shift = np.max(np.vstack([figure_coordinates - self.shape, [0, 0]]), axis=0)
            figure_coordinates -= (huge_coords_shift + negative_coords_shift + 1)

            coordinates.append(figure_coordinates)

        # Generate points-like holes
        if points_proportion:
            points_n = int(points_proportion * len(self))
            idxs = rng.choice(len(self), size=points_n)
            locations = self.points[idxs, :2]

            filtering_matrix[locations[:, 0], locations[:, 1]] = 1

            if isinstance(points_shape, int):
                points_shape = (points_shape, points_shape)
            filtering_matrix = binary_dilation(filtering_matrix, np.ones(points_shape))

            coordinates.append(np.argwhere(filtering_matrix > 0))

        coordinates = np.concatenate(coordinates)

        # Add noise and filtering matrix transformations
        if noise_level:
            noise = rng.normal(loc=coordinates,
                               scale=noise_level,
                               size=coordinates.shape)
            coordinates = np.unique(np.vstack([coordinates, noise.astype(int)]), axis=0)

        # Add valid coordinates onto filtering matrix
        idx = np.where((coordinates[:, 0] >= 0) &
                       (coordinates[:, 1] >= 0) &
                       (coordinates[:, 0] < self.i_length) &
                       (coordinates[:, 1] < self.x_length))[0]
        coordinates = coordinates[idx]

        filtering_matrix[coordinates[:, 0], coordinates[:, 1]] = 1

        # Process holes
        filtering_matrix = binary_fill_holes(filtering_matrix)
        filtering_matrix = binary_dilation(filtering_matrix, iterations=4)
        return filtering_matrix

    def make_holes(self, inplace=False, n=10, scale=1.0, max_scale=.25,
                   max_angles_amount=4, max_sharpness=5.0, locations=None,
                   points_proportion=1e-5, points_shape=1,
                   noise_level=0, seed=None):
        """ Make holes in a horizon.

        Note, this method can change horizon inplace or create a new instance. By default creates a new instance.
        In either case it returns a filtered horizon instance.
        """
        #pylint: disable=self-cls-assignment
        filtering_matrix = self.generate_holes_matrix(n=n, scale=scale, max_scale=max_scale,
                                                      max_angles_amount=max_angles_amount,
                                                      max_sharpness=max_sharpness, locations=locations,
                                                      points_proportion=points_proportion, points_shape=points_shape,
                                                      noise_level=noise_level, seed=seed)

        return self.filter(filtering_matrix, inplace=inplace)

    make_holes.__doc__ += '\n' + '\n'.join(generate_holes_matrix.__doc__.split('\n')[1:])

# Helper functions
@njit(parallel=True)
def _interpolate(src, kernel, min_neighbors=1, max_distance_threshold=None):
    """ Jit-accelerated function to apply 2d interpolation to nan values. """
    #pylint: disable=too-many-nested-blocks, consider-using-enumerate, not-an-iterable
    k = kernel.shape[0] // 2
    raveled_kernel = kernel.ravel() / np.sum(kernel)

    i_range, x_range = src.shape
    dst = src.copy()

    for iline in prange(0, i_range):
        for xline in range(0, x_range):
            central = src[iline, xline]

            if not isnan(central):
                continue # We interpolate values only to nan points

            # Get neighbors and check whether we can interpolate them
            element = src[max(0, iline-k):min(iline+k+1, i_range),
                          max(0, xline-k):min(xline+k+1, x_range)].ravel()

            notnan_neighbors = kernel.size - np.isnan(element).sum()
            if notnan_neighbors < min_neighbors:
                continue

            # Compare ptp with the max_distance_threshold
            if max_distance_threshold is not None:
                nanmax, nanmin = np.float32(element[0]), np.float32(element[0])

                for item in element:
                    if not isnan(item):
                        if isnan(nanmax):
                            nanmax = item
                            nanmin = item
                        else:
                            nanmax = max(item, nanmax)
                            nanmin = min(item, nanmin)

                if nanmax - nanmin > max_distance_threshold:
                    continue

            # Apply kernel to neighbors to get value for interpolated point
            s, sum_weights = np.float32(0), np.float32(0)
            for item, weight in zip(element, raveled_kernel):
                if not isnan(item):
                    s += item * weight
                    sum_weights += weight

            if sum_weights != 0.0:
                dst[iline, xline] = s / sum_weights
    return dst
