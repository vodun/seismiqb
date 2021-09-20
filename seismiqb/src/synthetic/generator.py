""" Functions for generation of 2d and 3d synthetic seismic arrays.
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation
from scipy.signal import ricker
from numba import njit


def make_surfaces(num_surfaces, grid_shape, shape, kind='cubic', perturbation_share=0.25, shares=None,
                  rng=None, seed=None):
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
    rng : np.random.Generator or None
        Generator of random numbers.
    seed : int or None
        Seed used for creation of random generator (check out `np.random.default_rng`).

    Returns
    -------
    np.ndarray
        Array of size num_surfaces X shape[:2] representing resulting surfaces-heights.
    """
    rng = rng or np.random.default_rng(seed)

    # check shapes and select interpolation-method
    grid_shape = (grid_shape, ) if isinstance(grid_shape, int) else grid_shape
    if len(shape) != len(grid_shape) + 1:
        raise ValueError("`(len(shape) - 1)` should be equal to `len(grid_shape)`.")

    if len(shape) == 2:
        interp = interp1d
    elif len(shape) == 3:
        interp = interp2d
    else:
        raise ValueError('The function only supports the generation of 1d and 2d-surfaces.')

    # make the grid
    grid = [np.linspace(0, 1, num_points) for num_points in grid_shape]

    # make the first surface
    surfaces = [np.zeros(grid_shape)]
    shares = shares if shares is not None else np.ones((num_surfaces, ))
    shares = np.array(shares) / np.sum(shares)
    for delta_h in shares:
        epsilon = perturbation_share * delta_h

        # make each surface in unit-terms
        surfaces.append(surfaces[-1] + delta_h * np.ones_like(surfaces[0])
                      + rng.uniform(low=-epsilon, high=epsilon, size=surfaces[0].shape))

    # interpolate and scale each surface to cube-shape
    results = []
    for surface in surfaces:
        func = interp(*grid, surface, kind=kind)
        results.append((func(*[np.arange(num_points) / num_points for num_points in shape[:-1]])
                        * shape[-1]).astype(np.int).T)
    return np.array(results)


def reflectivity(v, rho):
    """ Compute reflectivity coefficients given velocity and density models.
    Velocities and reflectivity coefficients can be either 2d or 3d.
    """
    rc = np.zeros_like(v)
    v_rho = v * rho
    rc[..., 1:] = (v_rho[..., 1:] - v_rho[..., :-1]) / (v_rho[..., 1:] + v_rho[..., :-1])
    return rc

@njit
def convolve_2d(array, kernel):
    """ Shape-preserving vector-wise convolution of a 2d-array with a kernel-vector.
    """
    # calculate offsets to trim arrays resulting from the convolution
    result = np.zeros_like(array)
    left_offset = (len(kernel) - 1) // 2
    right_offset = len(kernel) - 1 - left_offset

    for i in range(array.shape[0]):
        result[i, :] = np.convolve(array[i], kernel)[left_offset:-right_offset]
    return result


@njit
def convolve_3d(array, kernel):
    """ Shape-preserving vector-wise convolution of a 3d-array with a kernel-vector.
    """
    # calculate offsets to trim arrays resulting from the convolution
    result = np.zeros_like(array)
    left_offset = (len(kernel) - 1) // 2
    right_offset = len(kernel) - 1 - left_offset

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            result[i, j, :] = np.convolve(array[i, j], kernel)[left_offset:-right_offset]
    return result


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


def make_elastic_distortion(xs, rng=None, seed=None, n_points=10, zeros_share=0.2, kind='cubic',
                            perturb_values=True, perturb_peak=True, random_invert=True):
    """ Generate a hump-shaped distortion [0, 1] -> [0, 1] and apply it to a set of points.
    The transformation has form f(x) = x + distortion * mul. It represents an
    elastic transform of an image represented by `xs`. The left and the right tails of the
    distortion can be filled with zeros. Also, the peak of the hump can be randomly shifted to
    left or right. In addition, when needed, the distortion itself can be randomly inverted.
    """
    rng = rng or np.random.default_rng(seed)
    points = np.linspace(0, 1, n_points)

    # compute length of prefix of zeros
    n_zeros = int(n_points * zeros_share)
    if perturb_peak:
        delta_position = rng.integers(-n_zeros // 4, n_zeros // 4 + 1)
    else:
        delta_position = 0
    prefix_length = n_zeros // 2 + delta_position

    # form the values-hump and perturb it if needed
    values = np.zeros((n_points, ))
    n_values = n_points - n_zeros
    half_hump = np.linspace(0, 1, n_values // 2 + 1 + n_values % 2)[1:]
    hump = np.concatenate([half_hump, half_hump[::-1][n_values % 2:]])
    if perturb_values:
        step = 2 / n_values
        hump += rng.uniform(-step / 2, step / 2, (n_values, ))
    values[prefix_length: prefix_length + len(hump)] = hump
    spline = interp1d(points, values, kind=kind)

    # possibly invert the elastic transform
    if random_invert:
        if rng.choice([True, False]):
            return -spline(xs)
    return spline(xs)

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
        self.dim = None
        self.rng = rng or np.random.default_rng(seed)
        self.velocities = None
        self.velocity_model = None
        self.density_model = None
        self.synthetic = None
        self._reflection_surfaces = None
        self._horizon_heights = ()
        self._faults_coords = ()
        self._mask = None

    def make_velocities(self, num_reflections=200, vel_limits=(900, 5400), horizon_heights=(1/4, 1/2, 2/3),
                        horizon_multipliers=(7, 5, 4)):
        """ Generate and store array of velocities. Roughly speaking, seismic slide is a stack of layers of constant
        velocities. This method generates the array of velocity-values, that are to be used later for making of
        velocity model.

        Parameters
        ----------
        num_reflections : int
            The number of reflective surfaces.
        vel_limits : sequence
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
        low, high = vel_limits
        velocity_delta = (high - low) / num_reflections
        self.velocities = (np.linspace(low, high, num_reflections + 1) +
                           self.rng.uniform(low=-velocity_delta, high=velocity_delta, size=(num_reflections + 1, )))

        for height_share, jump_mul in zip(horizon_heights, horizon_multipliers):
            self.velocities[int(self.velocities.shape[0] * height_share)] += velocity_delta * jump_mul

        self._horizon_heights = horizon_heights
        return self

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
        if len(shape) in (2, 3):
            self.dim = len(shape)
        else:
            raise ValueError('The function only supports the generation of 2d and 3d synthetic seismic.')

        num_reflections = len(self.velocities) - 1
        surfaces = make_surfaces(num_reflections, grid_shape, perturbation_share=perturbation_share,
                                 shape=shape, rng=self.rng)
        _make_velocity_model = _make_velocity_model_2d if self.dim == 2 else _make_velocity_model_3d
        self.velocity_model = _make_velocity_model(self.velocities, surfaces, shape)

        # store surfaces-list to later use them as horizons
        self._reflection_surfaces = surfaces
        return self

    def _add_fault(self, fault_coordinates, num_points, max_shift, zeros_share, kind,
                   perturb_values, perturb_peak, random_invert, fetch_and_update_mask):
        """ Add fault to a velocity model.
        """
        x0, x1 = fault_coordinates[0][0], fault_coordinates[1][0]
        y0, y1 = fault_coordinates[0][1], fault_coordinates[1][1]
        x_low, x_high, y_low, y_high = (0, self.velocity_model.shape[0], min(y0, y1), max(y0, y1))

        # y-axis coordinate shift
        y0, y1 = y0 - y_low, y1 - y_low

        # coeffs of the line equation x = ky + b
        k = (x1 - x0) / (y1 - y0)
        b = (x0 * y1 - x1 * y0) / (y1 - y0)
        kx, ky = (k, 1)

        # make preparations for coordinate-map (i.e. elastic transform)
        xs, ys = np.meshgrid(np.arange(x_low, x_high), np.arange(0, y_high - y_low))

        # 0 to the left of the fault, 1 to the right
        indicator = (np.sign(xs - k * ys - b) + 1) / 2

        # compute measure of closeness of a point to the fault-center
        closeness = make_elastic_distortion(ys / (y_high - y_low), self.rng, n_points=num_points,
                                            perturb_peak=perturb_peak, perturb_values=perturb_values,
                                            kind=kind, zeros_share=zeros_share, random_invert=random_invert)

        # compute vector field for a coordinate-map and apply the map to the seismic
        delta_xs, delta_ys = (max_shift * kx * indicator * closeness,
                              max_shift * ky * indicator * closeness)
        crop = self.velocity_model[x_low:x_high, y_low:y_high]
        crop_elastic = map_coordinates(crop.astype(np.float32),
                                       (xs + delta_xs, ys + delta_ys),
                                       mode='nearest').T
        self.velocity_model[x_low:x_high, y_low:y_high] = crop_elastic

        # adjust mask if needed
        if fetch_and_update_mask is not None:
            if isinstance(fetch_and_update_mask, str):
                fetch_and_update_mask = {'mode': fetch_and_update_mask}
            fetch_and_update_mask['horizon_format'] = 'mask'

            # make mask and apply the same coordinate-map to it
            mask = self.fetch_horizons(**fetch_and_update_mask)
            crop = mask[x_low:x_high, y_low:y_high]
            crop_elastic = map_coordinates(crop.astype(np.int32),
                                           (xs + delta_xs, ys + delta_ys),
                                           mode='nearest').T

            # update the mask
            mask[x_low:x_high, y_low:y_high] = crop_elastic
            self._mask = mask

    def add_faults(self, faults=(((100, 50), (100, 370)),
                                 ((50, 320), (50, 470)),
                                 ((150, 320), (150, 470))),
                   num_points=10, max_shift=10, zeros_share=0.6, kind='cubic', perturb_values=True,
                   perturb_peak=False, random_invert=False, fetch_and_update_mask='horizons'):
        """ Add faults to the velocity model. Faults are basically elastic transforms of patches of
        generated seismic images. Elastic transforms are performed through coordinates-transformation
        in depth-projection. Those are smooth maps [0, 1] -> [0, 1] described as f(x) = x + distortion.
        In current version, distortions are always hump-shaped. Almost all parameters of the function
        are used to define properties of the hump-shaped distortion.

        Parameters
        ----------
        faults : sequence
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

        self._faults_coords = faults
        for fault in faults:
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
        ref_coeffs = reflectivity(self.velocity_model, self.density_model)
        wavelet = ricker(ricker_points, ricker_width)
        convolve = convolve_2d if self.dim == 2 else convolve_3d
        self.synthetic = convolve(ref_coeffs, wavelet)
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

    def _make_enumerated_mask(self, surfaces):
        """ Make enumerated mask from a sequence of surfaces. Each surfaces is marked by its ordinal
        number from `range(1, len(surfaces) + 1)` on a resulting mask.
        """
        mask = np.zeros_like(self.velocity_model)
        for i, horizon in enumerate(surfaces):
            mesh = np.meshgrid(*[np.arange(axis_shape) for axis_shape in horizon.shape])
            mask[(*mesh, horizon)] = i + 1
        return mask

    def _enumerated_to_heights(self, mask):
        """ Convert enumerated mask to heights.
        """
        surfaces = []
        n_levels = len(np.unique(mask)) - 1
        for i in range(1, n_levels + 1):
            heights = np.where(mask == i)[-1].reshape(self._reflection_surfaces[0].shape)
            surfaces.append(heights)
        return surfaces

    @staticmethod
    def _add_surface_to_mask(surface, mask):
        """ Add horizon-surface to mask.
        """
        mesh = np.meshgrid(*[np.arange(axis_shape) for axis_shape in surface.shape])
        mask[(*mesh, surface)] = 1

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
        if mode == 'all':
            indices = slice(0, None)
        elif mode == 'horizons':
            indices = [int(self._reflection_surfaces.shape[0] * height_share)
                       for height_share in self._horizon_heights]
        elif 'top' in mode:
            top_k = int(mode.replace('top', ''))
            indices = np.argsort(np.abs(np.diff(self.velocities)))[::-1][:top_k]
        else:
            raise ValueError('Mode can be one of `horizons`, `all` or `top[k]`')
        surfaces = self._reflection_surfaces[indices]

        if horizon_format == 'heights':
            return surfaces
        if horizon_format == 'mask':
            if self._mask is not None:
                return self._mask
            mask = np.zeros_like(self.velocity_model)
            for surface in surfaces:
                self._add_surface_to_mask(surface, mask)

            # add width to horizon-mask if needed
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
        # convert each fault to the point-cloud format
        point_clouds = []
        for fault in self._faults_coords:
            x0, x1 = fault[0][0], fault[1][0]
            y0, y1 = fault[0][1], fault[1][1]
            y_low, y_high = min(y0, y1), max(y0, y1)

            # coeffs of the line equation x = ky + b
            k = (x1 - x0) / (y1 - y0)
            b = (x0 * y1 - x1 * y0) / (y1 - y0)

            heights = np.arange(y_low, y_high)
            ilines, xlines = np.zeros_like(heights), (np.rint(k * heights + b)).astype(np.int)
            point_cloud = np.stack([ilines, xlines, heights], axis=1)
            point_clouds.append(point_cloud)

        # form masks out of point clouds if needed
        if faults_format == 'mask':
            mask = np.zeros_like(self.velocity_model)
            for point_cloud in point_clouds:
                xlines, heights = point_cloud[:, 1], point_cloud[:, 2]
                mask[xlines, heights] = 1

            # add width to faults-mask if needed
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


def generate_synthetic(shape=(50, 400, 800), num_reflections=200, vel_limits=(900, 5400), #pylint: disable=too-many-arguments
                       horizon_heights=(1/4, 1/2, 2/3), horizon_multipliers=(7, 5, 4), grid_shape=(10, 10),
                       perturbation_share=.2, density_noise_lims=(0.97, 1.3),
                       ricker_width=5, ricker_points=50, sigma=1.1, noise_mul=0.5,
                       faults=(((100, 50), (100, 370)),
                               ((50, 320), (50, 470)),
                               ((150, 320), (150, 470))),
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
    vel_limits : tuple
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
    faults : tuple or list
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
           .make_velocities(num_reflections, vel_limits, horizon_heights, horizon_multipliers)
           .make_velocity_model(shape, grid_shape, perturbation_share))

    # add faults if needed and possible
    if faults is not None:
        if len(faults) > 0:
            if dim == 2:
                fetch_and_update = {'mode': fetch_surfaces, 'horizon_format': geobodies_format[0],
                                    'width': geobodies_width[0]}
                gen.add_faults(faults, num_points_faults, max_shift, zeros_share_faults, fault_shift_interpolation,
                               perturb_values, perturb_peak, random_invert, fetch_and_update)
            else:
                raise ValueError("For now, faults are only supported for dim = 2.")

    gen = (gen.make_density_model(density_noise_lims)
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
