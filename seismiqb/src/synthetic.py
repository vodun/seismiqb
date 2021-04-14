""" Functions for generation of 2d and 3d synthetic seismic arrays.
"""
import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter
from scipy.signal import ricker
from numba import njit


def make_surfaces(num_surfaces, grid_shape, shape, kind='cubic', perturbation_share=0.25, shares=None):
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
    # check shapes and select interpolation-method
    grid_shape = (grid_shape, ) if isinstance(grid_shape, int) else grid_shape
    if len(shape) != len(grid_shape) + 1:
        raise ValueError('`shape` and `grid_shape` parameters should match.')

    if len(shape) == 2:
        interp = interp1d
    elif len(shape) == 3:
        interp = interp2d
    else:
        raise ValueError('The function only supports the generation of 1d curves and 2d-surfaces.')

    # make the grid
    grid = [np.linspace(0, 1, num_points) for num_points in grid_shape]

    # make the first curve
    curves = [np.zeros(grid_shape)]
    shares = shares if shares is not None else np.ones((num_surfaces, ))
    shares = np.array(shares) / np.sum(shares)
    for i in range(num_surfaces):
        delta_h = shares[i]
        epsilon = perturbation_share * delta_h

        # make each curve in unit-terms
        curves.append(curves[-1] + delta_h * np.ones_like(curves[0])
                      + np.random.uniform(low=-epsilon, high=epsilon, size=curves[0].shape))

    # interpolate and scale each curve to cube-shape
    results = []
    for curve in curves:
        func = interp(*grid, curve, kind=kind)
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
def make_colors_array_2d(colors, levels, shape):
    """ Color 2d-array in colors according to given levels.
    """
    array = np.zeros(shape=shape)
    for i in range(array.shape[0]):
        vec = array[i, :]
        for j in range(len(colors)):
            low = np.minimum(levels[j][i], array.shape[-1])
            vec[low : ] = colors[j]
    return array


@njit
def make_colors_array_3d(colors, levels, shape):
    """ Color 3d-array in colors according to given levels.
    """
    array = np.zeros(shape=shape)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            vec = array[i, j, :]
            for k in range(len(colors)):
                low = np.minimum(levels[k][i, j], array.shape[-1])
                vec[low : ] = colors[k]
    return array


def make_synthetic(shape=(50, 400, 800), num_reflections=200, vel_limits=(900, 5400), horizon_heights=(1/4, 1/2, 2/3),
                   horizon_jumps=(7, 5, 4), grid_shape=(10, 10), perturbation_share=.2, rho_noise_lims=(0.97, 1.3),
                   ricker_width=5, ricker_points=50, sigma=1.1, noise_mul=0.5, fetch_surfaces='horizons'):
    """ Generate synthetic 3d-cube.

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
    horizon_jumps : tuple
        Mutipliers controling the magnitide of sharp jumps. Should have the same length as `horizon_heights`-arg.
    grid_shapes : tuple
        Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
    perturbation_share : float
        Sets the limit of random perturbation for surfaces' creation. The limit is set relative to the depth
        of a layer of constant velocity. The larger the value, more 'curved' are the horizons.
    rho_noise_lims : tuple or None
        Density (rho)-model is given by (velocity model * noise). The param sets the limits for noise.
        If set to None, rho-model is equal to velocity-model.
    ricker_width : float
        Width of the ricker-wave - `a`-parameter of `scipy.signal.ricker`.
    ricker_points : int
        Number of points in the ricker-wave - `points`-parameter of `scipy.signal.ricker`.
    sigma : float or None
        sigma used for gaussian blur of the synthetic seismic.
    noise_mul : float or None
        If not None, gaussian noise scale by this number is applied to the synthetic.
    fetch_surfaces : str
        Can be either 'horizons', 'all' or None. When 'horizons', only horizon-surfaces
        (option `horizon_heights`) are returned. Choosing 'all' allows to return all of
        the reflections, while 'topK' option leads to fetching K surfaces correpsonding
        to K largest jumps in velocities-array.
    """
    if len(shape) in (2, 3):
        dim = len(shape)
    else:
        raise ValueError('The function only supports the generation of 2d and 3d synthetic seismic.')

    # generate array of velocities
    low, high = vel_limits
    llim = (high - low) / num_reflections
    velocities = (np.linspace(low, high, num_reflections + 1) +
                  np.random.uniform(low=-llim, high=llim, size=(num_reflections + 1, )))

    for height_share, jump_mul in zip(horizon_heights, horizon_jumps):
        velocities[int(velocities.shape[0] * height_share)] += llim * jump_mul

    # make velocity model
    curves = make_surfaces(num_reflections, grid_shape, perturbation_share=perturbation_share, shape=shape)
    make_colors_array = make_colors_array_2d if dim == 2 else make_colors_array_3d
    vel_model = make_colors_array(velocities, curves, shape)

    # make density model
    if rho_noise_lims is not None:
        rho = vel_model * np.random.uniform(*rho_noise_lims, size=vel_model.shape)
    else:
        rho = vel_model

    # obtain synthetic
    ref_coeffs = reflectivity(vel_model, rho)
    wavelet = ricker(ricker_points, ricker_width)
    convolve = convolve_2d if dim == 2 else convolve_3d
    result = convolve(ref_coeffs, wavelet)

    # add blur and noise if needed for a more realistic image
    if sigma is not None:
        result = gaussian_filter(result, sigma=sigma)
    if noise_mul is not None:
        result += noise_mul * np.random.random(result.shape) * result.std()

    # fetch horizons if needed
    if isinstance(fetch_surfaces, str):
        if fetch_surfaces == 'all':
            return result, curves
        if fetch_surfaces == 'horizons':
            return result, curves[[int(curves.shape[0] * height_share) for height_share in horizon_heights]]
        if 'top' in fetch_surfaces:
            top_k = int(fetch_surfaces.replace('top', ''))
            ixs = np.argsort(velocities)[::-1][top_k]
            return result, curves[ixs]
    return result
