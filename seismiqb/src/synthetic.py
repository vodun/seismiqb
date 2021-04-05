import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter


def make_curves_v1(num_curves, num_points, shape=(100, 100),
                   kind='cubic', perturbation_share=0.4, shares=None):
    """ First param is num_curves the second controls how volatile is the curve. Might wanna
    try changing the parameter with curve.
    """
    # make an array of curve-supports
    grid_x = np.linspace(0, 1, num_points)
    curves = [np.zeros_like(grid_x)]
    shares = shares if shares is not None else np.ones((num_curves, ))
    shares = np.array(shares) / np.sum(shares)
    for i in range(num_curves):
        delta_h = shares[i]
        epsilon = perturbation_share * delta_h

        # make each curve in unit-terms then scale it to cube-shape
        curves.append(curves[-1] + delta_h * np.ones_like(grid_x)
                      + np.random.uniform(low=-epsilon, high=epsilon, size=(num_points, )))

    # make an array of interpolations
    funcs = []
    for curve in curves:
        funcs.append(interp1d(grid_x, curve, kind=kind))

    # compute in integers
    results = []
    for func in funcs:
        results.append((func(np.arange(shape[0]) / shape[0]) * shape[1]).astype(np.int))

    return results

def make_surfaces_v1(num_surfaces, grid_shape, shape=(100, 100, 400),
                     kind='cubic', perturbation_share=0.25, shares=None):
    """ First param is num_surfaces while the second controls how volatile is the curve. Might wanna
    try changing the parameter with curve.
    """
    # make an array of curve-supports
    grid_x = np.linspace(0, 1, grid_shape[0])
    grid_y = np.linspace(0, 1, grid_shape[1])

    # make the first curve
    curves = [np.zeros(grid_shape)]
    shares = shares if shares is not None else np.ones((num_surfaces, ))
    shares = np.array(shares) / np.sum(shares)
    for i in range(num_surfaces):
        delta_h = shares[i]
        epsilon = perturbation_share * delta_h

        # make each curve in unit-terms then scale it to cube-shape
        curves.append(curves[-1] + delta_h * np.ones_like(curves[0])
                      + np.random.uniform(low=-epsilon, high=epsilon, size=curves[0].shape))

    # make an array of interpolations
    funcs = []
    for curve in curves:
        funcs.append(interp2d(grid_x, grid_y, curve, kind=kind))

    # compute in integers
    results = []
    for func in funcs:
        results.append((func(np.arange(shape[1]) / shape[1],
                             np.arange(shape[0]) / shape[0]) * shape[2]).astype(np.int))

    return results


def reflectivity(v, rho):
    """ Compute reflectivity coeficcients given velocity and density models.
    Velocities and reflectivity coefficients can be either 2d or 3d.
    """
    rc = np.zeros_like(v)
    rc[..., 1:] = ((v[..., 1:] * rho[..., 1:] - v[..., :-1] * rho[..., :-1]) /
                   (v[..., 1:] * rho[..., 1:] + v[..., :-1] * rho[..., :-1]))
    return rc

def rickerwave(f, dt):
    """ Generate RickerWave given frequency and time-sample rate.
    """
    if not f < 0.2 * (1 / (2 * dt)):
        raise ValueError("Frequency too high for the dt chosen.")

    nw = 2.2 / (f * dt)
    nw = 2 * int(nw / 2) + 1
    nc = int(nw / 2)

    k = np.arange(1, nw + 1)
    alpha = (nc - k + 1) * f * dt * np.pi
    beta = alpha ** 2
    ricker = (1 - beta * 2) * np.exp(-beta)
    return ricker

def convolve_2d(rc, w):
    """ Generate synthetic seismic given reflectivity series and ricker wavelet.
    """
    synth_l = np.zeros_like(rc)
    for i in range(rc.shape[0]):
        if rc.shape[-1] >= len(w):
            synth_l[i, :] = np.convolve(rc[i, :], w, mode='same')
        else:
            aux = int(len(w) / 2.0)
            synth_l[i, :] = np.convolve(rc[i, :], w, mode='full')[aux:-aux]
    return synth_l

def convolve_3d(rc, w):
    """ Generate synthetic seismic given reflectivity series and ricker wavelet.
    """
    synth_l = np.zeros_like(rc)
    for i in range(rc.shape[0]):
        for x in range(rc.shape[1]):
            if rc.shape[-1] >= len(w):
                synth_l[i, x, :] = np.convolve(rc[i, x, :], w, mode='same')
            else:
                aux = int(len(w) / 2.0)
                synth_l[i, x, :] = np.convolve(rc[i, x, :], w, mode='full')[aux:-aux]
    return synth_l


def make_synthetic_3d(shape=(50, 400, 800), num_reflections=200, vel_limits=(900, 5400), horizon_heights=(1/4, 1/2, 2/3),
                      horizon_jumps=(7, 5, 4), grid_shape=(10, 10), perturbation_share=.2, rho_noise_lims=(0.97, 1.3),
                      ricker_rate=2/3e3, ricker_frequency=30, sigma=1.1, noise=0.5):
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
    ricker_rate : float
        Time-sampling rate of the ricker-wave.
    ricker_frequency : int
        Frequency of the ricker-wave.
    sigma : float or None
        sigma used for gaussian blur of the synthetic seismic.
    noise_mul : float or None
        If not None, gaussian noise scale by this number is applied to the synthetic. 
    """
    # generate array of velocities
    low, high = vel_limits
    llim = (high - low) / num_reflections
    velocities = (np.linspace(low, high, num_reflections) +
                  np.random.uniform(low=-llim, high=llim, size=(num_reflections, )))

    for height_share, jump_mul in zip(horizon_heights, horizon_jumps):
        colors[int(colors.shape[0] * height_share)] += llim * jump_mul

    # make velocity model
    curves = make_surfaces_v1(num_reflections, grid_shape, perturbation_share=perturbation_share, shape=shape)
    vel_model = np.zeros(shape=shape)
    for i in range(vel_model.shape[0]):
        for x in range(vel_model.shape[1]):
            trace = vel_model[i, x, :]
            for j in range(num_reflections):
                low = np.minimum(curves[j][i, x], vel_model.shape[-1])
                trace[low : ] = velocities[j]

    # make density model
    if rho_noise_lims is not None:
        rho = vel_model * np.random.uniform(*rho_noise_lims, size=vel_model.shape)
    else:
        rho = vel_model

    # obtain synthetic
    ref_coeffs = reflectivity(vel_model, rho)
    wavelet = rickerwave(ricker_frequency, ricker_rate)
    result = convolve_3d(ref_coeffs, wavelet)

    # add blur and noise if needed for a more realistic image
    if sigma is not None:
        result = gaussian_filter(result, sigma=sigma)
    if noise_mul is not None:
        result += noise_mul * np.random.random(result.shape) * result.std()
    return result


def make_synthetic_2d():
    # set up params of the synthetic:
    shape = 600, 2000 # [n_traces_i X n_samples]
    num_reflections = 200

    # generate array of velocities
    low, high = 0.3, 1.8
    llim = (high - low) / num_reflections
    colors = np.linspace(low, high, num_reflections) + np.random.uniform(low=-llim, high=llim, size=(num_reflections, ))
    colors[100:] += llim * 15
    colors[150:] += llim * 10
    colors[50:] -= llim*8

    num_points = 10
    curves = make_curves_v1(num_reflections, num_points, shape=shape, perturbation_share=.2)

    # make velocity model
    vm = np.zeros(shape=shape)
    for i in range(vm.shape[0]):
        trace = vm[i, :]
        for j in range(num_reflections):
            low = np.minimum(curves[j][i], vm.shape[-1])
            trace[low : ] = colors[j]
    mul = 3000
    vm = vm * mul

    # make density model
    l, h = 0.95, 1.05
    rho_mults = np.random.uniform(low=l, high=h, size=vm.shape)
    rho = vm * rho_mults

    # obtain synthetic
    dt = 0.002 / 3
    rc = reflectivity(vm, rho)
    w = rickerwave(30, dt)
    synt = convolve_2d(rc, w)

    # add blur and noise for a more realistic image
    blurred = gaussian_filter(synt, sigma=1.1)
    noised = blurred + .7* np.random.random(blurred.shape) * blurred.std()
