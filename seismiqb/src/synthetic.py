import numpy as np
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage import gaussian_filter


def make_curves_v1(n_curves, n_points, shape=(100, 100),
                   kind='cubic', epsilon_share=0.4, shares=None):
    """ First param is n_colors the second controls how volatile is the curve. Might wanna
    try changing the parameter with curve.
    """
    # make an array of curve-supports
    grid_x = np.linspace(0, 1, n_points)
    curves = [np.zeros_like(grid_x)]
    shares = shares if shares is not None else np.ones((n_curves, ))
    shares = np.array(shares) / np.sum(shares)
    for i in range(n_curves):
        delta_h = shares[i]
        epsilon = epsilon_share * delta_h

        # make each curve in unit-terms then scale it to cube-shape
        curves.append(curves[-1] + delta_h * np.ones_like(grid_x)
                      + np.random.uniform(low=-epsilon, high=epsilon, size=(n_points, )))

    # make an array of interpolations
    funcs = []
    for curve in curves:
        funcs.append(interp1d(grid_x, curve, kind=kind))

    # compute in integers
    results = []
    for func in funcs:
        results.append((func(np.arange(shape[0]) / shape[0]) * shape[1]).astype(np.int))

    return results

def make_surfaces_v1(n_surfaces, n_axis_points, shape=(100, 100, 400),
                     kind='cubic', epsilon_share=0.25, shares=None):
    """ First param is n_surfaces while the second controls how volatile is the curve. Might wanna
    try changing the parameter with curve.
    """
    # make an array of curve-supports
    grid_x = np.linspace(0, 1, n_axis_points)
    grid_y = np.linspace(0, 1, n_axis_points)

    #make the first curve
    curves = [np.zeros((n_axis_points, n_axis_points))]
    shares = shares if shares is not None else np.ones((n_surfaces, ))
    shares = np.array(shares) / np.sum(shares)
    for i in range(n_surfaces):
        delta_h = shares[i]
        epsilon = epsilon_share * delta_h

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

##############
# 3d "pipeline"
##############

# set up params of the synthetic:
shape = (50, 400, 800) # [n_traces_i n_traces_x X n_samples]
n_colors = 200

# generate array of velocities
low, high = 0.3, 1.8
llim = (high - low) / n_colors
colors = np.linspace(low, high, n_colors) + np.random.uniform(low=-llim, high=llim, size=(n_colors, ))
colors[colors.shape[0] // 2:] += llim * 7
colors[2 * colors.shape[0] // 3:] += llim * 5
colors[colors.shape[0] // 4:] -= llim*4

# make velocity model
n_points = 10
curves = make_surfaces_v1(n_colors, n_points, epsilon_share=.2, shape=shape)

vm = np.zeros(shape=shape)
for i in range(vm.shape[0]):
    for x in range(vm.shape[1]):
        trace = vm[i, x, :]
        for j in range(n_colors):
            low = np.minimum(curves[j][i, x], vm.shape[-1])
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
synt = convolve_3d(rc, w)

# add blur and noise for a more realistic image
blurred = gaussian_filter(synt, sigma=1.1)
noised = blurred + .7 * np.random.random(blurred.shape) * blurred.std()


##############
# 2d "pipeline"
##############

# set up params of the synthetic:
shape = 600, 2000 # [n_traces_i X n_samples]
n_colors = 200

# generate array of velocities
low, high = 0.3, 1.8
llim = (high - low) / n_colors
colors = np.linspace(low, high, n_colors) + np.random.uniform(low=-llim, high=llim, size=(n_colors, ))
colors[100:] += llim * 15
colors[150:] += llim * 10
colors[50:] -= llim*8

n_points = 10
curves = make_curves_v1(n_colors, n_points, shape=shape, epsilon_share=.2)

# make velocity model
vm = np.zeros(shape=shape)
for i in range(vm.shape[0]):
    trace = vm[i, :]
    for j in range(n_colors):
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
