import os
import sys

import numpy as np

from sklearn.linear_model import LinearRegression
from skimage.metrics import structural_similarity

sys.path.append('../../seismiqb')
from seismiqb import SyntheticGenerator, plot_image, plot_loss
from seismiqb.batchflow import NumpySampler as NS



def make_params():
    """ Parameter-generator example. Generates crops of fixed shape with random quantity of horizons.
    """
    SYNTHETIC_SHAPE = (128, 256)
    # Sampling procedures for params of synhetic generation
    num_horizons_sampler = NS('c', a=np.arange(0, 5)).apply(lambda x: x[0][0])

    samplers = {
        # Horizons
        'num_reflections': NS('c', a=np.arange(25, 50)).apply(lambda x: x[0][0]),
        'horizon_heights': NS('u', low=.05, high=.95).apply(lambda x: tuple(x.reshape(-1))),
        'horizon_multipliers': NS('c', a=list(range(-13, -7)) + list(range(12, 20))).apply(lambda x: tuple(x.reshape(-1))),

        # Faults

        # Impedance creation
        'grid_shape': NS('c', a=np.arange(5, 10)).apply(lambda x: (x[0][0], )),
        'density_noise_lims': (NS('u', low=0.8, high=1) & NS('u', low=1, high=1.2)).apply(lambda x: tuple(x.reshape(-1))),

        # Conversion to seismic
        'ricker_width': NS('u', low=3.3, high=5.5).apply(lambda x: x[0][0]),
        'ricker_points': NS('c', a=np.arange(50, 130)).apply(lambda x: x[0][0]),
        'noise_mul': NS('u', low=0.1, high=0.3).apply(lambda x: x[0][0]),
    }

    constants = {
        'shape': SYNTHETIC_SHAPE,
        'vel_limits': (5000, 11000)
    }

    # Making parameters-dict
    params = {}

    # Sampling
    num_horizons = num_horizons_sampler.sample(1)
    for name, sampler in samplers.items():
        if name in ('horizon_heights', 'horizon_multipliers'):
            params[name] = sampler.sample(num_horizons)
        else:
            params[name] = sampler.sample(1)

    # Taking Constants
    for name, value in constants.items():
        params[name] = value

    return params


def make_data(size, parameters_generator):
    synthetic_seismic = []
    impedance_models = []

    for _ in range(size):
        generator = SyntheticGenerator()
        params = parameters_generator()

        (generator.make_velocities(num_reflections=params['num_reflections'],
                                   horizon_heights=params['horizon_heights'],
                                   horizon_multipliers=params['horizon_multipliers'],
                                   vel_limits=params['vel_limits'])
                  .make_velocity_model(params['shape'], grid_shape=params['grid_shape'])
                  .make_density_model(params['density_noise_lims'])
                  .make_synthetic(ricker_width=params['ricker_width'], ricker_points=params['ricker_points'])
                  .postprocess_synthetic(noise_mul=params['noise_mul']))

        synthetic_seismic.append(generator.synthetic)
        impedance_models.append(generator.velocity_model)

    synthetic = np.array(synthetic_seismic)[:, np.newaxis, :, :]
    impedance = np.array(impedance_models)[:, np.newaxis, :, :]
    return (synthetic, impedance)


# Normalization methods
def normalize_seismic(array, function='mean-std', **kwargs):
    if callable(function):
        result = function(array)

    elif function=='mean-std': # item-wise
        mean = np.mean(array, axis=tuple(range(1, array.ndim)), keepdims=True)
        std = np.std(array, axis=tuple(range(1, array.ndim)), keepdims=True)
        result = (array - mean) / std
    return result


def normalize_impedance(array, function='mean-std', **kwargs):
    if callable(function):
        result = function(array)
    elif function=='mean-std': # item-wise
        mean = np.mean(array, axis=tuple(range(1, array.ndim)), keepdims=True)
        std = np.std(array, axis=tuple(range(1, array.ndim)), keepdims=True)
        result = (array - mean) / std
    elif function=='min-max': # item-wise
        min_ = np.min(array, axis=tuple(range(1, array.ndim)), keepdims=True)
        max_ = np.max(array, axis=tuple(range(1, array.ndim)), keepdims=True)
        result = (array - min_) / (max_ - min_)
    elif function=='global_mean-std': # use passed stats
        result = (array - kwargs['mean']) / kwargs['std']
    elif function=='global_min-max': # use passed stats
        result = (array - kwargs['min']) / (kwargs['max'] - kwargs['min'])
    return result

def denormalize_impedance(array, function='mean-std', **kwargs):
    if callable(function):
        result = function(array)
    elif function=='mean-std':
        result = array * kwargs['std'] + kwargs['mean']
    elif function=='min-max':
        result = array * (kwargs['max'] - kwargs['min']) + kwargs['min']

    return result


# Conversion to seismic
def impedance_to_seismic(impedance_predictions, density_noise_lims=None, ricker_width=4.3):
    # `impedance_predictions` should be de-normalized
    synthetic = []
    for impedance in impedance_predictions:
        generator = SyntheticGenerator()

        generator.velocity_model = impedance.squeeze()
        (generator.make_density_model(density_noise_lims=density_noise_lims)
                  .make_synthetic(ricker_width=ricker_width))
        synthetic.append(generator.synthetic)
    return np.stack(synthetic).astype(np.float32)

def adjust_seismic(prediction, real):
    regressor = LinearRegression()
    regressor.fit(X=prediction.reshape(-1, 1),
                  y=real.reshape(-1, 1))
    a, b = regressor.coef_, regressor.intercept_
    return a * prediction + b



# Metric computation
def ssim(im1, im2, **kwargs):
    return structural_similarity(im1, im2)


class ComputeMetric:
    def __init__(self):
        self.metrics_fixed = [0.0]
        self.metrics_random = [0.0]

    def __call__(self, model, batch=None, pipeline=None, fetches='predictions',
                 seismic_normalize=True, impedance_denormalize=True, ricker_width=4.3,
                 iteration=0, trigger=True, show=True, savedir=None):
        if trigger is False:
            return (self.metrics_fixed if batch is not None else self.metrics_random)[-1]

        # Choose correct batch
        flag = batch is not None
        batch = batch.__copy__() if flag else pipeline.next_batch()
        if seismic_normalize:
            seismic_normalize = seismic_normalize if isinstance(seismic_normalize, dict) else {'function': 'mean-std'}
            batch.images = normalize_seismic(batch.images, **seismic_normalize)

        # Make predictions and seismic
        seismic_images = batch.images
        impedance_predictions = model.predict(images=seismic_images, fetches=fetches)

        if impedance_denormalize:
            denormalized_impedance = denormalize_impedance(impedance_predictions, **impedance_denormalize)
        else:
            denormalized_impedance = impedance_predictions
        seismic_predictions = impedance_to_seismic(denormalized_impedance, ricker_width=ricker_width)

        # Compute metrics
        ssims = []
        for seismic_image, seismic_prediction in zip(seismic_images, seismic_predictions):
            adjusted_seismic = adjust_seismic(seismic_prediction, real=seismic_image)
            ssim_coefficient = ssim(seismic_image.squeeze(), adjusted_seismic)
            ssims.append(ssim_coefficient)

        metric = np.mean(ssim_coefficient)

        (self.metrics_fixed if flag else self.metrics_random).append(metric)
        return metric
compute_metric = ComputeMetric()



def show_results_real(model, batch=None, pipeline=None, fetches='predictions', n=4,
                      seismic_normalize=True, impedance_denormalize=True, ricker_width=4.3,
                      iteration=0, trigger=True, show=True, savedir=None):
    if trigger is False:
        return

    # Choose correct batch
    flag = batch is not None
    batch = batch.__copy__() if flag else pipeline.next_batch()
    if seismic_normalize:
        seismic_normalize = seismic_normalize if isinstance(seismic_normalize, dict) else {'function': 'mean-std'}
        batch.images = normalize_seismic(batch.images, **seismic_normalize)

    # Make predictions and seismic
    seismic_images = batch.images
    impedance_predictions = model.predict(images=seismic_images, fetches=fetches)

    if impedance_denormalize:
        denormalized_impedance = denormalize_impedance(impedance_predictions, **impedance_denormalize)
    else:
        denormalized_impedance = impedance_predictions
    seismic_predictions = impedance_to_seismic(denormalized_impedance, ricker_width=ricker_width)

    # Plots: synthetic, true and predicted impedance
    thousand_ctr = iteration // 1000
    savepath = f'{savedir}/{thousand_ctr * 1000}_{(thousand_ctr + 1) * 1000}'
    os.makedirs(f'{savepath}', exist_ok=True)

    # Create lists for plotting
    images_data = []
    titles = []
    for idx in range(n):
        field_name = batch.unsalt(batch.indices[idx])
        adjusted_seismic = adjust_seismic(seismic_predictions[idx].squeeze(), real=seismic_images[idx].squeeze())
        ssim_coefficient = ssim(seismic_images[idx].squeeze(),
                                adjusted_seismic)

        images_data.extend([seismic_images[idx].squeeze(),
                            denormalized_impedance[idx].squeeze(),
                            adjusted_seismic])
        titles.extend([f'Crop from {field_name}',
                       'Predicted impedance\ndenormalized',
                       f'Recreated seismic\nadjusted {ssim_coefficient:4.2f}'])

    # Plot parameters
    plot_kwargs = {
        'figsize': (20, 10*n),
        'separate': True,
        'colorbar': True, 'cmap': 'gray',
        'title': titles,
        'vmin': None, 'vmax': None,
        'nrows': n, 'ncols': 3,
        'show': show,
        'savepath' : f'{savepath}/IMAGES_{"FIXED_" if flag else "RANDOM_"}VALIDATION_{iteration}.png' if savedir else None
    }

    plot_image(images_data, **plot_kwargs)
    return


def show_result_synthetic(model, batch, n=4, fetches='predictions',
                          iteration=0, trigger=True, show=True, savedir=None):
    if trigger is False:
        return

    # Make predictions and seismic
    synthetic_images = batch.images
    synthetic_impedance = batch.impedance
    impedance_predictions = model.predict(images=synthetic_images, fetches=fetches)


    # Plots: synthetic, true and predicted impedance
    thousand_ctr = iteration // 1000
    savepath = f'{savedir}/{thousand_ctr * 1000}_{(thousand_ctr + 1) * 1000}'
    os.makedirs(f'{savepath}', exist_ok=True)


    # Create lists for plotting
    images_data = []
    titles = []
    for idx in np.random.choice(a=len(batch), size=n, replace=False):

        images_data.extend([synthetic_images[idx].squeeze(),
                            synthetic_impedance[idx].squeeze(),
                            impedance_predictions[idx].squeeze()])
        titles.extend(['Synthetic',
                       'True impedance',
                       'Predicted impedance'])
    # Plot parameters
    plot_kwargs = {
        'figsize': (20, 10*n),
        'separate': True,
        'colorbar': True, 'cmap': 'gray',
        'title': titles,
        'vmin': None, 'vmax': None,
        'nrows': n, 'ncols': 3,
        'show': show,
        'savepath' : f'{savepath}/IMAGES_TRAIN_{iteration}.png' if savedir else None,
    }

    plot_image(images_data, **plot_kwargs)
    return


def show_progress(metrics, model, trigger=True, iteration=0, show=True, savedir=None):

    if trigger is False:
        return

    # Plots: synthetic, true and predicted impedance
    thousand_ctr = iteration // 1000
    savepath = f'{savedir}/{thousand_ctr * 1000}_{(thousand_ctr + 1) * 1000}'
    os.makedirs(f'{savepath}', exist_ok=True)

    plot_loss(metrics,
              title=f'SSIM over batch',
              rolling_mean=True, final_mean=False, show=show,
              savepath=f'{savepath}/SSIM_VALIDATION_{iteration}.png' if savedir else None)

    plot_loss(model.loss_list, show=show, savepath=f'{savepath}/MODEL_LOSS_{iteration}.png' if savedir else None)
    plot_loss(model.loss_list, show=show, savepath=f'{savedir}/MODEL_LOSS.png' if savedir else None)
    return

