""" Helper function for models validation."""
import os

import numpy as np
from scipy.ndimage import find_objects, gaussian_filter
from scipy.ndimage.morphology import binary_dilation
from scipy.signal import find_peaks
from skimage.measure import label

from batchflow import Pipeline, B
from batchflow.models.torch import TorchModel

from .. import SeismicDataset, RegularGrid
from ..utils import Accumulator3D


def show_slide_inference(model_or_path, field, loc, axis=0, zoom='auto',
                         batch_size=64, crop_shape=(1, 256, 512),
                         inference_3d=True, inference_width=10,
                         minsize=5, dilation_iterations=1,
                         plot=False, savepath=None, **kwargs):
    """ Show model inference on a field slide.

    Parameters
    ----------
    model_or_path : str or instance of :class:`batchflow.TorchModel`
        Model to use or path to file from which initialize a model.
    field : instance of :class:`seismiqb.Field`
        Field from which to get a slide.
    loc : int
        Number of slide.
    axis : int
        Number of axis to load slide along.
    zoom : tuple of slices, None or 'auto'
        Tuple of slices to apply directly to 2d images.
        If None, slicing is not applied.
        If 'auto', zero traces on bounds will be dropped.
    batch_size : int
        Number of batches to generate for a slide.
        Affects inference speed and memory used.
    crop_shape : tuple of ints
        Shape of crop locations to generate for a slide.
        Affects inference speed and memory used.
    inference_3d : bool
        Whether to apply inference on orthogonal projection.
        If True, then result is smoothed depend on both projections inference.
    inference_width : int
        Amount of neighboring slides to infer. Affects result smoothing.
    minsize : int
        Objects with size less than minsize will be removed from prediction.
    dilation_iterations : int
        Number of dilation iterations to apply. Makes predictions more visible.
    savepath : str or None
        Path to a file or directory to save inferred slide (if provided).
    kwargs : dict
        Other parameters to pass to the plotting function.
    """
    # Prepare inference parameters
    ranges = [None, None, None]
    ranges[axis] = [loc-inference_width, loc+1+inference_width]

    grid = RegularGrid(field=field,
                       threshold=0, orientation=axis,
                       ranges=ranges,
                       batch_size=batch_size,
                       crop_shape=crop_shape, overlap_factor=2)

    if inference_3d:
        grid_other = RegularGrid(field=field,
                                 threshold=0, orientation=1-axis,
                                 ranges=ranges,
                                 batch_size=batch_size,
                                 crop_shape=crop_shape, overlap_factor=2)
        grid += grid_other

    accumulator = Accumulator3D.from_grid(grid=grid, aggregation='weighted', fill_value=0)

    # Inference
    model = TorchModel(model_or_path) if isinstance(model_or_path, str) else model_or_path

    inference_pipeline = (
        Pipeline()
        .make_locations(generator=grid, batch_size=batch_size)
        .load_cubes(dst='images')
        .normalize(src='images')

        .import_model(name='model', source=model)
        .predict_model('model', inputs=B('images'), outputs='sigmoid', save_to=B('predictions'))
        .update_accumulator(src='predictions', accumulator=accumulator)
    ) << SeismicDataset(field)

    inference_pipeline.run(n_iters=grid.n_iters, notifier='t', pbar=False)
    prediction = accumulator.aggregate()

    # Smoothing
    prediction = gaussian_filter(prediction, sigma=1.5)

    # Peaking
    prediction[prediction < 0.1] = 0.0
    peaked = np.zeros_like(prediction)

    for i in range(prediction.shape[0]):
        for x in range(prediction.shape[1]):
            trace = prediction[i, x, :]
            peaks, _ = find_peaks(trace, prominence=0.15, width=2)
            peaked[i, x, peaks] = 1
    prediction = np.take(peaked, indices=inference_width, axis=axis)

    # Filter small objects
    labeled = label(prediction, connectivity=2)
    objects = find_objects(labeled)

    for i, slc in enumerate(objects):
        indices = np.nonzero(labeled[slc] == i + 1)

        if len(indices[0]) <= minsize:
            coords = tuple(indices[i] + slc[i].start for i in range(2))
            prediction[coords] = 0

    # Modify slice
    prediction = (prediction > 0.1).astype(np.int32)
    prediction = binary_dilation(prediction, iterations=dilation_iterations).astype(np.float32)
    prediction[prediction < 0.1] = np.nan

    # Plotting
    if plot:
        if zoom is None:
            zoom = (slice(None), slice(None))

        if os.path.isdir(savepath):
            name = (f'{field.short_name}_axis_{axis}_loc_{loc}_inference_width_{inference_width}'
                    f'_inference_3d_{inference_3d}_zoom_{zoom}.png')
            savepath = os.path.join(savepath, name)
        if savepath is not None:
            kwargs['savepath'] = savepath

        plotter = field.show_slide(loc, axis=axis, show=False, zoom=zoom,
                                suptitle_size=4, suptitle_y=0.88,
                                title=None, suptitle=None,
                                **kwargs)
        plotter.plot(prediction.T)
        plotter.redraw()

        if savepath is not None:
            plotter.config.update(plotter[0][0].config)
            plotter.save()
    return prediction
