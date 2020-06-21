""" A holder for horizon enhancement steps inherited from `.class:Detector` with the following functionality:
    - training a model on a horizon area with a given percentage of holes.
    - making inference of a Horizon Extension algorithm to cover the holes in a given horizon.
"""
import numpy as np


from ...batchflow import Pipeline, FilesIndex
from ...batchflow import B, V, C, D, P, R

from ..cubeset import SeismicCubeset

from .torch_models import ExtensionModel, MODEL_CONFIG
from .detector import Detector

class Enhancer(Detector):
    """
    Provides interface for train, inference and quality assesment for the task of horizon enhancement.

    Parameters
    ----------
    batch_size : int
        Size of batches for train and inference.
    crop_shape : tuple of 3 ints
        Size of sampled crops for train and inference.
    model_config : dict
        Neural network architecture.
    model_path : str
        Path for pre-trained model.
    device : str or int
        Device specification.
    show_plots : bool
        Whether to show plots to the current output stream.
    save_dir : str
        Path to save images, logs, and other data.
    logger : None or callable
        If None, then logger is created inside `save_dir`.
        If callable, then it is used directly to log messages.
    bar : bool
        Whether to show progress bars for training and inference.
    """
    #pylint: disable=unused-argument, logging-fstring-interpolation, no-member

    # Dataset creation: geometries, labels, grids, samplers
    def _make_dataset(self, horizon):
        """ Create an instance of :class:`.SeismicCubeset` for a given horizon.

        Parameters
        ----------
        horizon : instance of :class:`.Horizon`
            Horizon for the inferred cube.

        Logs
        ----
        Inferred cube.

        Returns
        -------
        Instance of dataset.
        """
        cube_path = horizon.geometry.path

        dsi = FilesIndex(path=[cube_path], no_ext=True)
        dataset = SeismicCubeset(dsi)
        dataset.load_geometries()
        dataset.labels[dataset.indices[0]] = [horizon]

        self.log(f'Created dataset from horizon {horizon.name}')
        return dataset


    def train(self, horizon, **kwargs):
        """ Train model for horizon detection.
        If `model_path` was supplied during instance initialization, model is loaded instead.

        In order to change architecture of the model, pass different `model_config` to the instance initialization.
        In order to change training procedure, re-define :meth:`.get_train_template`.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be enhanced
        mask_transform : an instance of :class:`batchflow.Pipeline`
            Pipeline with pre-defined transformations performed with `masks` component.
            If None, default filtering mask pipeline is used.
            Default is defined in :meth:`.get_train_template`.

        Logs
        ----
        Start of training; end of training; average loss at the last 50 iterations.

        Plots
        -----
        Graph of loss over iterations.
        """
        dataset = self._make_dataset(horizon)
        self.make_sampler(dataset, use_grid=False, bins=np.array([500, 500, 100]))
        super().train(dataset, **kwargs)


    def get_train_template(self):
        """ Define training pipeline.

        Following parameters are fetched from pipeline config: `model_config` and `crop_shape`.
        """
        load = (
            Pipeline()
            .crop(points=D('train_sampler')(self.batch_size),
                  shape=self.crop_shape, side_view=True)
            .create_masks(dst='masks', width=3)
            .mask_rebatch(src='masks', threshold=0.99, passdown=None)
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'],
                              shape=self.crop_shape)
            .scale(mode='q', src='images')
        )

        def binarize(batch):
            batch.prior_masks = (batch.prior_masks > 0).astype(int).astype(np.float32)


        distort = (
            Pipeline()
            .shift_masks(src='masks', dst='prior_masks')
            .transpose(src='prior_masks', order=(1, 2, 0))
            .elastic_transform(alpha=P(R('uniform', 30, 50)), sigma=P(R('uniform', 6, 7)),
                               src='prior_masks')
            .call(binarize)
            .transpose(src='prior_masks', order=(2, 0, 1))
        )

        augment = (
            Pipeline()
            .transpose(src=['images', 'masks', 'prior_masks'], order=(1, 2, 0))
            .rotate(angle=P(R('uniform', -30, 30)),
                    src=['images', 'masks', 'prior_masks'], p=0.3)
            .flip(src=['images', 'masks', 'prior_masks'], axis=1, p=0.3)
            .transpose(src=['images', 'masks', 'prior_masks'], order=(2, 0, 1))
        )

        train = (
            Pipeline()
            .init_variable('loss_history', default=[])
            .init_model('dynamic', ExtensionModel, 'base', C('model_config'))
            .train_model('base', fetches='loss', save_to=V('loss_history', mode='a'),
                         images=B('images'),
                         prior_masks=B('prior_masks'),
                         masks=B('masks'))
        )
        return load + distort + augment + train


    def get_inference_template(self):
        """ Define inference pipeline.
        """
        inference_template = (
            Pipeline()
            # Init everything
            .init_variable('predicted_masks', default=list())
            .import_model('base', C('model_pipeline'))
            # Load data
            .crop(points=D('grid_gen')(), shape=self.crop_shape,
                  side_view=C('side_view', default=False))
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=3)
            .adaptive_reshape(src=['images', 'prior_masks'],
                              shape=self.crop_shape)
            .scale(mode='q', src='images')
            # Use model for prediction
            .predict_model('base',
                           B('images'),
                           B('prior_masks'),
                           fetches='predictions',
                           save_to=V('predicted_masks', mode='e'))
            .assemble_crops(src=V('predicted_masks'), dst='assembled_pred',
                            grid_info=D('grid_info'), order=C('order', default=(0, 1, 2)))
        )
        return inference_template





    @staticmethod
    def enhance(horizon, n_steps=1, cube_path=None, model_config=None, crop_shape=(1, 64, 64),
                batch_size=128, save_dir='.', device=None, stride=16, n_iters=400):
        """ !!. """
        model_config = MODEL_CONFIG if model_config is None else model_config
        enhancer = Enhancer(save_dir=save_dir, model_config=model_config, device=device,
                            crop_shape=crop_shape, batch_size=batch_size)
        dataset = enhancer._make_dataset(horizon)
        enhancer.make_sampler(dataset, use_grid=False, bins=np.array([500, 500, 100]))
        enhancer.train(dataset, n_iters=n_iters, use_grid=False)
        enhanced = enhancer.inference(dataset, n_steps=n_steps, horizon=horizon,
                                      crop_shape=crop_shape, batch_size=batch_size, stride=stride)
        return enhanced
