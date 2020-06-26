""" A holder for horizon enhancement steps inherited from `.class:Detector` with the following functionality:
    - training a model on a horizon with synthetic distortions.
    - making inference on a selected data.
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
        """ Train model for horizon extension.
        Creates dataset and sampler for a given horizon and calls `meth:Detector.train`.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be enhanced
        kwargs : see documentation of `.class:Detector`

        Note
        ----
        In order to change training procedure, re-define :meth:`.get_train_template`.

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

    def inference(self, horizon, **kwargs):
        """ Runs enhancement procedure for a given horizon with trained/loaded model.
        Creates dataset for a given horizon and calls `meth:Detector.inference.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be enhanced
        kwargs : see documentation of `.meth:Detector.train`
        """
        dataset = self._make_dataset(horizon)
        super().inference(dataset, **kwargs)

    def get_load_ppl(self):
        """ Define data loading pipeline.
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
        return load

    def get_mask_transform_ppl(self):
        """ Define transformations performed with `masks` component.
        """
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
        return distort

    def get_augmentation_ppl(self):
        """ Define augmentation pipeline.
        """
        augment = (
            Pipeline()
            .transpose(src=['images', 'masks', 'prior_masks'], order=(1, 2, 0))
            .rotate(angle=P(R('uniform', -30, 30)),
                    src=['images', 'masks', 'prior_masks'], p=0.3)
            .flip(src=['images', 'masks', 'prior_masks'], axis=1, p=0.3)
            .transpose(src=['images', 'masks', 'prior_masks'], order=(2, 0, 1))
        )
        return augment

    def get_train_model_ppl(self):
        """ Define model initialization and model training pipeline.
    
        Note
        ----
        Following parameters are fetched from pipeline config: `model_config` 
        """
        train = (
            Pipeline()
            .init_variable('loss_history', default=[])
            .init_model('dynamic', ExtensionModel, 'base', C('model_config'))
            .train_model('base', fetches='loss', save_to=V('loss_history', mode='a'),
                         images=B('images'),
                         prior_masks=B('prior_masks'),
                         masks=B('masks'))
        )
        return train

    def get_train_template(self):
        """ Define whole training procedure pipeline including data loading,
        mask transforms, augmentation and model training.
        """
        train = self.get_load_ppl() + self.get_mask_transform_ppl() + self.get_augmentation_ppl() + \
                self.get_train_model_ppl()
        return train

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
    def run(horizon, n_steps=1, model_config=None, n_iters=400, crop_shape=(1, 64, 64),
                batch_size=128, orientation='ix', device=None, save_dir='.', cube_path=None):
        """ Run all steps of the Enhancement procedure including creating dataset for
        the given horizon, creating instance of the class and running train and inference
        methods.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be extended
        n_steps : int
            Number of steps of the Extension algorithm.
        model_config : dict
            Neural network architecture.
        n_iters : int
            Number of iterations to train model for.
        crop_shape : tuple of 3 ints
            Size of sampled crops for train and inference.
        batch_size : int
            Size of batches for train and inference.
        orientation : {'i', 'x', 'ix'}
            Orientation of the inference:
            If 'i', then cube is split into inline-oriented slices.
            If 'x', then cube is split into crossline-oriented slices.
            If 'ix', then both of previous approaches applied, and results are merged.
        device : str or int
            Device specification.
        save_dir : str
            Path to save images, logs, and other data.

        Returns
        -------
        Extended horizon.
        """
        model_config = MODEL_CONFIG if model_config is None else model_config
        enhancer = Enhancer(save_dir=save_dir, model_config=model_config, device=device,
                            crop_shape=crop_shape, batch_size=batch_size)
        if isinstance(horizon, str):
            if not cube_path:
                raise ValueError('Cube path must be provided along with a path to horizon')
            dataset = self.make_dataset(cube_path, horizon_paths=horizon)
        else:
            dataset = enhancer._make_dataset(horizon)
        enhancer.make_sampler(dataset, use_grid=False, bins=np.array([500, 500, 100]))
        enhancer.train(horizon, n_iters=n_iters, use_grid=False)
        enhanced = enhancer.inference(horizon, n_steps=n_steps, orientation=orientation)
        return enhanced
