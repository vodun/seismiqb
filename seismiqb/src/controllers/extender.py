""" A holder for horizon extension steps inherited from `.class:Detector` with the following functionality:
    - training a model on a horizon area with a given percentage of holes.
    - making inference of a Horizon Extension algorithm to cover the holes in a given horizon.
"""
import numpy as np


from ...batchflow import Pipeline, FilesIndex
from ...batchflow import B, V, C, D, P, R, L

from ..cubeset import SeismicCubeset, Horizon

from .torch_models import ExtensionModel, MODEL_CONFIG
from .detector import Detector


class Extender(Detector):
    """
    Provides interface for train, inference and quality assesment for the task of horizon extension.

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
            A horizon to be extended
        kwargs : see documentation of `.meth:Detector.train`

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
            .rotate_axes(src=['images', 'masks'])
            .scale(mode='q', src='images')
        )
        return load

    def get_mask_transform_ppl(self):
        """ Define transformations performed with `masks` component.
        """
        def functor(scale):
            return lambda m: np.sin(m[:, 0] * scale)

        filter_out = (
            Pipeline()
            .filter_out(src='masks', dst='prior_masks',
                        expr=lambda m: m[:, 0],
                        low=P(R('uniform', low=0.2, high=0.4)),
                        length=P(R('uniform', low=0.30, high=0.5))) +
            Pipeline()
            .filter_out(src='masks', dst='prior_masks',
                        expr=lambda m: m[:, 0],
                        low=P(R('uniform', low=0.1, high=0.4)),
                        length=P(R('uniform', low=0.10, high=0.4))) @ 0.5 +
            Pipeline()
            .filter_out(src='masks', dst='prior_masks',
                        expr=L(functor)(R('uniform', low=15, high=35)), low=0.0) @ 0.7
        )
        return filter_out

    def get_augmentation_ppl(self):
        """ Define augmentation pipeline.
        """
        augment = (
            Pipeline()
            .rotate(angle=P(R('uniform', -30, 30)),
                    src=['images', 'masks', 'prior_masks'], p=0.3)
            .flip(src=['images', 'masks', 'prior_masks'], axis=1, p=0.3)
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
            .transpose(src=['images', 'masks', 'prior_masks'], order=(2, 0, 1))
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

    def inference(self, horizon, n_steps=30, batch_size=128, stride=16):
        """Extend, i.e. fill the holes of the given horizon with the
        Horizon Extension algorithm using loaded/trained model.
        For each step of the Extension algorithm crops close to the horizon boundaries
        are generated so they contain a part of the known horizon that
        will be used as a prior mask to the ExtensionModel model.
        Predicted horizons for every crop will be merged to the known horizon on each step.

        Parameters
        ----------
        horizon : an instance of :class:`.Horizon`
            A horizon to be extended
        n_steps : int
            Number of steps of the Extension algorithm.
        batch_size : int
            Size of batches for train and inference.
        stride : int
            Distance between a horizon border and a corner of sampled crop that is fed to the model.

        Logs
        ----
        Start of inference, increased horizon legth at each step of the procedure.

        Returns
        -------
        Extended horizon
        """
        dataset = self._make_dataset(horizon)

        config = {
            'batch_size': batch_size,
            'model_pipeline': self.model_pipeline
        }
        prev_len = len(horizon)

        self.log('Extender inference started')
        for _ in self.make_pbar(range(n_steps), desc=f'Extender inference on {horizon.name}'):
            dataset.make_extension_grid(dataset.indices[0],
                                        crop_shape=self.crop_shape,
                                        stride=stride,
                                        labels_src=horizon,
                                        batch_size=batch_size)

            # Add current horizon to dataset labels in order to make create_masks work
            dataset.labels[dataset.indices[0]] = [horizon]

            inference_pipeline = (self.get_inference_template() << config) << dataset
            try:
                for _ in range(dataset.grid_iters):
                    inference_pipeline.next_batch(1, n_epochs=None)
            except TypeError:
                # no additional predicts
                break

            
            horizons = [*inference_pipeline.v('predicted_horizons')]
            for hor in horizons:
                merge_code, _ = Horizon.verify_merge(horizon, hor,
                                                    mean_threshold=5.5,
                                                    adjacency=5)
                if merge_code == 3:
                    _ = horizon.overlap_merge(hor, inplace=True)

            curr_len = len(horizon)
            if (curr_len - prev_len) < 25:
                break
            self.log(f'Extended from {prev_len} to {curr_len}, + {curr_len - prev_len}')
            prev_len = curr_len
        self.predictions = [horizon]
        return horizon

    def get_inference_template(self):
        """ Define inference pipeline.
        """
        inference_template = (
            Pipeline()
            # Init everything
            .init_variable('predicted_horizons', default=list())
            .import_model('base', C('model_pipeline'))
            # Load data
            .crop(points=L(D('grid_gen')), shape=L(D('shapes_gen')))
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=3)
            .adaptive_reshape(src=['images', 'prior_masks'],
                              shape=self.crop_shape)
            .rotate_axes(src=['images', 'prior_masks'])
            .scale(mode='q', src='images')
            # Use model for prediction
            .transpose(src=['images', 'prior_masks'], order=(2, 0, 1))
            .concat_components(src=('images', 'prior_masks'), dst='model_inputs')
            .predict_model('base',
                           B('images'),
                           B('prior_masks'),
                           fetches='predictions',
                           save_to=B('predicted_masks', mode='w'))
            .transpose(src='predicted_masks', order=(1, 2, 0))
            .masks_to_horizons(src='predicted_masks', threshold=0.5, minsize=16,
                               order=L(D('orders_gen')), dst='horizons', skip_merge=True)
            .update(V('predicted_horizons', mode='e'), B('horizons'))
        )
        return inference_template

    @staticmethod
    def extend(horizon, n_steps=1, model_config=None, n_iters=400, crop_shape=(1, 64, 64),
                batch_size=128, device=None, stride=16, save_dir='.'):
        """ Run all steps of the Extension procedure including creating dataset for
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
        device : str or int
            Device specification.
        stride : int
            Distance between a horizon border and a corner of sampled crop that is fed to the model.
        save_dir : str
            Path to save images, logs, and other data.

        Returns
        -------
        Extended horizon.
        """
        model_config = MODEL_CONFIG if model_config is None else model_config
        extender = Extender(save_dir=save_dir, model_config=model_config, device=device,
                            crop_shape=crop_shape, batch_size=batch_size)
        extender.train(horizon, n_iters=n_iters, use_grid=False)
        extended = extender.inference(horizon, n_steps=n_steps,
                                      batch_size=batch_size, stride=stride)
        return extended
