""" A holder for horizon extension steps inherited from `.class:Enhancer` with:
    - redifined get_mask_transform_ppl to filter out a true mask to train an extension model
      on a horizon area with a given percentage of holes.
    - making inference of a Horizon Extension algorithm to cover the holes in a given horizon.
"""
import numpy as np


from ...batchflow import Pipeline, FilesIndex
from ...batchflow import B, V, C, D, P, R, L

from ..cubeset import SeismicCubeset, Horizon

from .torch_models import ExtensionModel, MODEL_CONFIG
from .enhancer import Enhancer


class Extender(Enhancer):
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

    def get_mask_transform_ppl(self):
        """ Define transformations performed with `masks` component.
        """
        def functor(scale):
            return lambda m: np.sin(m[:, 0] * scale)

        filter_out = (
            Pipeline()
            .transpose(src='masks', order=(1, 2, 0))
            .filter_out(src='masks', dst='prior_masks',
                        expr=lambda m: m[:, 0],
                        low=P(R('uniform', low=0.2, high=0.4)),
                        length=P(R('uniform', low=0.30, high=0.5)))
            .filter_out(src='masks', dst='prior_masks',
                        expr=lambda m: m[:, 0],
                        low=P(R('uniform', low=0.1, high=0.4)),
                        length=P(R('uniform', low=0.10, high=0.4)), p=0.5)
            .filter_out(src='masks', dst='prior_masks',
                        expr=L(functor)(R('uniform', low=15, high=35)), low=0.0, p=0.7)
            .transpose(src=['masks', 'prior_masks'], order=(2, 0, 1))
        )
        return filter_out

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
            .scale(mode='q', src='images')
            # Use model for prediction
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
    def run(horizon, n_steps=1, model_config=None, n_iters=400, crop_shape=(1, 64, 64),
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
