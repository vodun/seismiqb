""" A holder for horizon enhancement steps inherited from `.class:Detector` with the following functionality:
    - training a model on a horizon with synthetic distortions.
    - making inference on a selected data.
"""
from textwrap import indent
import numpy as np

from ...batchflow import Pipeline, B, V, C, P, R

from .horizon import HorizonController
from ...batchflow.models.torch import EncoderDecoder


class Enhancer(HorizonController):
    """
    Provides interface for train, inference and quality assesment for the task of horizon enhancement.
    """
    #pylint: disable=arguments-renamed

    def train(self, horizon, **kwargs):
        """ Train model for horizon enhancement. """
        dataset = self.make_dataset(horizon=horizon)
        sampler = self.make_sampler(dataset)
        sampler.show_locations(show=self.plot, savepath=self.make_savepath('sampler_locations.png'))
        sampler.show_sampled(show=self.plot, savepath=self.make_savepath('sampler_generated.png'))
        self.log(f'Created sampler\n{indent(str(sampler), " "*4)}')

        return super().train(dataset=dataset, sampler=sampler, **kwargs)


    def inference(self, horizon, model, config=None, **kwargs):
        """ Runs enhancement procedure for a given horizon with provided model. """
        dataset = self.make_dataset(horizon=horizon)

        #TODO: return filtering matrix to grid?
        # if filtering_matrix is None:
        #     filtering_matrix = 1 - (horizon.full_matrix > 0)
        prediction = super().inference(dataset=dataset, model=model, config=config, **kwargs)[0]
        prediction.name = f'enhanced_{horizon.name}'
        return prediction


    def load_pipeline(self):
        """ Defines data loading procedure.
        Following parameters are fetched from pipeline config: `width` and `rebatch_threshold`.
        """
        return (
            Pipeline()
            .make_locations(generator=C('sampler'), batch_size=C('batch_size'))
            .create_masks(dst='masks', width=C('width', default=3))
            .mask_rebatch(src='masks', threshold=C('rebatch_threshold', default=0.7))
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'])
            .normalize(src='images')
        )

    def distortion_pipeline(self):
        """ Defines transformations performed with `masks` component. """
        def binarize(batch):
            batch.prior_masks = (batch.prior_masks > 0).astype(np.float32)

        return (
            Pipeline()
            .shift_masks(src='masks', dst='prior_masks')
            .transpose(src='prior_masks', order=(1, 2, 0))
            .elastic_transform(alpha=P(R('uniform', 30, 50)), sigma=P(R('uniform', 6, 7)),
                               src='prior_masks', p=0.5)
            .bend_masks(src='prior_masks', angle=P(R('uniform', -15, 15)))
            .call(binarize, B())
            .transpose(src='prior_masks', order=(2, 0, 1))
        )

    def augmentation_pipeline(self):
        """ Defines augmentation pipeline. """
        return (
            Pipeline()
            .transpose(src=['images', 'masks', 'prior_masks'], order=(1, 2, 0))
            .rotate(angle=P(R('uniform', -30, 30)),
                    src=['images', 'masks', 'prior_masks'], p=0.3)
            .flip(src=['images', 'masks', 'prior_masks'], axis=1, p=0.3)
            .transpose(src=['images', 'masks', 'prior_masks'], order=(2, 0, 1))
        )

    def train_pipeline(self):
        """ Define model initialization and model training pipeline.
        Following parameters are fetched from pipeline config: `model_config`.
        """
        return (
            Pipeline()
            .init_variable('loss_history', default=[])
            .init_model(mode='dynamic', model_class=C('model_class', default=EncoderDecoder),
                        name='model', config=C('model_config'))
            .concat_components(src=['images', 'prior_masks'], dst='images', axis=1)
            .train_model('model', fetches='loss', save_to=V('loss_history', mode='a'),
                         images=B('images'),
                         masks=B('masks'))
        )

    def get_train_template(self, distortion_pipeline=None, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        _ = kwargs
        return (
            self.load_pipeline() +
            (distortion_pipeline or self.distortion_pipeline()) +
            self.augmentation_pipeline() +
            self.train_pipeline()
        )

    def get_inference_template(self):
        """ Defines inference pipeline. """
        inference_template = (
            Pipeline()
            .init_variable('predictions', [])

            # Load data
            .make_locations(generator=C('grid'))
            .load_cubes(dst='images')
            .create_masks(dst='prior_masks', width=C('width', default=3))
            .adaptive_reshape(src=['images', 'prior_masks'])
            .normalize(src='images')
            .concat_components(src=['images', 'prior_masks'], dst='images', axis=1)

            # Use model for prediction
            .predict_model('model',
                           B('images'),
                           fetches='predictions',
                           save_to=B('predictions'))
            .update_accumulator(src='predictions', accumulator=C('accumulator'))
        )
        return inference_template


    # One method to rule them all
    def run(self, cube_paths=None, horizon_paths=None, horizon=None, **kwargs):
        """ Run the entire procedure of horizon enhancement. """
        dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)
        horizon = dataset.labels[0][0]

        model = self.train(horizon=horizon, **kwargs)

        prediction = self.inference(horizon, model, **kwargs)
        prediction = self.postprocess(prediction)
        info = self.evaluate(prediction, dataset=dataset)
        return prediction, info
