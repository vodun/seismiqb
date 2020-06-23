""" !!. """
#pylint: disable=import-error, no-name-in-module, wrong-import-position
import sys
from copy import copy
from scipy.ndimage import sobel
import numpy as np


sys.path.append('../seismiqb')
from seismiqb.batchflow import Pipeline, HistoSampler, NumpySampler
from seismiqb.batchflow import B, V, C, D, P, R
from seismiqb.batchflow.models.torch import EncoderDecoder

from seismiqb.src.cubeset import Horizon, HorizonMetrics

from .detector import Detector



class Extractor(Detector):
    """ !!. """
    #pylint: disable=unused-argument, logging-fstring-interpolation, no-member, protected-access
    #pylint: disable=access-member-before-definition, attribute-defined-outside-init
    def make_sampler(self, dataset, bins=50, side_view=False, **kwargs):
        _ = kwargs
        geometry = dataset.geometries[0]
        idx = np.nonzero(geometry.zero_traces != 1)
        points = np.hstack([idx[0].reshape(-1, 1),
                            idx[1].reshape(-1, 1)])

        geometry_sampler = HistoSampler(np.histogramdd(points/geometry.cube_shape[:2], bins=bins))
        geometry_sampler = geometry_sampler & NumpySampler('u', low=0., high=0.9)

        dataset.create_sampler(mode=geometry_sampler)
        dataset.modify_sampler('train_sampler', finish=True)

        dataset.show_slices(
            src_sampler='train_sampler', normalize=False, shape=self.crop_shape,
            adaptive_slices=False, side_view=side_view,
            cmap='Reds', interpolation='bilinear', show=self.show_plots, figsize=(15, 15),
            savepath=self.make_save_path(f'slices_{geometry.short_name}.png')
        )

        dataset.show_slices(
            src_sampler='train_sampler', normalize=True, shape=self.crop_shape,
            adaptive_slices=False, side_view=side_view,
            cmap='Reds', interpolation='bilinear', show=self.show_plots, figsize=(15, 15),
            savepath=self.make_save_path(f'slices_n_{geometry.short_name}.png')
        )



    def inference_1(self, dataset, heights_range=None, orientation='i', overlap_factor=2,
                    filter=True,
                    thresholds=None, coverage_threshold=0.5, std_threshold=5., metric_threshold=0.5,
                    chunk_size=100, chunk_overlap=0.2, minsize=10000, **kwargs):
        """ Split area for inference into `big` chunks, inference on each of them, merge results. """
        #pylint: disable=redefined-builtin, too-many-branches
        _ = kwargs
        thresholds = thresholds or np.arange(0.2, 1.0, 0.1)

        geometry = dataset.geometries[0]
        spatial_ranges, heights_range = self.make_inference_ranges(dataset, heights_range)
        config, crop_shape_grid, strides_grid = self.make_inference_config(orientation, overlap_factor)

        # Actual inference
        axis = np.argmin(crop_shape_grid[:2])
        iterator = range(spatial_ranges[axis][0], spatial_ranges[axis][1], int(chunk_size*(1 - chunk_overlap)))

        storage = [[]]*len(thresholds)
        for chunk in self.make_pbar(iterator, desc=f'Inference on {geometry.name}| {orientation}'):
            current_spatial_ranges = copy(spatial_ranges)
            current_spatial_ranges[axis] = [chunk, min(chunk + chunk_size, spatial_ranges[axis][-1])]

            dataset.make_grid(dataset.indices[0], crop_shape_grid,
                              *current_spatial_ranges, heights_range,
                              batch_size=self.batch_size,
                              strides=strides_grid)
            inference_pipeline = (self.get_inference_template() << config) << dataset
            for _ in range(dataset.grid_iters):
                batch = inference_pipeline.next_batch(D('size'))

            # Specific to Extractor:
            for sign in [-1, +1]:
                mask = sign * batch.assembled_pred

                for i, threshold in enumerate(thresholds):
                    chunk_horizons = Horizon.from_mask(mask, dataset.grid_info,
                                                       threshold=threshold, minsize=minsize)
                    storage[i].extend(chunk_horizons)

        merged_horizons = []
        for horizon_list in storage:
            merged = Horizon.merge_list(horizon_list, mean_threshold=2.5,
                                        adjacency=3, minsize=minsize)
            merged_horizons.extend(merged)
        merged_horizons = Horizon.merge_list(merged_horizons, mean_threshold=0.5)
        del storage

        filtered_horizons = []
        for horizon in merged_horizons:
            # CHECK 1: coverage
            if horizon.coverage >= coverage_threshold:

                # CHECK 2: std
                matrix = sobel(np.copy(horizon.matrix))
                matrix[horizon.matrix == Horizon.FILL_VALUE] = 0
                matrix[abs(matrix) > 100] = 0
                std_coeff = np.std(matrix)

                if std_coeff <= std_threshold:

                    # CHECK 3: metric
                    hm = HorizonMetrics(horizon)
                    corrs = hm.evaluate('support_corrs', supports=50, agg='nanmean')

                    if filter:
                        horizon.filter(filtering_matrix=(corrs <= metric_threshold).astype(np.int32))
                        if horizon.coverage <= coverage_threshold:
                            continue

                    corr_coeff = np.nanmean(corrs)

                    if corr_coeff >= metric_threshold:
                        horizon._corr_coeff = corr_coeff
                        filtered_horizons.append(horizon)
                        self.log(f'depth: {horizon.h_mean:6.6}; cov: {horizon.coverage:6.6};'
                                 f' std: {std_coeff:6.6}; metric: {corr_coeff:6.6}')
        del merged_horizons


        horizons = []
        for horizon in filtered_horizons:
            for i, already_stored in enumerate(horizons):

                if abs(horizon.h_mean - already_stored.h_mean) < 2.:
                    if horizon._corr_coeff > already_stored._corr_coeff:
                        _ = horizons.pop(i)
                        horizons.append(horizon)
                        break
                    break
            else:
                horizons.append(horizon)

        return horizons


    def get_train_template(self):
        """ Defines training procedure.

        Following parameters are fetched from pipeline config: `model_config`, `crop_shape`,
        `adaptive_slices` and `rebatch_threshold`.
        """
        train_template = (
            Pipeline()
            # Initialize pipeline variables and model
            .init_variable('loss_history', [])
            .init_model('dynamic', EncoderDecoder, 'model', C('model_config'))

            # Load data: images only!
            .crop(points=D('train_sampler')(self.batch_size),
                  shape=self.crop_shape, adaptive_slices=C('adaptive_slices'),
                  side_view=C('side_view', default=False))
            .load_cubes(dst='images')
            .adaptive_reshape(src='images', shape=self.crop_shape)
            .scale(mode='q', src='images')

            # Augmentations
            .transpose(src='images', order=(1, 2, 0))
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .flip(axis=1, src='images', seed=P(R('uniform', 0, 1)), p=0.3)
            .rotate(angle=P(R('uniform', -15, 15)),
                    src='images', p=0.3)
            .scale_2d(scale=P(R('uniform', 0.85, 1.15)),
                      src='images', p=0.3)
            .perspective_transform(alpha_persp=P(R('uniform', 25, 50)),
                                   src='images', p=0.3)
            .elastic_transform(alpha=P(R('uniform', 35, 45)),
                               sigma=P(R('uniform', 4, 4.5)),
                               src='images', p=0.2)
            .transpose(src='images', order=(2, 0, 1))

            # Training
            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('images'),
                         save_to=V('loss_history', mode='a'))
        )
        return train_template
