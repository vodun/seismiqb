""" A convenient holder for horizon detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
    - evaluating predictions
    - and more
"""
#pylint: disable=import-error, no-name-in-module, wrong-import-position, protected-access
import os
import gc
import logging

from time import perf_counter
from ast import literal_eval
from textwrap import indent
from pprint import pformat
from copy import copy
from glob import glob

import psutil

import numpy as np
import torch

from ...batchflow import Config, Pipeline, Monitor, Notifier
from ...batchflow import B, D, C, V, P, R
from ...batchflow.models.torch import EncoderDecoder

from ..cubeset import SeismicCubeset, Horizon
from ..metrics import HorizonMetrics
from ..plotters import plot_loss, plot_image



class BaseController:
    """ !!. """
    #pylint: disable=attribute-defined-outside-init
    DEFAULTS = Config({
        'savedir': None,
        'monitor': True,
        'bar': False,
        'plot': False,
        'logger': None,
        'train': {
            'model_class': EncoderDecoder,
            'model_config': None,

            'batch_size': None,
            'crop_shape': None,

            'rebatch_threshold': 0.8,
            'rescale_batch_size': True,

            'prefetch': 1,
            'n_iters': 200,
            'early_stopping': True,
        },
        'inference': {
            'orientation': 'ix',
            'batch_size': None,
            'crop_shape': None,

            # Grid making parameters
            'spatial_ranges': None,
            'heights_range': None,
            'overlap_factor': 2,
            'filtering_matrix': None,
            'filter_threshold': 0.,

            'chunk_size': 100,
            'chunk_overlap': 0.1,
        },
        'evaluate': {
            'supports': 100,
            'device': 'gpu',

            'add_prefix': False,
            'dump': False,
            'name': '',
        }
    })

    def __init__(self, config):
        self.config = Config(copy(self.DEFAULTS))
        self.config += config

        self.monitor = self.config.monitor
        self.plot = self.config.plot

        gpu_list = literal_eval(os.getenv('CUDA_VISIBLE_DEVICES'))
        self.gpu_list = list(gpu_list) if isinstance(gpu_list, tuple) else [gpu_list]
        self.make_filelogger()

    # Utility functions
    def make_savepath(self, *postfix):
        """ Create nested path from provided strings.
        Uses `savedir` config option.

        If `savedir` config option is None, then None is returned: that is used as signal to omit saving
        of, for example, metric map images, etc.
        """
        savedir = self.config['savedir']
        if savedir is not None:
            path = os.path.join(savedir, *postfix[:-1])
            os.makedirs(path, exist_ok=True)
            return os.path.join(savedir, *postfix)
        return None

    # Logging
    def make_filelogger(self):
        """ Create logger inside `savedir`.

        Note that logging is important.
        """
        log_path = self.make_savepath('controller.log')
        if log_path:
            handler = logging.FileHandler(log_path, mode='w')
            handler.setFormatter(logging.Formatter('%(asctime)s      %(message)s'))

            logger = logging.getLogger(str(id(self)))
            logger.addHandler(handler)
            self.filelogger = logger.error
        else:
            self.filelogger = None

    def log(self, msg):
        """ Log supplied message into both filelogger and supplied one. """
        process = psutil.Process(os.getpid())
        uss = process.memory_full_info().uss / (1024 ** 3)
        msg = f'{self.__class__.__name__} ::: {uss:2.4f} ::: {msg}'

        logger = self.config.logger
        if logger:
            logger(msg)
        if self.filelogger:
            self.filelogger(msg)

    def log_to_file(self, msg, path):
        """ Log message to a separate file. """
        log_path = self.make_savepath(path)
        if log_path:
            with open(log_path, 'w') as file:
                print(msg, file=file)

    # Dataset creation: geometries, labels, grids, samplers
    def make_dataset(self, cube_paths=None, horizon_paths=None, horizon=None):
        """ Create an instance of :class:`.SeismicCubeset` with cubes and horizons.

        Parameters
        ----------
        cube_paths : sequence or str
            Cube path(s) to load into dataset.
        horizon_paths : dict or str
            Horizons for each cube. Either a mapping from cube name to paths, or path only (if only one cube is used).
        horizon : None or Horizon
            If supplied, then the dataset is initialized with a single horizon.

        Logs
        ----
        Inferred cubes and horizons for them.

        Returns
        -------
        Instance of dataset.
        """
        if horizon is not None:
            dataset = SeismicCubeset.from_horizon(horizon)
        else:
            dataset = SeismicCubeset(cube_paths)
            dataset.load_geometries()

            if horizon_paths:
                if isinstance(horizon_paths, str):
                    horizon_paths = {dataset.indices[0]: glob(horizon_paths)}
                dataset.create_labels(horizon_paths)

        self.log(f'Created dataset\n{indent(str(dataset), " "*4)}')
        return dataset

    def make_grid(self, dataset, frequencies, **kwargs):
        """ Create a grid, based on quality map, for each of the cubes in supplied `dataset`.
        Works inplace.

        Parameters
        ----------
        dataset : :class:`.SeismicCubeset`
            Dataset with cubes.
        frequencies : sequence of ints
            List of frequencies, corresponding to `easy` and `hard` places in the cube.
        kwargs : dict
            Other arguments, passed directly in quality grid creation function.

        Logs
        ----
        Grid coverage: ratio of the number of points inside the grid to the total number of non-bad traces in cube.

        Plots
        -----
        Map with quality grid.
        """
        for idx in dataset.indices:
            geometry = dataset.geometries[idx]
            geometry.make_quality_grid(frequencies, **kwargs)

            postfix = f'_{idx}' if len(dataset.indices) > 1 else ''
            plot_image(
                geometry.quality_grid, title=f'Quality grid for {idx}',
                show=self.plot, cmap='Reds', interpolation='bilinear',
                savepath=self.make_savepath(f'quality_grid{postfix}.png')
            )

            grid_coverage = (np.nansum(geometry.quality_grid) /
                             (np.prod(geometry.cube_shape[:2]) - np.nansum(geometry.zero_traces)))
            self.log(f'Created grid on {idx}; coverage is: {grid_coverage:4.4f}')

    def make_sampler(self, dataset, bins=None, use_grid=False, grid_src='quality_grid', side_view=False, **kwargs):
        """ Create sampler. Works inplace.

        Plots
        -----
        Maps with examples of sampled slices of `crop_shape` size, both normalized and not.
        """
        dataset.create_sampler(quality_grid=use_grid, bins=bins)
        dataset.modify_sampler('train_sampler', finish=True, **kwargs)
        self.log('Created sampler')

        # Cleanup
        dataset.sampler = None
        dataset.samplers = None

        crop_shape = self.config['train']['crop_shape']
        for i, idx in enumerate(dataset.indices):
            postfix = f'_{idx}' if len(dataset.indices) > 1 else ''
            dataset.show_slices(
                src_sampler='train_sampler', idx=i, normalize=False,
                shape=crop_shape, side_view=side_view,
                adaptive_slices=use_grid, grid_src=grid_src,
                show=self.plot, cmap='Reds', interpolation='bilinear', figsize=(15, 15),
                savepath=self.make_savepath(f'sampled{postfix}.png')
            )

            dataset.show_slices(
                src_sampler='train_sampler', idx=i, normalize=True,
                shape=crop_shape, side_view=side_view,
                adaptive_slices=use_grid, grid_src=grid_src,
                show=self.plot, cmap='Reds', interpolation='bilinear', figsize=(15, 15),
                savepath=self.make_savepath(f'sampled_normalized{postfix}.png')
            )

    # Train
    def train(self, dataset, **kwargs):
        """ Train model.

        In order to change architecture of the model, pass different `model_config` to the instance initialization.
        In order to change training procedure, re-define :meth:`.get_train_template`.

        Logs
        ----
        Start of training; end of training; average loss at the last 25 iterations.

        Plots
        -----
        Graph of loss over iterations.
        """
        # Prepare parameters
        pipeline_config = Config({**self.config['train'], **kwargs})
        n_iters, prefetch, rescale = pipeline_config.pop(['n_iters', 'prefetch', 'rescale_batch_size'])

        notifier = {
            'bar': 'n' if self.config.bar else False,
            'monitors': 'loss_history',
            'file': self.make_savepath('末 model_loss.log'),
        }
        self.log(f'Train started on device={self.gpu_list}')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.05, gpu_list=self.gpu_list)
            monitor.__enter__()

        # Make pipeline
        train_pipeline = self.get_train_template(**kwargs) << pipeline_config << dataset

        # Log: pipeline_config to a file
        self.log_to_file(pformat(pipeline_config.config, depth=2), '末 train_config.txt')

        # Test batch to initialize model and log stats
        batch = train_pipeline.next_batch(D.size)
        model = train_pipeline.m('model')

        self.log(f'Target batch size: {pipeline_config["batch_size"]}')
        self.log(f'Actual batch size: {len(batch)}')
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')

        # Log: full model repr
        self.log_to_file(repr(model.model), '末 model_repr.txt')

        # Log: short model repr
        model.model.apply(lambda module: setattr(module, 'short_repr', True))
        msg = repr(model.model)
        model.model.apply(lambda module: setattr(module, 'short_repr', False))
        self.log_to_file(msg, '末 model_shortrepr.txt')

        # Rescale batch size, if needed
        if rescale:
            scale = pipeline_config['batch_size'] / len(batch)
            pipeline_config['batch_size'] = int(pipeline_config['batch_size'] * scale)
            self.log(f'Rescaling batch size to: {pipeline_config["batch_size"]}')

        train_pipeline.set_config(pipeline_config)

        # Run training procedure
        start_time = perf_counter()
        self.log(f'Train run: n_iters={n_iters}, prefetch={prefetch}')
        train_pipeline.run(D.size, n_iters=n_iters, prefetch=prefetch, notifier=notifier)
        elapsed = perf_counter() - start_time

        # Log: resource graphs
        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 train_resource.png'), show=self.plot)

        # Log: loss over iteration
        plot_loss(model.loss_list, show=self.plot,
                  savepath=self.make_savepath('末 model_loss.png'))
        finish_loss = np.mean(model.loss_list[-25:])

        # Log: model train information
        self.log_to_file(model._information(config=True, devices=True, model=False, misc=True), '末 model_info.txt')

        # Log: stats
        self.log(f'Trained for {model.iteration} iterations in {elapsed:4.1f}s')
        self.log(f'Average of 25 last loss values: {finish_loss:4.3f}')
        self.log(f'Cache sizes: {[item.cache_size for item in dataset.geometries.values()]}')
        self.log(f'Cache lengths: {[item.cache_length for item in dataset.geometries.values()]}')

        # Cleanup
        torch.cuda.empty_cache()
        train_pipeline.reset('variables')
        for item in dataset.geometries.values():
            item.reset_cache()
        self.log('')

        self.train_log = {
            'start_time': start_time,
            'elapsed': elapsed,
        }
        return model

    # Inference
    def inference(self, dataset, model, **kwargs):
        """ !!. """
        # Prepare parameters
        config = Config({**self.config['inference'], **kwargs})
        orientation = config.pop('orientation')
        self.log(f'Starting {orientation} inference')

        # Log: pipeline_config to a file
        self.log_to_file(pformat(config.config, depth=2), '末 inference_config.txt')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.05, gpu_list=self.gpu_list)
            monitor.__enter__()

        horizons = []

        start_time = perf_counter()
        for letter in orientation:
            horizons_ = self._inference(dataset=dataset, model=model,
                                        orientation=letter, config=config)
            self.log(f'Done {letter}-inference')
            horizons.extend(horizons_)
        elapsed = perf_counter() - start_time

        horizons = Horizon.merge_list(horizons, minsize=1000)
        self.log(f'Inference done in {elapsed:4.1f}')

        # Log: resource graphs
        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 inference_resource.png'), show=self.plot)

        # Log: lengths of predictions
        if horizons:
            horizons.sort(key=len, reverse=True)
            self.log(f'Num of predicted horizons: {len(horizons)}')
            self.log(f'Total number of points in all of the horizons {sum(len(item) for item in horizons)}')
            self.log(f'Len max: {len(horizons[0])}')
        else:
            self.log('Zero horizons were predicted; possible problems..?')

        self.inference_log = {
            'elapsed': elapsed,
        }
        return horizons

    def _inference(self, dataset, model, orientation, config):
        # Prepare parameters
        geometry = dataset.geometries[0]
        spatial_ranges, heights_range = config.get(['spatial_ranges', 'heights_range'])
        chunk_size, chunk_overlap = config.get(['chunk_size', 'chunk_overlap'])

        # Make spatial and height ranges
        if spatial_ranges is None:
            spatial_ranges = [[0, item - 1] for item in geometry.cube_shape[:2]]

        if heights_range is None:
            bases = dataset.labels[0]
            if bases:
                min_height = min(horizon.h_min for horizon in bases) - config.crop_shape[2]//2
                max_height = max(horizon.h_max for horizon in bases) + config.crop_shape[2]//2
                heights_range = [max(0, min_height), min(geometry.depth, max_height)]
            else:
                heights_range = [0, geometry.depth]

        # Update config according to the orientation
        if orientation == 'i':
            axis = 0
            config.update({
                'crop_shape_grid': config.crop_shape,
                'side_view': 0.0,
                'order': (0, 1, 2),
            })
        elif orientation == 'x':
            axis = 1
            config.update({
                'crop_shape_grid': np.array(config.crop_shape)[[1, 0, 2]],
                'side_view': 1.0,
                'order': (1, 0, 2),
            })

        # Make iterator of chunks over given axis
        chunk_iterator = range(spatial_ranges[axis][0],
                               spatial_ranges[axis][1],
                               int(chunk_size*(1 - chunk_overlap)))

        # Actual inference
        self.log(f'Starting {orientation}-inference with {len(chunk_iterator)} chunks')
        self.log(f'Inference over {spatial_ranges}, {heights_range}')
        notifier = Notifier('n' if self.config.bar else False, desc=orientation, update=False,
                            file=self.make_savepath(f'末 inference_chunks_{orientation}.log'))
        chunk_iterator = notifier(chunk_iterator)

        horizons = []
        for chunk in chunk_iterator:
            chunk_spatial_ranges = copy(spatial_ranges)
            chunk_spatial_ranges[axis] = [chunk, min(chunk + chunk_size, spatial_ranges[axis][-1])]

            horizons_ = self._inference_on_chunk(dataset=dataset, model=model,
                                                 ranges=(*chunk_spatial_ranges, heights_range),
                                                 config=config)
            horizons.extend(horizons_)

        # Cleanup
        for item in dataset.geometries.values():
            item.reset_cache()
        gc.collect()
        torch.cuda.empty_cache()

        return horizons

    def _inference_on_chunk(self, dataset, model, ranges, config):
        overlap_factor, filtering_matrix, filter_threshold = config.get(['overlap_factor',
                                                                         'filtering_matrix',
                                                                         'filter_threshold'])
        # _ = config.pop('prefetch', None) # TODO: prefetch?

        # Create grid over chunk ranges
        dataset.make_grid(dataset.indices[0],
                          np.array(config.crop_shape)[list(config.order)],
                          *ranges,
                          batch_size=config.batch_size,
                          overlap_factor=overlap_factor,
                          filtering_matrix=filtering_matrix,
                          filter_threshold=filter_threshold)

        # Create pipeline TODO: make better `add_model`
        inference_pipeline = self.get_inference_template() << config << dataset
        inference_pipeline.models.add_model('model', model)

        # Make predictions over chunk
        predictions = []
        for _ in range(dataset.grid_iters):
            batch = inference_pipeline.next_batch(D.size)
            predictions.extend(item for item in batch.predictions)

        # Assemble prediction together in accordance to the created grid
        assembled_prediction = dataset.assemble_crops(predictions, order=config.order)

        # Extract Horizon instances
        horizons = Horizon.from_mask(assembled_prediction, dataset.grid_info,
                                     threshold=0.5, minsize=50)

        # Cleanup
        gc.collect()
        return horizons

    # Evaluate
    def evaluate(self, predictions, targets=None, dataset=None, **kwargs):
        """ !!. """
        #pylint: disable=cell-var-from-loop
        config = Config({**self.config['evaluate'], **kwargs})
        add_prefix, dump, name = config.pop(['add_prefix', 'dump', 'name'])
        supports, device = config.pop(['supports', 'device'])


        if targets is None and dataset is not None:
            targets = dataset.labels[0]

        predictions = predictions if isinstance(predictions, (tuple, list)) else [predictions]
        n = len(predictions)

        results = []
        for i, horizon in enumerate(predictions):
            info = {}
            prefix = [horizon.geometry.short_name, f'{i}_horizon'] if add_prefix else []

            # Basic demo: depth map and properties
            horizon.show(show=self.plot, savepath=self.make_savepath(*prefix, name + 'depth_map.png'))

            with open(self.make_savepath(*prefix, name + 'results_self.txt'), 'w') as result_txt:
                horizon.evaluate(compute_metric=False, printer=lambda msg: print(msg, file=result_txt))

            hm = HorizonMetrics((horizon, targets))

            # Metric maps
            corrs = hm.evaluate('support_corrs', supports=supports, device=device,
                                plot=True, show=self.plot,
                                savepath=self.make_savepath(*prefix, name + 'corrs.png'))

            phase = hm.evaluate('instantaneous_phase', device=device,
                                plot=True, show=self.plot,
                                savepath=self.make_savepath(*prefix, name + 'instantaneous_phase.png'))

            perturbed_mean, perturbed_max = hm.evaluate('perturbed', device=device,
                                                        plot=True, show=self.plot,
                                                        savepath=self.make_savepath(*prefix, name + 'perturbed.png'))

            # Compare to targets
            if targets:
                _, oinfo = hm.evaluate('find_best_match', agg=None)
                info = {**info, **oinfo}

                with open(self.make_savepath(*prefix, name + 'results.txt'), 'w') as result_txt:
                    hm.evaluate('compare', hist=False,
                                plot=True, show=self.plot,
                                printer=lambda msg: print(msg, file=result_txt),
                                savepath=self.make_savepath(*prefix, name + 'l1.png'))
                msg = (f'Predicted horizon {i} compared to target:'
                       f'\nwindow_rate={info["window_rate"]:4.3f}\navg error={info["mean"]:4.3f}')
                self.log(indent(msg, ' '*59))

            # Save surface to disk
            if dump:
                dump_name = name + '_' if name else ''
                dump_name += f'{i}_' if n > 1 else ''
                dump_name += horizon.name or 'predicted'
                horizon.dump(path=self.make_savepath(*prefix, dump_name), add_height=False)

            info['corrs'] = np.nanmean(corrs)
            info['phase'] = np.nanmean(np.abs(phase))
            info['perturbed_mean'] = np.nanmean(perturbed_mean)
            info['perturbed_max'] = np.nanmean(perturbed_max)
            results.append((info))

            msg = (f'Predicted horizon {i}:\nlen={len(horizon)}\ncoverage={horizon.coverage:4.3f}'
                   f'\ncorrs={info["corrs"]:4.3f}\nphase={info["phase"]:4.3f}\n avg depth={horizon.h_mean:4.3f}')
            self.log(indent(msg, ' '*59))
        return results

    # Pipelines
    def load_pipeline(self, dynamic_factor=1, dynamic_low=None, dynamic_high=None, **kwargs):
        """ Define data loading pipeline.

        Following parameters are fetched from pipeline config: `adaptive_slices`, 'grid_src' and `rebatch_threshold`.
        """
        _ = kwargs
        self.log(f'Generating data with dynamic factor of {dynamic_factor}')
        return (
            Pipeline()
            .init_variable('shape', None)
            .call(generate_shape, shape=C('crop_shape'),
                  dynamic_factor=dynamic_factor, dynamic_low=dynamic_low, dynamic_high=dynamic_high,
                  save_to=V('shape'))
            .make_locations(points=D('train_sampler')(C('batch_size')),
                            shape=V('shape'),
                            side_view=C('side_view', default=False),
                            adaptive_slices=C('adaptive_slices'),
                            grid_src=C('grid_src', default='quality_grid'))

            .create_masks(dst='masks', width=C('width', default=3))
            .mask_rebatch(src='masks', threshold=C('rebatch_threshold', default=0.1))
            .load_cubes(dst='images')
            .adaptive_reshape(src=['images', 'masks'], shape=V('shape'))
            .normalize(mode='q', src='images')
        )

    def augmentation_pipeline(self, **kwargs):
        """ Define augmentation pipeline. """
        _ = kwargs
        return (
            Pipeline()
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], seed=P(R('uniform', 0, 1)), p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -15, 15)),
                    src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', 0.85, 1.15)),
                      src=['images', 'masks'], p=0.3)
            .elastic_transform(alpha=P(R('uniform', 35, 45)), sigma=P(R('uniform', 4, 4.5)),
                               src=['images', 'masks'], p=0.2)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))
        )

    def train_pipeline(self, **kwargs):
        """ Define model initialization and model training pipeline.

        Following parameters are fetched from pipeline config: `model_config`.
        """
        _ = kwargs
        return (
            Pipeline()
            .init_variable('loss_history', [])
            .init_model('dynamic', C('model_class', default=EncoderDecoder), 'model', C('model_config'))

            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         microbatch=C('microbatch', default=True),
                         save_to=V('loss_history', mode='a'))
        )

    def get_train_template(self, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        return (
            self.load_pipeline(**kwargs) +
            self.augmentation_pipeline(**kwargs) +
            self.train_pipeline(**kwargs)
        )


    def get_inference_template(self):
        """ Defines inference procedure.

        Following parameters are fetched from pipeline config: `model_pipeline`, `crop_shape`, `side_view` and `order`.
        """
        inference_template = (
            Pipeline()
            # Load data
            .make_locations(points=D('grid_gen')(), shape=C('crop_shape'),
                            side_view=C('side_view'))
            .load_cubes(dst='images')
            .adaptive_reshape(src='images', shape=C('crop_shape'))
            .normalize(mode='q', src='images')

            # Predict with model, then aggregate
            .predict_model('model',
                           B('images'),
                           fetches='predictions',
                           save_to=B('predictions'))
        )
        return inference_template



def generate_shape(_, shape, dynamic_factor=1, dynamic_low=None, dynamic_high=None):
    """ Dynamically generate shape of a crop to get. """
    dynamic_low = dynamic_low or dynamic_factor
    dynamic_high = dynamic_high or dynamic_factor

    i, x, h = shape
    x_ = np.random.randint(x // dynamic_low, x * dynamic_high + 1)
    h_ = np.random.randint(h // dynamic_low, h * dynamic_high + 1)
    return (i, x_, h_)
