""" A convenient holder for faults detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
"""
import os
from datetime import datetime
from shutil import copyfile
import h5py
import tqdm

import numpy as np
from pprint import pformat

from .base import BaseController
from .best_practices import MODEL_CONFIG

from ..crop_batch import SeismicCropBatch
from ..geometry import SeismicGeometry
from ..labels import Fault
from ..cubeset import SeismicCubeset
from ..samplers import SeismicSampler, RegularGrid, FaultSampler, ConstantSampler
from ..metrics import FaultsMetrics
from ..utils import adjust_shape_3d, Accumulator3D, InputLayer
from ..plotters import plot_image

from ...batchflow import Config, Pipeline, Notifier, Dataset, Monitor
from ...batchflow import B, C, D, P, R, V, F, I, M
from ...batchflow import NumpySampler as NS
from ...batchflow.models.torch import TorchModel, EncoderDecoder



class FaultController(BaseController):
    """ Controller for faults detection tasks. """
    DEFAULTS = Config({
        **BaseController.DEFAULTS,
        # Data
        'dataset': {
            'path': '/cubes/',
            'train_cubes': [],
            'transposed_cubes': [],
            'inference_cubes': dict(),
            'label_dir': '/INPUTS/FAULTS/NPY_WIDTH_{}/*',
            'width': 3,
            'ext': 'qblosc',
            'weights': None,
            'threshold': 0,
            'uniform_cubes': True,
            'uniform_faults': True,
            'extend_annotation': True,
        },

        # Model parameters
        'train': {
            # Training parameters
            'batch_size': 1024,
            'crop_shape': [1, 128, 512],
            'n_iters': 2000,
            'callback/each': 100,
            'visualize_crops': True,

            # Augmentation parameters
            'angle': 25,
            'scale': (0.7, 1.5, 1),
            'augment': True,

            # Normalization parameters
            'norm_mode': 'minmax',
            'itemwise': False,
            'normalization_layer': False,
            'normalization_window': (1, 1, 100),

            # Model parameters
            'phase': True,
            'continuous_phase': False,
            'filters': [64, 96, 128, 192, 256],
            'model': 'UNet',
            'loss': 'bce',
            'output': 'sigmoid',
            'slicing': 'native',
            'prefetch': 0,
            'rescale_batch_size': False,
        },

        'inference': {
            'batch_size': 32,
            'crop_shape': [1, 128, 512],
            'inference_batch_size': 32,
            'smooth_borders': False,
            'stride': 0.5,
            'orientation': 'ilines',
            'slicing': 'native',
            'output': 'sigmoid',
            'norm_mode': 'minmax',
            'itemwise': False,
            'aggregation': 'mean'
        }
    })

    # Model (not controller) config
    BASE_MODEL_CONFIG = {
        'optimizer': {'name': 'Adam', 'lr': 0.01},
        "decay": {'name': 'exp', 'gamma': 0.9, 'frequency': 100, 'last_iter': 2000},
        'microbatch': C('microbatch', default=True),
        'initial_block': {
            'enable': C('phase'),
            'filters': C('filters')[0] // 2,
            'kernel_size': 5,
            'downsample': False,
            'attention': 'scse',
            'phases': C('phase'),
            'continuous': C('continuous_phase'),
            'window': C('normalization_window'),
            'normalization': C('normalization_layer')
        },
        'loss': C('loss')
    }

    UNET_CONFIG = {
        **MODEL_CONFIG,
        'initial_block/base_block': InputLayer,
        'body/encoder/blocks/filters': C('filters')[:-1],
        'body/embedding/filters': C('filters')[-1],
        'body/decoder/upsample/kernel_size': 5,
        'body/decoder/blocks/filters': C('filters')[-2::-1],
        'common/activation': 'relu6',
    }

    def make_train_dataset(self, **kwargs):
        """ Create train dataset as an instance of :class:`.SeismicCubeset` with cubes and faults.
        Arguments are from 'dataset' subconfig and kwargs. """
        return self._make_dataset(train=True, **kwargs)

    def make_inference_dataset(self, labels=False, **kwargs):
        """ Create inference dataset as an instance of :class:`.SeismicCubeset` with cubes and faults.
        Arguments are from 'dataset' subconfig and kwargs. """
        return self._make_dataset(train=False, labels=labels, **kwargs)

    def _make_dataset(self, train=True, labels=True, bar=False, **kwargs):
        """ Create an instance of :class:`.SeismicCubeset` with cubes and faults. Arguments are from 'dataset'
        subconfig and kwargs. """
        config = {**self.config['dataset'], **kwargs}
        width = config['width']
        label_dir = config['label_dir']
        cubes = config['train_cubes'] if train else config['inference_cubes']

        paths = [self.amplitudes_path(item) for item in cubes]
        dataset = SeismicCubeset(index=paths)

        if labels:
            transposed = config['transposed_cubes']
            direction = {f'{cube}': 0 if cube not in transposed else 1 for cube in cubes}
            dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True,
                         direction=direction, verify=True, bar=bar)
        else:
            dataset.load_geometries(logs=False)

        return dataset

    def make_sampler(self, dataset, **kwargs):
        """ Create sampler for generating locations to train on. """
        config = {**self.config['dataset'], **kwargs}
        weights = config.get('weights')
        threshold = config['threshold']
        crop_shape = self.config['train']['crop_shape']
        extend = self.config['dataset']['extend_annotation']

        uniform_cubes = config['uniform_cubes']
        uniform_faults = config['uniform_faults']

        if self.config['train/augment'] and self.config['train/adjust']:
            crop_shape = self.adjust_shape(crop_shape)
            self.log(f'Adjusted crop shape: {crop_shape}.')

        if weights is None:
            if len(dataset) > 0:
                if uniform_cubes:
                    weights = [1 for _ in dataset.labels.values()]
                else:
                    weights = [len(labels) for labels in dataset.labels.values()]
            else:
                weights = [1]
        weights = np.array(weights)
        weights = weights / weights.sum()

        self.log(f'Train dataset cubes weights: {weights}.')

        sampler = SeismicSampler(labels=dataset.labels, crop_shape=crop_shape, mode='fault',
                                 threshold=threshold, extend=extend, **kwargs)

        new_sampler = 0 & ConstantSampler(np.int32(0), dim=FaultSampler.dim)
        for cube_weight, cube_sampler in zip(weights, sampler.samplers.values()):
            new_cube_sampler = 0 & ConstantSampler(np.int32(0), dim=FaultSampler.dim)
            if uniform_faults:
                fault_weights = np.array([len(sampler.locations) for sampler in cube_sampler])
                fault_weights = fault_weights / fault_weights.sum()
            else:
                fault_weights = np.array([1 / len(cube_sampler) for _ in cube_sampler])

            for fault_weight, fault_sampler in zip(fault_weights, cube_sampler):
                new_cube_sampler = new_cube_sampler | (cube_weight * fault_weight & fault_sampler)

            new_sampler = new_sampler | new_cube_sampler
            sampler.sampler = new_sampler

        return sampler

    def load_train(self):
        """ Create loading pipeline for train stages. """
        return (Pipeline()
            .make_locations(batch_size=C('batch_size'), generator=C('sampler'))
            .load_cubes(dst='images', slicing=C('slicing'))
            .create_masks(dst='masks')
            .adaptive_reshape(src=['images', 'masks'])
            .normalize(mode=C('norm_mode'), itemwise=C('itemwise'), src='images')
        )

    def load_test(self, create_masks=True):
        """ Create loading pipeline for inference stages. """
        ppl = (Pipeline()
            .make_locations(batch_size=C('batch_size'), generator=C('sampler'))
            .load_cubes(dst='images', slicing=C('slicing'))
            .adaptive_reshape(src='images')
            .normalize(mode=C('norm_mode'), itemwise=C('itemwise'), src='images')
        )
        if create_masks:
            ppl += (Pipeline()
                .create_masks(dst='masks')
                .adaptive_reshape(src='masks')
            )
        return ppl

    def make_synthetic_dataset(self):
        ds = Dataset(1, SeismicCropBatch)
        # ds = Dataset(range(self.config['train/batch_size']), SeismicCropBatch)
        ds.geometries = {i: SeismicGeometry('/cubes/CUBE_01_ETP/amplitudes_01_ETP.qblosc') for i in ds.indices}
        return ds

    def load_synthetic(self, **kwargs):
        refs = NS('c', a=np.arange(50, 70)).apply(lambda x: x[0, 0])
        heights = (NS('u', dim=2) * [0.2, 0.2] + [0.25, 0.55]).apply(lambda x: np.squeeze(x))
        muls = NS('c', a=np.arange(-8, -4)).apply(lambda x: x[:, 0])

        il, xl, h = self.config['train/crop_shape']

        def faults():
            x0 = np.random.randint(14, xl-10)
            x1 = np.random.randint(14, xl-10) # same prmtzn for x1 as for x0; you can change it ofc
            y0 =  np.random.randint(0, 200)
            y1 = np.random.randint(300, h)
            return (
                ((x0, y0), (x1, y1)),
            )

        nhors = NS('c', a=[0, 1, 2]).apply(lambda x: x[0, 0])
        locations = [slice(0, i) for i in self.config['train/crop_shape']]
        return (Pipeline()
                .generate_synthetic(shape=self.config['train/crop_shape'],
                                    grid_shape=(10,),
                                    num_reflections=F(refs.sample)(size=1),
                                    horizon_heights=F(heights.sample)(size=1),
                                    horizon_multipliers=F(muls.sample)(size=F(nhors.sample)(size=1)),
                                    faults=F(faults),
                                    dst=('images', 'horizons', 'masks'),
                                    zeros_share_faults=0.1,
                                    mul=10, # controls max fault-shift
                                    fault_width=5 # default value is 3
                                    )
                .rebatch(C('batch_size'), components=['images', 'masks'])
                .update(B('locations'), [locations] * C('batch_size'))
                .normalize(mode=C('norm_mode'), itemwise=C('itemwise'), src='images')
                .run_later(batch_size=1, shuffle=False, n_iters=None, n_epochs=None, drop_last=False)
        )

    def augmentation_pipeline(self):
        """ Augmentations for training. """
        return (Pipeline()
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], seed=P(R('uniform', 0, 1)), p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -C('angle'), C('angle'))), src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', C('scale')[0], C('scale')[1])), src=['images', 'masks'], p=0.3)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))
            .central_crop(C('crop_shape'), src=['images', 'masks'])
            .cutout_2d(src=['images'], patch_shape=np.array((1, 40, 40)), n=3, p=0.2)
        )

    def train_pipeline(self):
        """ Create train pipeline. """
        model_class = F(self.get_model_class)(C('model'))
        model_config = F(self.get_model_config)(C('model'))
        ppl = (Pipeline()
            .init_variable('loss_history', [])
            .init_model(name='model', model_class=model_class, mode='dynamic', config=model_config)
            .adaptive_expand(src=['images', 'masks'])
            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         save_to=V('loss_history', mode='a'))
            .call(self.visualize_predictions, train_pipeline=B().pipeline,
                  savepath='prediction', each=self.config['train/callback/each'],
                  iteration=I())
        )
        if self.config['train/visualize_crops']:
            ppl += (Pipeline()
                .predict_model('model', images=B('images')[:1], fetches=C('output'), save_to=B('prediction'))
            )
            if self.config['train/phase']:
                ppl += Pipeline().update(B('phases'), M('model').get_intermediate_activations(
                    images=B('images'),
                    layers=M('model').model[0][0].phase_layer)
                )
            if self.config['train/normalization_layer']:
                ppl += Pipeline().update(B('norm_images'), M('model').get_intermediate_activations(
                    images=B('images'),
                    layers=M('model').model[0][0].normalization_layer)
                )
        return ppl

    def train(self, dataset, sampler=None, synthetic=False, config=None, **kwargs):
        if synthetic:
            config = config or {}
            pipeline_config = Config({**self.config['common'], **self.config['train'], **config, **kwargs})
            n_iters, batch_size = pipeline_config.pop(['n_iters', 'batch_size'])
            if n_iters:
                n_iters = n_iters * (batch_size or 1)
            pipeline_config['n_iters'] = n_iters
            config = pipeline_config

        return super().train(dataset, sampler=sampler, config=config, synthetic=synthetic, **kwargs)


    def make_notifier(self):
        """ Make notifier. """
        if self.config['train/visualize_crops']:
            cube_name = ""#B().unsalt(B('indices')[0])
            src = [
                {'source': [B('images'), B('masks'), cube_name, B('locations')],
                 'name': 'masks', 'loc': B('location'), 'plot_function': self.custom_plotter},
                {'source': [B('images'), B('prediction'), cube_name, B('locations')],
                 'name': 'prediction', 'loc': B('location'), 'plot_function': self.custom_plotter}
            ]
            if self.config['train/normalization_layer']:
                src += [
                    {'source': [B('norm_images'), None, cube_name, B('locations')],
                    'name': 'norm_images', 'loc': B('location'), 'plot_function': self.custom_plotter},
                ]
            if self.config['train/phase']:
                src += [
                    {'source': [B('phases'), None, cube_name, B('locations')],
                    'name': 'phases', 'loc': B('location'), 'plot_function': self.custom_plotter},
                ]
            return Notifier(True, graphs=['loss_history', *src],
            total=self.config['train/n_iters'],
            figsize=(40, 10))
        return super().make_notifier()

    def custom_plotter(self, ax=None, container=None, **kwargs):
        """ Plot examples during train stage. """
        images, masks, cube_name, locations = container['data']

        loc = [(slc.start, slc.stop) for slc in locations[0]]
        loc = '  '.join([f'{item[0]}:{item[1]}' for item in loc])

        if masks is not None:
            title = f"{cube_name}  {loc}\n{container['name']}"
            if images.ndim == 4:
                images, masks = images[0][0], masks[0][0]
            elif images.ndim == 5:
                pos = images.shape[2] // 2
                images, masks = images[0][0][pos], masks[0][0][pos]
            plot_image([images, masks > 0.5], overlap=True, title=title, ax=ax, displayed_name="")
        else:
            title = f"{cube_name}  {loc}\n{container['name']}"
            if images.ndim == 4:
                images = images[0][0]
            elif images.ndim == 5:
                pos = images.shape[2] // 2
                images = images[0][0][pos]
            plot_image(images, overlap=True, title=title, ax=ax, displayed_name="")

    def get_train_template(self, synthetic=False, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        if not synthetic:
            return (
                self.load_train(**kwargs) +
                (self.augmentation_pipeline(**kwargs) if self.config['train/augment'] else Pipeline()) +
                self.train_pipeline(**kwargs)
            )

        return self.load_synthetic(**kwargs) + self.train_pipeline(**kwargs)

    def get_model_config(self, name):
        """ Get model config depending model architecture. """
        if name == 'UNet':
            return (Config(self.BASE_MODEL_CONFIG) + Config(self.UNET_CONFIG)).flatten()
        raise ValueError(f'Unknown model name: {name}')

    def get_model_class(self, name):
        """ Get model class depending model architecture. """
        if name == 'UNet':
            return EncoderDecoder
        return TorchModel

    def dump_model(self, model, path):
        """ Dump model. """
        model.save(os.path.join(self.config['savedir'], path))

    # Inference functional

    def metrics_pipeline_template(self, n_bins=1000):
        """ Metrics pipeline. """
        return (Pipeline()
            .init_variable('metric', [])
            .init_variable('semblance_hist', np.zeros(n_bins))
            .compute_attribute(src='images', dst='semblance', attribute='semblance', window=(1, 5, 20))
            .update(V('metric', mode='e'), F(FaultsMetrics().similarity_metric)(B('semblance'), B('predictions')))
            .update(V('semblance_hist'), V('semblance_hist') + F(np.histogram)(B('semblance').flatten(),
                    bins=n_bins, range=(0, 1))[0])
        )

    def get_inference_template(self, train_pipeline=None, model_path=None, create_masks=False):
        """ Define the whole inference procedure pipeline. """
        if train_pipeline is not None:
            test_pipeline = Pipeline().import_model('model', train_pipeline)
        else:
            test_pipeline = Pipeline().load_model('model', TorchModel, 'dynamic', path=model_path)

        test_pipeline += self.load_test(create_masks=create_masks)
        test_pipeline += (
            Pipeline()
            .adaptive_expand(src='images')
            .init_variable('predictions', [])
            .init_variable('target', [])
            .predict_model('model', B('images'), fetches=C('output'), save_to=B('predictions'))
            .adaptive_squeeze(src='predictions')
            .run_later(D('size'))
        )

        if self.config['inference/smooth_borders']:
            if isinstance(self.config['inference/smooth_borders'], bool):
                step = 0.1
            else:
                step = self.config['inference/smooth_borders']
            test_pipeline += Pipeline().update(B('predictions') , F(smooth_borders)(B('predictions'), step))

        if create_masks:
            test_pipeline += Pipeline().update(V('target', mode='e'), B('masks'))

        test_pipeline += self.metrics_pipeline_template()
        test_pipeline += (Pipeline()
            .update_accumulator(src='predictions', accumulator=C('accumulator'))
        )

        return test_pipeline

    def get_inference_ranges(self, cubes):
        """ Parse inference ranges. """
        cubes = cubes.copy()
        if isinstance(cubes, (list, tuple)):
            cubes = {cube: (0, None, None, None) for cube in cubes}
        for cube in cubes:
            cubes[cube] = [cubes[cube]] if isinstance(cubes[cube][0], int) else cubes[cube]
        return cubes

    def make_accumulator(self, geometry, slices, crop_shape, strides, orientation=0, path=None):
        """ Make grid and accumulator for inference. """
        batch_size = self.config['inference']['inference_batch_size']
        aggregation = self.config['inference']['aggregation']

        grid = RegularGrid(geometry=geometry,
                           threshold=0,
                           orientation=orientation,
                           ranges=slices,
                           batch_size=batch_size,
                           crop_shape=crop_shape, strides=strides)

        accumulator = Accumulator3D.from_aggregation(aggregation=aggregation,
                                                     origin=grid.origin,
                                                     shape=grid.shape,
                                                     fill_value=0.0,
                                                     path=path)

        return grid, accumulator

    def inference_on_slides(self, train_pipeline=None, model_path=None, create_mask=False, pbar=False, **kwargs):
        """ Make inference on slides. """
        config = {**self.config['dataset'], **self.config['inference'], **kwargs}
        crop_shape = config['crop_shape']
        cubes = config['inference_cubes']
        strides = config['stride']

        strides = strides if isinstance(strides, tuple) else [strides] * 3

        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        self.log('Create test pipeline and dataset.')
        dataset = self.make_inference_dataset(create_mask)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_mask)
        inference_pipeline = inference_pipeline << config << dataset


        cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.get_inference_ranges(cubes).items()
        }

        outputs = {}
        for cube_idx in dataset.indices:
            outputs[cube_idx] = []
            geometry = dataset.geometries[cube_idx]
            for item in cubes[cube_idx]:
                self.log(f'Create prediction for {cube_idx}: {item[1:]}. orientation={item[0]}.')
                orientation = item[0]
                slices = item[1:]
                if len(slices) != 3:
                    slices = (None, None, None)

                grid, accumulator = self.make_accumulator(geometry, slices, crop_shape, strides, orientation)
                ppl = inference_pipeline << {'sampler': grid, 'accumulator': accumulator}

                notifier = {
                    'file': self.make_savepath('末 inference_bar.log'),
                }

                ppl.run(n_iters=grid.n_iters, notifier=notifier)
                prediction = accumulator.aggregate()

                image = geometry[
                    slices[0][0]:slices[0][1],
                    slices[1][0]:slices[1][1],
                    slices[2][0]:slices[2][1]
                ]

                if orientation == 1:
                    image = image.transpose([1, 0, 2])
                    prediction = prediction.transpose([1, 0, 2])

                metrics = np.nanmean(ppl.v('metric')), np.argmax(ppl.v('semblance_hist')) / 1000
                outputs[cube_idx] += [[slices, image, prediction, *metrics]]
        return outputs

    def visualize_predictions(self, *args, overlap=True, threshold=0.05, each=100, iteration=0, skeletonize=False, clear=False, width=3, figsize=(20, 20), **kwargs):
        """ Plot predictions for cubes and ranges specified in 'inference' section of config. """
        if each is not None and iteration % each == 0:
            results = self.inference_on_slides(*args, **kwargs)
            for cube, cube_results in results.items():
                for item in cube_results:
                    slices, image, prediction, faults_metric, noise_metric = item[:5]

                    if self.config['savedir'] is not None:
                        savepath = os.path.join(self.config['savedir'], 'prediction')
                        if not os.path.exists(savepath):
                            os.makedirs(savepath)
                        savepath = os.path.join(
                            savepath,
                            f'{cube}_{slices}_{iteration}_{faults_metric:.03f}_{noise_metric:.03f}.png'
                        )
                        show = False
                    else:
                        savepath = None
                        show = True

                    if skeletonize:
                        prediction = Fault.skeletonize_faults(prediction, threshold=threshold, mode='array', width=width)
                    if clear:
                        prediction = Fault.remove_predictions_on_bounds(image, prediction)

                    if overlap:
                        plot_image([image[0], prediction[0] > threshold], separate=False, figsize=figsize,
                                   savepath=savepath, show=show)
                    else:
                        plot_image(prediction[0], figsize=figsize, savepath=savepath, show=show)
            return faults_metric, noise_metric
        return None, None

    def inference_on_cube(self, train_pipeline=None, model_path=None, fmt='sgy', save_to=None, filename=None,
                          tmp='hdf5', bar=True, **kwargs):
        """ Make inference on cube. """
        config = {**self.config['dataset'], **self.config['inference'], **kwargs}
        strides = config['stride'] if isinstance(config['stride'], tuple) else [config['stride']] * 3

        crop_shape = config['crop_shape']
        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        self.log('Create test pipeline and dataset.')

        dataset = self.make_inference_dataset(create_mask=False)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_masks=False)
        inference_pipeline.set_config(config)

        # Log: pipeline_config to a file
        self.log_to_file(pformat(config, depth=2), '末 inference_config.txt')

        cubes = config['inference_cubes']
        cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.get_inference_ranges(cubes).items()
        }

        if save_to:
            dirname = save_to
        else:
            dirname = os.path.join(
                    os.path.dirname(dataset.geometries[0].path),
                    'PREDICTIONS/FAULTS',
            )
            if not os.path.exists(os.path.dirname(dirname)):
                os.makedirs(os.path.dirname(dirname))
            if not os.path.exists(dirname):
                os.makedirs(dirname)

        outputs = dict()
        for cube_idx in dataset.indices:
            outputs[cube_idx] = []
            geometry = dataset.geometries[cube_idx]
            for item in cubes[cube_idx]:
                if self.monitor:
                    monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
                    monitor.__enter__()

                axis = item[0]
                ranges = item[1:]
                if len(ranges) != 3:
                    ranges = (None, None, None)

                filenames = {ext: os.path.join(dirname, filename)
                             for ext in ['hdf5', 'sgy', 'blosc', 'qblosc', 'npy', 'meta']}

                if fmt in ['hdf5', 'blosc', 'qblosc']:
                    path = filenames[fmt]
                elif fmt in ['sgy', 'segy']:
                    path = filenames[tmp]
                elif fmt == 'npy':
                    path = None

                self.log(f'Create prediction {path} for {cube_idx}: {item[1:]}. axis={item[0]}.')

                grid, accumulator = self.make_accumulator(geometry, ranges, crop_shape, strides, axis, path)
                ppl = inference_pipeline << dataset << {'sampler': grid, 'accumulator': accumulator}

                notifier = {
                    'file': self.make_savepath('末 inference_bar.log'),
                }

                self.log(f'n_iters: {grid.n_iters}')

                ppl.run(n_iters=grid.n_iters, notifier=notifier)
                prediction = accumulator.aggregate()

                self.log(f'Finish prediction {path}')

                if self.monitor:
                    monitor.__exit__(None, None, None)
                    monitor.visualize(savepath=self.make_savepath('末 inference_resource.png'), show=self.plot)

                if fmt == 'npy':
                    outputs.append(prediction)
                if fmt == 'sgy':
                    copyfile(dataset.geometries[0].path_meta, filenames['meta'])
                    dataset.geometries[0].make_sgy(
                        path_hdf5=filenames[tmp],
                        path_spec=dataset.geometries[0].segy_path.decode('utf-8'),
                        path_segy=filenames['sgy'],
                        remove_hdf5=True, zip_result=True, pbar=True
                    )
                if fmt in ['hdf5', 'blosc', 'qblosc']:
                    accumulator.file.close()

    def process(self, cube_path, prediction_path, orientation=0, skeletonize=True, clear=True, threshold=0.05, width=3, bar=True):
        orientation = int(orientation)
        geometry = SeismicGeometry(cube_path)
        prediction = h5py.File(prediction_path, mode='a')

        self.log(f'Start processing {prediction_path}')

        for i in range(geometry.cube_shape[orientation]):
            self.log(f'Process slide {i}/{geometry.cube_shape[orientation]}')
            if orientation == 0:
                image, slide = geometry.file['cube_i'][i:i+1], prediction['cube_i'][i:i+1]
            else:
                image, slide = geometry.file['cube_x'][i:i+1].transpose(0, 2, 1), prediction['cube_i'][:, i:i+1].transpose(1, 0, 2)
            if prediction:
                slide = Fault.skeletonize_faults(slide, threshold=threshold, mode='array', width=width, bar=False)
            if clear:
                slide = Fault.remove_predictions_on_bounds(image, slide)

            if orientation == 0:
                prediction['cube_i'][i:i+1] = slide
            else:
                prediction['cube_i'][:, i:i+1] = slide.transpose(1, 0, 2)

        self.log(f'Start faults splitting for {prediction_path}')

        faults = Fault.from_mask(SeismicGeometry(prediction_path), geometry, chunk_size=100, threshold=0.05)
        ds = SeismicCubeset(index=cube_path)
        ds.load(label_dir='dummy', labels_class=Fault)
        ds.labels[0] = faults
        ds.dump_labels(f'/PREDICTIONS/FAULTS/SKELETON_{orientation}', separate=True)

        self.log(f'Finish faults splitting for {prediction_path}')

    # Path utils

    def amplitudes_path(self, cube):
        """ Get full path for cube. """
        ext = self.config['dataset/ext']
        filename = self.config['dataset/path'] + f'{cube}/{cube}.{ext}'
        if os.path.exists(filename):
            return filename
        raise ValueError(f"File doesn't exist: {filename}")

    def cube_name_from_alias(self, path):
        """ Get cube name from alias. """
        return os.path.splitext(self.amplitudes_path(path).split('/')[-1])[0]

    def cube_name_from_path(self, path):
        """ Get cube name from path. """
        return os.path.splitext(path.split('/')[-1])[0]

    def make_filename(self, prefix, orientation, ext):
        """ Make filename for infered cube. """
        return (prefix + '_{}.{}').format(orientation, ext)

    # Train utils

    def adjust_shape(self, crop_shape):
        """ Adjust shape before augmentations. """
        config = self.config['train']
        crop_shape = np.array(crop_shape)
        load_shape = adjust_shape_3d(crop_shape[[1, 2, 0]], config['angle'], scale=config['scale'])
        return (load_shape[2], load_shape[0], load_shape[1])

# Train utils

def smooth_borders(crops, step):
    """ Smooth image borders. """
    mask = border_smoothing_mask(crops.shape[-3:], step)
    mask = np.expand_dims(mask, axis=0)
    if len(crops.shape) == 5:
        mask = np.expand_dims(mask, axis=0)
    crops = crops * mask
    return crops

def border_smoothing_mask(shape, step):
    """ Make mask to smooth borders. """
    mask = np.ones(shape)
    axes = [(1, 2), (0, 2), (0, 1)]
    if isinstance(step, (int, float)):
        step = [step] * 3
    for length, s, axis in zip(shape, step, axes):
        if isinstance(s, float):
            s = int(length * s)
        if length >= 2 * s:
            _mask = np.ones(length, dtype='float32')
            _mask[:s] = np.linspace(0, 1, s+1)[1:]
            _mask[:-s-1:-1] = np.linspace(0, 1, s+1)[1:]
            _mask = np.expand_dims(_mask, axis)
            mask = mask * _mask
    return mask
