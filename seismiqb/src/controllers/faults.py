""" A convenient holder for faults detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
"""
import os

import numpy as np
import torch

from .base import BaseController
from ..labels import Fault, skeletonize
from ..dataset import SeismicDataset
from ..geometry import SeismicGeometry
from ..samplers import SeismicSampler, RegularGrid, FaultSampler, ConstantSampler
from ..metrics import FaultsMetrics
from ..utils import adjust_shape_3d, Accumulator3D, GaussianLayer, expand_dims, squueze, faults_sizes
from ..plotters import plot_image

from ...batchflow import Config, Pipeline, Notifier, Monitor
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
            'inference_cubes': {},
            'label_dir': '~/INPUTS/FAULTS/NPY_WIDTH_{}/*',
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
            'callback': [
                (
                    'visualize_predictions',
                    100,
                    {'model': B().pipeline, 'iteration': I()})
            ],
            'visualize_crops': True,

            # Augmentation parameters
            'angle': 25,
            'scale': (0.7, 1.5, 1),
            'augment': False,
            'adjust': False,

            # Mask creation params
            'sparse': False,

            # Normalization parameters
            'norm_mode': 'minmax',
            'itemwise': True,

            # Model parameters
            'model_class': EncoderDecoder,
            'native_slicing': True,
            'prefetch': 0,
            'rescale_batch_size': False,
            'model_config': None,
        },

        'inference': {
            'crop_shape': [1, 128, 512],
            'batch_size': 32,
            'smooth_borders': True,
            'stride': 0.5,
            'orientation': 'ilines',
            'native_slicing': True,
            'output': 'sigmoid',
            'norm_mode': 'minmax',
            'itemwise': True,
            'aggregation': 'max',
            'prefetch': 4,
            'margin': (0, 0)
        }
    })

    def make_train_dataset(self, **kwargs):
        """ Create train dataset as an instance of :class:`.SeismicDataset` with cubes and faults.
        Arguments are from 'dataset' subconfig and kwargs. """
        return self._make_dataset(train=True, **kwargs)

    def make_inference_dataset(self, labels=False, **kwargs):
        """ Create inference dataset as an instance of :class:`.SeismicDataset` with cubes and faults.
        Arguments are from 'dataset' subconfig and kwargs. """
        return self._make_dataset(train=False, labels=labels, **kwargs)

    def _make_dataset(self, train=True, labels=True, **kwargs):
        """ Create an instance of :class:`.SeismicDataset` with cubes and faults. Arguments are from 'dataset'
        subconfig and kwargs. """
        config = {**self.config['dataset'], **kwargs}
        width = config['width']
        label_dir = config['label_dir']
        cubes = config['train_cubes'] if train else config['inference_cubes']
        if isinstance(cubes, str):
            cubes = [cubes]

        paths = [self.amplitudes_path(item) for item in cubes]

        if labels:
            transposed = config['transposed_cubes']
            direction = {f'{cube}': 0 if cube not in transposed else 1 for cube in cubes}
            index = {path: label_dir.format(width) for path in paths}
            dataset = SeismicDataset(index=index, labels_class=Fault, direction=direction)
        else:
            dataset = SeismicDataset(index=paths, logs=False)

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
            .load_cubes(dst='images', native_slicing=C('native_slicing'))
            .create_masks(dst='masks', sparse=C('sparse'))
            .adaptive_reshape(src=['images', 'masks'])
            .normalize(mode=C('norm_mode'), itemwise=C('itemwise'), src='images')
        )

    def load_test(self, create_masks=True):
        """ Create loading pipeline for inference stages. """
        ppl = (Pipeline()
            .make_locations(batch_size=C('batch_size'), generator=C('sampler'))
            .load_cubes(dst='images', native_slicing=C('native_slicing'))
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
        """ Make Dataset class for use with seismic data. """
        ds = SeismicDataset(index='/cubes/031_CHIST/031_CHIST.qblosc') #TODO: remove hardcode
        return ds

    def load_synthetic(self, **kwargs):
        """ Create pipeline to generate synthetic data. """
        refs = NS('c', a=np.arange(50, 70)).apply(lambda x: x[0, 0])
        heights = (NS('u', dim=2) * [0.2, 0.2] + [0.25, 0.55]).apply(np.squeeze)
        muls = NS('c', a=np.arange(-8, -4)).apply(lambda x: x[:, 0])

        _, xl, h = self.config['train/crop_shape']

        def faults():
            x0 = np.random.randint(14, xl-10)
            x1 = np.random.randint(14, xl-10) # same prmtzn for x1 as for x0; you can change it ofc
            y0 = np.random.randint(0, 200)
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
                                    dst=('images', 'horizons_', 'masks'),
                                    zeros_share_faults=0.1,
                                    max_shift=10, # controls max fault-shift
                                    geobodies_width=(5, 5) # default value is 3
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
        model_class = self.config['train/model_class']
        model_config = self.config['train/model_config']

        ppl = (Pipeline()
            .init_variable('loss_history', [])
            .init_model(name='model', model_class=model_class, mode='dynamic', config=model_config)
            .adaptive_expand(src=['images', 'masks'])
            .train_model('model',
                         fetches='loss',
                         images=B('images'),
                         masks=B('masks'),
                         save_to=V('loss_history', mode='a'))
        )
        callbacks = [
            (getattr(self, name), each, kwargs) for name, each, kwargs in self.config['train/callback']
        ]

        for callback, each, kwargs in callbacks:
            ppl += Pipeline().call(callback, each=each, **kwargs)

        if self.config['train/visualize_crops']:
            output = self.config['inference/output']
            ppl += (Pipeline()
                .predict_model('model', images=B('images')[:1], fetches=output, save_to=B('prediction'))
            )
            if self.config['train/model_config/initial_block/phases']:
                ppl += Pipeline().update(B('phases'), M('model').get_intermediate_activations(
                    images=B('images')[:1],
                    layers=M('model').model[0][0].phase_layer)
                )
            if self.config['train/model_config/initial_block/normalization']:
                ppl += Pipeline().update(B('norm_images'), M('model').get_intermediate_activations(
                    images=B('images')[:1],
                    layers=M('model').model[0][0].normalization_layer)
                )
        return ppl

    def train(self, dataset, sampler=None, synthetic=False, config=None, **kwargs):
        """ Train the model. """
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
            cube_name = B().unsalt(B('indices')[0])
            src = [
                {'source': [B('images'), B('masks'), cube_name, B('locations')],
                 'name': 'masks', 'loc': B('location'), 'plot_function': self.custom_plotter},
                {'source': [B('prediction'), None, cube_name, B('locations')],
                 'name': 'prediction', 'loc': B('location'), 'plot_function': self.custom_plotter}
            ]
            if self.config['train/model_config/initial_block/normalization']:
                src += [
                    {'source': [B('norm_images'), None, cube_name, B('locations')],
                    'name': 'norm_images', 'loc': B('location'), 'plot_function': self.custom_plotter},
                ]
            if self.config['train/model_config/initial_block/phases']:
                src += [
                    {'source': [B('phases'), None, cube_name, B('locations')],
                    'name': 'phases', 'loc': B('location'), 'plot_function': self.custom_plotter},
                ]
            return Notifier(True, graphs=['loss_history', *src], total=self.config['train/n_iters'], figsize=(40, 10))
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

    def get_inference_template(self, model, create_masks):
        """ Define the whole inference procedure pipeline. """
        test_pipeline = Pipeline().init_model(name='model', source=model)

        test_pipeline += self.load_test(create_masks=create_masks)
        test_pipeline += (
            Pipeline()
            .adaptive_expand(src='images')
            .init_variable('predictions', [])
            .init_variable('target', [])
            .predict_model('model', B('images'), fetches=self.config['inference/output'], save_to=B('predictions'))
            .adaptive_squeeze(src='predictions')
            .fill_bounds(src='predictions', margin=C('margin'))
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

    def make_accumulator(self, field, ranges, crop_shape, strides, orientation=0, full=False, path=None):
        """ Make grid and accumulator for inference. """
        batch_size = self.config['inference']['batch_size']
        aggregation = self.config['inference']['aggregation']

        grid = RegularGrid(field=field,
                           threshold=0,
                           orientation=orientation,
                           ranges=ranges,
                           batch_size=batch_size,
                           crop_shape=crop_shape,
                           strides=strides)

        origin = (0, 0, 0) if full else grid.origin
        shape = field.shape if full else grid.shape

        accumulator = Accumulator3D.from_aggregation(aggregation=aggregation,
                                                     origin=origin,
                                                     shape=shape,
                                                     fill_value=0.0,
                                                     path=path)

        return grid, accumulator

    def visualize_predictions(self, model, overlap=True, each=100, iteration=0, threshold=0.01, figsize=(20, 20)):
        """ Plot predictions for cubes and ranges specified in 'inference' section of config. """
        if each is not None and (iteration + 1) % each == 0:
            dataset = self.make_inference_dataset()
            cubes = self.config['dataset/inference_cubes']
            for cube in cubes:
                if isinstance(cubes, list):
                    ranges = None, None, None
                    orientation = 0
                else:
                    ranges = cubes[cube][1:]
                    orientation = cubes[cube][0]
                    if len(ranges) == 0:
                        ranges = None, None, None

                if self.config['savedir'] is not None:
                    savepath = os.path.join(self.config['savedir'], 'predictions', cube)
                    if not os.path.exists(savepath):
                        os.makedirs(savepath)
                    savepath = os.path.join(savepath, f'{orientation}_{ranges}_{iteration}_')
                    show = False
                else:
                    savepath = None
                    show = True

                origin = [item[0] if item is not None else None for item in ranges]
                loc = ranges[orientation][0]
                zoom_slice = tuple(slice(*ranges[i]) for i in range(3) if i != orientation)

                model_prediction = self.inference(dataset, model, idx=cube, ranges=ranges, orientation=orientation)

                processed_prediction = self.process_bounds(prediction=model_prediction, geometry=dataset[cube].geometry,
                                                           inplace=False, origin=origin)
                self.smooth_predictions(prediction=processed_prediction, inplace=True)
                self.skeletonize(prediction=processed_prediction, inplace=True, orientation=orientation)

                image = dataset[cube].load_slide(loc, orientation=orientation)[zoom_slice]

                for prediction, name in zip([model_prediction, processed_prediction], ['model', 'processed']):
                    slide = prediction[0] if orientation == 0 else prediction[:, 0]

                    filename = savepath + f'{name}_' + 'prediction.png' if savepath else None
                    slide[slide <= threshold] = np.nan
                    plot_image([image, slide], figsize=figsize, separate=False, savepath=filename, show=show)

                    filename = savepath + f'{name}_' + 'mask.png' if savepath else None
                    plot_image(slide, figsize=figsize, savepath=filename, show=show)

    def inference(self, dataset, model, idx=0, ranges=None, orientation=0, dst=None, full=False, **kwargs):
        """ Start inference.

        Parameters
        ----------
        dataset : SeismicDataset

        model : str ot TorchModel
            path to the model or the model itself.
        idx : int, optional
            index of the cube in the dataset to infer, by default 0.
        ranges : tuple of tuples or None, optional
            each tuple is range for correspong axis to infer, by default None. If None, the whole cube will be infered.
        orientation : int, optional
            axis to infer along, by default 0
        dst : str or None, optional
            path to HDF5 to store result, by default None. If None,  the result will be returned in numpy.ndarray.
        full : bool, optional
            Is applied when 'ranges' is not None. If True then the result will have the same shape as initial cube with
            prediction in 'ranges', if False then the result will be of the same shape as 'ranges' defines.

        Returns
        -------
        numpy.ndarray or HDF5 dataset (depends on 'dst')
        """
        config = {**self.config['inference'], **kwargs}
        crop_shape = config['crop_shape']
        if orientation == 0:
            crop_shape = np.minimum(crop_shape, dataset[idx].shape)
        else:
            crop_shape = np.minimum(crop_shape, np.array(dataset[idx].shape)[[1, 0, 2]])

        strides = config['stride']
        prefetch = config['prefetch']

        strides = strides if isinstance(strides, tuple) else [strides] * 3
        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        ranges = (None, None, None) if ranges is None else ranges

        self.log(f'Start inference for {dataset.indices[0]} at range {ranges} with orientation {orientation}.')
        self.log('Create test pipeline and dataset.')

        if isinstance(model, str):
            model_ = TorchModel()
            model_.load(model)
            model = model_

        if dst is not None:
            dst = dataset.fields[idx].make_path(dst)
            self.log(f'Result will be saved to {dst}.')
        else:
            self.log('Result will be saved to numpy array.')

        inference_pipeline = self.get_inference_template(model, create_masks=False)
        inference_pipeline = inference_pipeline << config << dataset

        grid, accumulator = self.make_accumulator(dataset.fields[idx], ranges, crop_shape,
                                                  strides, orientation, full, dst)
        ppl = inference_pipeline << dataset << {'sampler': grid, 'accumulator': accumulator}

        notifier = {'file': self.make_savepath('末 inference_bar.log')} #TODO: make notifier
        self.log(f'n_iters: {grid.n_iters}')


        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
            monitor.__enter__()

        ppl.run(n_iters=grid.n_iters, notifier=notifier, prefetch=prefetch)

        if self.monitor:
            monitor.__exit__(None, None, None)
            monitor.visualize(savepath=self.make_savepath('末 inference_resource.png'), show=self.plot)
            del monitor

        prediction = accumulator.aggregate()

        self.log('Finish inference.')

        return prediction

    def smooth_predictions(self, prediction, chunk_size=100, kernel=(11, 11, 11), inplace=False, dst=None):
        """ Smooth proba prediction by gaussian kernel by chunks along the first axis.

        Parameters
        ----------
        prediction : numpy.ndarray or HDF5/BLOSC dataset.

        chunk_size : int, optional
            The size of the chunk along the first axis, by default 100. If None, smoothing will be performed without
            chunk splitting.
        kernel : tuple, optional
            gaussian kernel size, by default (11, 21, 11)
        inplace : bool, optional
            make smoothing inplace or not, by default False. If True and dst is None, numpy.ndarray of the same shape
            will be created.
        dst : numpy.ndarray, HDF5/BLOSC dataset or None, optional
            dst for processed prediction if inplace is False.

        Returns
        -------
        numpy.ndarray or HDF5/BLOSC dataset
            smoothed prediction
        """
        self.log('Start smoothing predictions.')
        kernel = np.minimum(kernel, prediction.shape)
        if chunk_size is None:
            chunk_size = prediction.shape[0]
        if chunk_size < kernel[0]:
            raise ValueError(f"The first dim of kernel can't be greater then chunk_size but {kernel[0]} > {chunk_size}")
        chunk_size = min(chunk_size, prediction.shape[0])

        if inplace:
            dst = prediction
        elif dst is None:
            dst = np.empty_like(prediction)
        elif dst.shape != prediction.shape:
            raise ValueError(f"dst must be of the same shape as prediction but {dst.shape} != {prediction.shape}")

        start = 0
        overlap = None

        while True:
            stop = min(start + chunk_size, prediction.shape[0])
            chunk = prediction[start:stop]

            # Only the first and the last chunks need zero padding, other chunks must have valid padding
            # to make smoothing seamless.
            padding = [[item // 2, item - item // 2 - 1] for item in kernel]
            if start > 0:
                padding[0][0] = 0
            if stop < prediction.shape[0]:
                padding[0][1] = 0

            inputs = torch.tensor(chunk)
            layer = GaussianLayer(inputs, kernel_size=kernel, padding=padding)
            output = layer(inputs).numpy()

            # Compute positions of the processed `inputs` (depends on the padding)
            output_start = 0 if start == 0 else start + kernel[0] // 2
            output_stop = output_start + output.shape[0]

            # `overlap` is needed in the case when `dst == predictions` to correctly smooth data.
            if start != 0:
                dst[start:start+kernel[0] // 2] = overlap

            if stop == prediction.shape[0]:
                dst[output_start:output_stop] = output
                break

            dst[output_start:output_stop - kernel[0] // 2] = output[:-kernel[0] // 2 + 1]
            overlap = output[-kernel[0] // 2+1:]
            start = output_stop - kernel[0] // 2
        self.log('Finish smoothing predictions.')
        return dst

    def skeletonize(self, prediction, orientation=0, skeleton_width=3, proba=True, inplace=False, dst=None, **kwargs):
        """ Skeletonize predictions along one axis.

        Parameters
        ----------
        prediction : numpy.ndarray or dataset of HDF5.

        orientation : int, optional
            axis to perform skeletonize, by default 0
        skeleton_width : int, optional
            width of the skeletonized faults, by default 3
        proba : bool, optional
            return soft proba max (probabilities from initial prediction) or hard binary mask.
        inplace : bool, optional
            make smoothing inplace or not, by default False. If True and dst is None, numpy.ndarray of the same shape
            will be created.
        dst : numpy.ndarray, HDF5/BLOSC dataset or None, optional
            dst for processed prediction if inplace is False.

        Returns
        -------
        numpy.ndarray or HDF5/BLOSC dataset
            skeletonized prediction
        """
        self.log('Start skeletonize.')
        if inplace:
            dst = prediction
        elif dst is None:
            dst = np.empty_like(prediction)
        elif dst.shape != prediction.shape:
            raise ValueError(f"dst must be of the same shape as prediction but {dst.shape} != {prediction.shape}")

        for i in range(prediction.shape[orientation]):
            slide = prediction[i] if orientation == 0 else prediction[:, i]

            if proba:
                slide = slide * skeletonize(slide, width=skeleton_width, **kwargs)
            else:
                slide = skeletonize(slide, width=skeleton_width, **kwargs)

            pooling = torch.nn.MaxPool3d((1, skeleton_width, 1), stride=1, padding=(0, skeleton_width // 2, 0))
            slide = torch.tensor(slide)
            slide = squueze(pooling(expand_dims(slide)), 2).numpy()

            if orientation == 0:
                dst[i] = slide
            else:
                dst[:, i] = slide
        self.log('Finish skeletonize.')
        return dst

    def process_bounds(self, prediction, geometry, origin=(0, 0, 0), window=30,
                       fill_value=0, inplace=False, dst=None, **kwargs):
        """ Remove prediction on bounds (between zero regions and filled).

        Parameters
        ----------
        prediction : numpy.ndarray or dataset of HDF5.
            prediction of the part of the cube which position is defined by origin.
        geometry : SeismicGeometry
            geometry to find bounds
        origin : tuple, optional
            position of the prediction respectively to geometry, by default (0, 0, 0)
        window : int, optional
            size of vertical window to find constant regions, by default 30
        fill_value : int, optional
            Value for bounds in prediction, by default 0
        inplace : bool, optional
            make smoothing inplace or not, by default False. If True and dst is None, numpy.ndarray of the same shape
            will be created.
        dst : numpy.ndarray, HDF5/BLOSC dataset or None, optional
            dst for processed prediction if inplace is False.
        kwargs : dict
            kwargs for :meth:`~.FaultController.bounds_mask`.

        Returns
        -------
        numpy.ndarray or HDF5/BLOSC dataset
            skeletonized prediction
        """
        self.log('Start bounds processing.')
        if inplace:
            dst = prediction
        elif dst is None:
            dst = np.empty_like(prediction)
        elif dst.shape != prediction.shape:
            raise ValueError(f"dst must be of the same shape as prediction but {dst.shape} != {prediction.shape}")

        if isinstance(geometry, str):
            geometry = SeismicGeometry(geometry)

        for i in range(prediction.shape[0]):
            image = geometry.load_slide(origin[0]+i)[
                origin[1]:origin[1]+prediction.shape[1],
                origin[2]:origin[2]+prediction.shape[2]
            ]

            mask = self.bounds_mask(image, window=window, **kwargs)
            slide = prediction[i].copy()
            slide[mask] = fill_value
            dst[i] = slide
        self.log('Finish bounds processing.')
        return dst

    def bounds_mask(self, image, window=30, dilation=30, padding=True, threshold=0):
        """ Remove predictions from cube bounds.

        Parameters
        ----------
        image : numpy.ndarray
            seismic image to find bounds
        window : int, optional
            size of vertical window to find constant regions, by default 30
        dilation : int, optional
            dilation of detected bounds, by default 30
        padding : bool, optional
            make padding or not, by default True
        threshold : int, optional
            threshold for ptp in the window, by default 0

        Returns
        -------
        numpy.ndarray
            mask of bounds
        """
        dilation = [dilation] * image.ndim if isinstance(dilation, int) else dilation
        if padding:
            pad_width = [(0, 0)] * image.ndim
            pad_width[-1] = (window // 2, window // 2)
            image = np.pad(image, pad_width=pad_width)

        # Create array of windows to find constant regions
        shape = (*image.shape[:-1], image.shape[-1] - window + window % 2, window)
        strides = (*image.strides, image.strides[-1])
        strided = np.lib.stride_tricks.as_strided(image, shape, strides=strides)

        # Find centers of almost constant vertical windows
        if padding:
            mask = strided.ptp(axis=-1) <= threshold
        else:
            mask = np.ones_like(image, dtype=np.bool)
            slices = [slice(None)] * image.ndim
            slices[-1] = slice(window // 2, -window // 2 + 1)
            mask[slices] = strided.ptp(axis=-1) <= threshold

        for i, width in enumerate(dilation):
            slices = [[slice(None) for _ in range(image.ndim)] for _ in range(2)]
            slices[0][i] = slice(1, None)
            slices[1][i] = slice(None, -1)
            slices = [tuple(item) for item in slices]
            for _ in range(1, width):
                mask[slices[0]] = np.logical_or(mask[slices[0]], mask[slices[1]])
                mask[slices[1]] = np.logical_or(mask[slices[0]], mask[slices[1]])

        return mask

    def compute_sizes(self, prediction, orientation=0, normalize=True, multiply=True, inplace=False, dst=None):
        """ Compute sizes (depth length) of connected objects on 2D slides.

        Parameters
        ----------
        prediction : numpy.ndarray or dataset of HDF5.

        orientation : int, optional
            axis of slides to compute, by default 0
        normalize : bool, optional
            normalize size by depth of the slide or not.
        multiply : bool, optional
            multiply prediction on sizes or not. Can be needed if prediction is probabilities.
        inplace : bool, optional
            make smoothing inplace or not, by default False. If True and dst is None, numpy.ndarray of the same shape
            will be created.
        dst : numpy.ndarray, HDF5/BLOSC dataset or None, optional
            dst for processed prediction if inplace is False.

        Returns
        -------
        numpy.ndarray or HDF5/BLOSC dataset
            prediction where values are multiplied by size of computed objects.
        """
        self.log('Start sizes computation.')
        if inplace:
            dst = prediction
        elif dst is None:
            dst = np.empty_like(prediction)
        elif dst.shape != prediction.shape:
            raise ValueError(f"dst must be of the same shape as prediction but {dst.shape} != {prediction.shape}")

        for i in range(prediction.shape[orientation]):
            slide = prediction[i] if orientation == 0 else prediction[:, i]

            if multiply:
                slide = slide * faults_sizes(slide, normalize)
            else:
                slide = faults_sizes(slide, normalize)

            if orientation == 0:
                dst[i] = slide
            else:
                dst[:, i] = slide
        self.log('Finish sizes computation.')
        return dst

    def filter_prediction(self, prediction, orientation=0, threshold=0.1, inplace=False, dst=None):
        """ Filter prediction by sizes of connected objects on 2D slides.

        Parameters
        ----------
        prediction : numpy.ndarray or dataset of HDF5.

        orientation : int, optional
            axis of slides to compute, by default 0
        threshold : float, optional
            threshold for size of objects. If threshold < 1 then it defines relative length of the appropriate faults
            with respect to the depth size of the cube,
        inplace : bool, optional
            make smoothing inplace or not, by default False. If True and dst is None, numpy.ndarray of the same shape
            will be created.
        dst : numpy.ndarray, HDF5/BLOSC dataset or None, optional
            dst for processed prediction if inplace is False.

        Returns
        -------
        numpy.ndarray or HDF5/BLOSC dataset
            filtered prediction
        """
        self.log('Start filtering.')
        if inplace:
            dst = prediction
        elif dst is None:
            dst = np.empty_like(prediction)
        elif dst.shape != prediction.shape:
            raise ValueError(f"dst must be of the same shape as prediction but {dst.shape} != {prediction.shape}")

        normalize = (threshold < 1)

        for i in range(prediction.shape[orientation]):
            slide = prediction[i] if orientation == 0 else prediction[:, i]

            sizes = faults_sizes(slide, normalize)

            if orientation == 0:
                dst[i] = slide
                dst[i][sizes < threshold] = 0
            else:
                dst[:, i] = slide
                dst[:, i][sizes < threshold] = 0
        self.log('Finish filtering.')
        return dst

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
