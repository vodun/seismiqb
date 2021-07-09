""" A convenient holder for faults detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
"""
import os
from datetime import datetime
import tqdm
from shutil import copyfile

import numpy as np
import torch

from skimage.morphology import skeletonize, skeletonize_3d
from scipy.ndimage import binary_erosion, binary_dilation, generate_binary_structure, binary_fill_holes


from ...batchflow import Config, Pipeline, Notifier
from ...batchflow import B, C, D, P, R, V, F, I
from ...batchflow.models.torch import TorchModel, ResBlock, EncoderDecoder
from .base import BaseController
from ..cubeset import SeismicCubeset, SeismicGeometry
from ..samplers import SeismicSampler, RegularGrid
from ..fault import Fault
from ..layers import InputLayer
from ..utils import adjust_shape_3d, fill_defaults, compute_attribute
from ..utility_classes import Accumulator3D
from ..plotters import plot_image

class FaultController(BaseController):
    DEFAULTS = Config({
        **BaseController.DEFAULTS,
        # Data
        'dataset': {
            'path': '/cubes/',
            'train_cubes': [],
            'transposed_cubes': [],
            'label_dir': '/INPUTS/FAULTS/NPY_WIDTH_{}/*',
            'width': 3,
            'ext': 'qblosc'
        },

        # Model parameters
        'train': {
            # Augmentation parameters
            'batch_size': 1024,
            'microbatch': 8,
            'side_view': False,
            'angle': 25,
            'scale': (0.7, 1.5),
            'adjust': True,
            'crop_shape': [1, 128, 512],
            'filters': [64, 96, 128, 192, 256],
            'itemwise': False,
            'phase': True,
            'continuous_phase': False,
            'normalization_layer': False,
            'normalization_window': (1, 1, 100),
            'model': 'UNet',
            'loss': 'bce',
            'output': 'sigmoid',
            'slicing': 'native',
            'prefetch': 0,
            'rescale_batch_size': False,
            'n_iters': 2000,
            'callback/each': 100,
        },

        'inference': {
            'cubes': dict(),
            'batch_size': 32,
            'side_view': False,
            'crop_shape': [1, 128, 512],
            'inference_batch_size': 32,
            'inference_chunk_shape': (100, None, None),
            'smooth_borders': False,
            'stride': 0.5,
            'orientation': 'ilines',
            'slicing': 'native',
            'output': 'sigmoid',
            'itemwise': False
        }
    })
    # .run_later(D('size'), n_iters=C('n_iters'), n_epochs=None, prefetch=0, profile=False, bar=C('bar')

    BASE_MODEL_CONFIG = {
        'optimizer': {'name': 'Adam', 'lr': 0.01},
        "decay": {'name': 'exp', 'gamma': 0.9, 'frequency': 100, 'last_iter': 2000},
        'microbatch': C('microbatch'),
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
        'initial_block/base_block': InputLayer,
        'body/encoder': {
            'num_stages': 4,
            'order': 'sbd',
            'blocks': {
                'base': ResBlock,
                'n_reps': 1,
                'filters': C('filters')[:-1],
                'attention': 'scse',
            },
        },
        'body/embedding': {
            'base': ResBlock,
            'n_reps': 1,
            'filters': C('filters')[-1],
            'attention': 'scse',
        },
        'body/decoder': {
            'num_stages': 4,
            'upsample': {
                'layout': 'tna',
                'kernel_size': 5,
            },
            'blocks': {
                'base': ResBlock,
                'filters': C('filters')[-2::-1],
                'attention': 'scse',
            },
        },
        'head': {
            'base_block': ResBlock,
            'filters': [16, 8],
            'attention': 'scse'
        },
        'output': torch.sigmoid,
        'common/activation': 'relu6',
        'loss': C('loss')
    }

    def make_dataset(self, **kwargs):
        config = {**self.config['dataset'], **kwargs}
        width = config['width']
        label_dir = config['label_dir']
        paths = [self.amplitudes_path(item) for item in config['train_cubes']]
        transposed = config['transposed_cubes']
        direction = {f'amplitudes_{cube}': 0 if cube not in transposed else 1 for cube in config['train_cubes']}

        dataset = SeismicCubeset(index=paths)
        dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True,
                     direction=direction, verify=True)

        return dataset

    def load_pipeline(self, train=True, create_masks=True, **kwargs):
        """ Create loading pipeline common for train and inference stages.

        Parameters
        ----------
        create_masks : bool, optional
            create mask or not, by default True
        use_adjusted_shapes : bool, optional
            use or not adjusted shapes to perform augmentations changing shape (rotations and scaling),
            by default False.

        Returns
        -------
        batchflow.Pipeline
        """
        load_shape = F(self.adjust_shape)(C('crop_shape'), C('angle'), C('scale')[0]) if train and self.config['train/adjust'] else C('crop_shape')
        load_shape = F(np.array)(load_shape)
        shape = {self.cube_name_from_alias(k): load_shape for k in self.config['dataset/train_cubes']}
        shape.update({self.cube_name_from_alias(k): load_shape[[1, 0, 2]] for k in self.config['dataset/transposed_cubes']})

        ppl = (Pipeline()
            .make_locations(batch_size=C('batch_size'), generator=C('sampler'))
            .load_cubes(dst='images', slicing=C('slicing'))
        )

        if create_masks:
            ppl +=  Pipeline().create_masks(dst='masks')
            components = ['images', 'masks']
        else:
            components = ['images']

        ppl += (Pipeline()
            .adaptive_reshape(src=components)
            .normalize(mode='q', itemwise=C('itemwise'), src='images')
        )
        return ppl

    def augmentation_pipeline(self, **kwargs):
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

    def train_pipeline(self, **kwargs):
        model_class = F(self.get_model_class)(C('model'))
        model_config = F(self.get_model_config)(C('model'))
        return (Pipeline()
            .init_variable('loss_history', [])
            .init_variable('loss', [])
            .init_model('model', model_class, 'dynamic', model_config)
            .adaptive_expand(src=['images', 'masks'])
            .train_model('model',
                         fetches=['loss', C('output')],
                         images=B('images'),
                         masks=B('masks'),
                         save_to=[V('loss', mode='w'), B('predictions')])
            .call(self.plot_inference, train_pipeline=B().pipeline, savepath='prediction', each=self.config['train/callback/each'], iteration=I())
            .update(V('loss_history', mode='a'), V('loss'))
        )

    def custom_plotter(self, ax=None, container=None, **kwargs):
        """ Zero-out center area of the image, change plot parameters. """
        data = container['data']
        data = data[0][0]
        if data.ndim == 3:
            data = data[0]
        ax.imshow(data.T)
        ax.set_title(container['name'], fontsize=18)

        ax.set_xlabel('axis one', fontsize=18)
        ax.set_ylabel('axis two', fontsize=18)

    def make_notifier(self):
        # return Notifier(None, graphs=[
        #     'loss_history',
        #     {'source': B('images'), 'name': 'images', 'plot_function': self.custom_plotter},
        #     {'source': B('masks'), 'name': 'masks', 'plot_function': self.custom_plotter},
        #     {'source': B('predictions').astype('float32'), 'name': 'predictions', 'plot_function': self.custom_plotter}
        # ])
        # self.visdom = VisdomMonitor(self.config['experiment_id'])
        return Notifier(None, graphs=['loss_history',
            # {'source': , 'name': 'loss', 'plot_function': self.custom_plotter},
            {'source': B('images'), 'name': 'images', 'plot_function': self.custom_plotter},
            {'source': B('masks'), 'name': 'masks', 'plot_function': self.custom_plotter},
            {'source': B('predictions').astype('float32'), 'name': 'predictions', 'plot_function': self.custom_plotter},
        ])

    # def loss_plotter(self, ax=None, container=None, **kwargs):
    #     loss_history = copy.copy(container['data'])
    #     self.visdom.plot(np.arange(len(loss_history)), loss_history, 'loss')

    # def image_plotter(self, ax=None, container=None, **kwargs):
    #     image = container['data'][0]
    #     if image.ndim == 4:
    #         image = image[0]
    #     image = image.transpose(0, 2, 1)
    #     image = np.clip(image, 0, 1) * 255
    #     self.visdom.imshow(image, container['name'])

    def get_train_template(self, **kwargs):
        """ Define the whole training procedure pipeline including data loading, augmentation and model training. """
        return (
            self.load_pipeline(create_masks=True, **kwargs) +
            (self.augmentation_pipeline(**kwargs) if self.config['train/adjust'] else Pipeline()) +
            self.train_pipeline(**kwargs)
        )

    def get_inference_template(self, train_pipeline=None, model_path=None, create_masks=False):
        if train_pipeline is not None:
            test_pipeline = Pipeline().import_model('model', train_pipeline)
        else:
            test_pipeline = Pipeline().load_model('model', TorchModel, 'dynamic', path=model_path)

        test_pipeline += self.load_pipeline(create_masks=create_masks, train=False)

        if create_masks:
            comp = ['images', 'masks']
        else:
            comp = ['images']

        test_pipeline += (
            Pipeline()
            .adaptive_expand(src='images')
            .init_variable('predictions', [])
            .init_variable('target', [])
            .predict_model('model', B('images'), fetches=C('output'), save_to=B('predictions'))
            .adaptive_squeeze(src='predictions')
            .run_later(D('size'))
        )

        smooth_borders = self.config['inference/smooth_borders']
        if smooth_borders:
            if isinstance(smooth_borders, bool):
                step = 0.1
            else:
                step = smooth_borders
            test_pipeline += Pipeline().update(B('predictions') , F(self.smooth_borders)(B('predictions'), step))

        # if create_masks:
        #     test_pipeline += Pipeline().update(V('target', mode='e'), B('masks'))

        test_pipeline += self.metrics_pipeline_template()
        test_pipeline += (Pipeline()
            .adaptive_reshape(src='predictions')
            .update_accumulator(src='predictions', accumulator=C('accumulator'))
        )

        return test_pipeline

    def metrics_pipeline_template(self, n_bins=1000):
        return (Pipeline()
            .init_variable('metric', [])
            .init_variable('semblance_hist', np.zeros(n_bins))
            .compute_attribute(src='images', dst='semblance', attribute='semblance', window=(1, 5, 20))
            .update(V('metric', mode='e'), F(similarity_metric)(B('semblance'), B('predictions')))
            .update(V('semblance_hist'), V('semblance_hist') + F(np.histogram)(B('semblance').flatten(),
                    bins=n_bins, range=(0, 1))[0])
        )

    def make_inference_dataset(self, labels=False, **kwargs):
        config = {**self.config['inference'], **self.config['dataset'], **kwargs}
        inference_cubes = config['cubes']
        width = config['width']
        label_dir = config['label_dir']

        cubes_paths = [self.amplitudes_path(item) for item in inference_cubes]
        dataset = SeismicCubeset(index=cubes_paths)
        if labels:
            dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True, verify=True, bar=False)
        else:
            dataset.load_geometries()
        return dataset

    def parse_locations(self, cubes):
        cubes = cubes.copy()
        if isinstance(cubes, (list, tuple)):
            cubes = {cube: (0, None, None, None) for cube in cubes}
        for cube in cubes:
            cubes[cube] = [cubes[cube]] if isinstance(cubes[cube][0], int) else cubes[cube]
        return cubes

    def make_sampler(self, dataset, ratios=None, threshold=0, **kwargs):
        """ Create sampler for generating locations to train on. """
        crop_shape = self.config['train']['crop_shape']

        if ratios is None:
            ratios = {}

            if len(dataset) > 0:
                for i in range(len(dataset)):
                    faults = dataset.labels[i]
                    fault_area = sum([len(np.unique(faults[j].points)) for j in range(len(faults))])
                    cube_area = np.prod(dataset.geometries[i].cube_shape)
                    ratios[dataset.indices[i]] = fault_area / cube_area
            else:
                ratios[dataset.indices[0]] = 1

        weights = np.array([ratios[i] for i in dataset.indices])
        weights /= weights.sum()
        weights = weights.clip(max=0.3)
        weights = weights.clip(min=0.1)
        weights /= weights.sum()

        self.weights = weights.tolist()

        sampler = SeismicSampler(labels=dataset.labels, crop_shape=crop_shape, mode='fault',
                                 threshold=threshold, proportions=self.weights, **kwargs)

        return sampler

    def make_accumulator(self, geometry, slices, crop_shape, strides, orientation=0, path=None):
        batch_size = self.config['inference']['inference_batch_size']
        if orientation == 1:
            crop_shape = np.array(crop_shape)[[1, 0, 2]]
        name = 'cube_i' if orientation == 0 else 'cube_x'

        grid = RegularGrid(geometry=geometry,
                            threshold=0,
                            orientation=orientation,
                            ranges=slices,
                            batch_size=batch_size,
                            crop_shape=crop_shape, strides=strides)

        accumulator = Accumulator3D.from_aggregation(aggregation='mean',
                                                     origin=grid.origin,
                                                     shape=grid.shape,
                                                     fill_value=0.0,
                                                     name=name,
                                                     path=path)

        return grid, accumulator

    def inference_on_slides(self, train_pipeline=None, model_path=None, create_mask=False, pbar=False, **kwargs):
        config = {**self.config['inference'], **kwargs}
        strides = config['stride'] if isinstance(config['stride'], tuple) else [config['stride']] * 3
        batch_size = config['batch_size']

        crop_shape = config['crop_shape']
        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        self.log(f'Create test pipeline and dataset.')
        dataset = self.make_inference_dataset(create_mask)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_mask)
        inference_pipeline.set_config(config)

        inference_cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.parse_locations(config['cubes']).items()
        }

        outputs = {}
        for cube_idx in dataset.indices:
            outputs[cube_idx] = []
            geometry = dataset.geometries[cube_idx]
            shape = geometry.cube_shape
            for item in inference_cubes[cube_idx]:
                self.log(f'Create prediction for {cube_idx}: {item[1:]}. axis={item[0]}.')
                axis = item[0]
                slices = item[1:]
                if len(slices) != 3:
                    slices = (None, None, None)

                slices = fill_defaults(slices, [[0, i] for i in shape])
                grid, accumulator = self.make_accumulator(geometry, slices, crop_shape, strides, axis)
                ppl = inference_pipeline << dataset << {'sampler': grid, 'accumulator': accumulator}

                ppl.run(n_iters=grid.n_iters, bar=pbar)
                prediction = accumulator.aggregate()

                image = geometry[
                    slices[0][0]:slices[0][1],
                    slices[1][0]:slices[1][1],
                    slices[2][0]:slices[2][1]
                ]
                if axis == 1:
                    image = image.transpose([1, 0, 2])
                    prediction = prediction.transpose([1, 0, 2])

                outputs[cube_idx] += [[slices, image, prediction, np.nanmean(ppl.v('metric')), np.argmax(ppl.v('semblance_hist')) / 1000]]
                if create_mask:
                    outputs[cube_idx][-1] += [dataset.assemble_crops(ppl.v('target'), order=order, fill_value=0).astype('float32')]
        return outputs

    def plot_inference(self, *args, savepath=None, overlap=True, threshold=0.05, each=100, iteration=0, **kwargs):
        if iteration % each == 0:
            results = self.inference_on_slides(*args, **kwargs)
            savepath = os.path.join(self.config['savedir'], savepath) if savepath is not None else None
            for cube in results:
                for item in results[cube]:
                    slices, image, prediction, faults_metric, noise_metric = item[:5]
                    _savepath = None if savepath is None else f'{savepath}_{slices}_{iteration}_{faults_metric:.03f}_{noise_metric:.03f}.png'
                    prediction = prediction[0]
                    prediction[prediction < threshold] = 0
                    if overlap:
                        plot_image([image[0], prediction], separate=False, cmap='gray', figsize=(20, 20), savepath=_savepath)
                    else:
                        plot_image(prediction, figsize=(20, 20), savepath=_savepath)
            return faults_metric, noise_metric
        return None, None

    def inference_on_cube(self, train_pipeline=None, model_path=None, fmt='sgy', save_to=None, prefix=None,
                          tmp='hdf5', bar=True, **kwargs):
        config = {**self.config['inference'], **kwargs}
        strides = config['stride'] if isinstance(config['stride'], tuple) else [config['stride']] * 3
        batch_size = config['batch_size']

        crop_shape = config['crop_shape']
        strides = np.maximum(np.array(crop_shape) * np.array(strides), 1).astype(int)

        self.log(f'Create test pipeline and dataset.')
        dataset = self.make_inference_dataset(create_mask=False)
        inference_pipeline = self.get_inference_template(train_pipeline, model_path, create_masks=False)
        inference_pipeline.set_config(config)

        inference_cubes = {
            self.cube_name_from_path(self.amplitudes_path(k)): v for k, v in self.parse_locations(config['cubes']).items()
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

        prefix = prefix or ''

        for cube_idx in dataset.indices:
            geometry = dataset.geometries[cube_idx]
            shape = geometry.cube_shape
            for item in inference_cubes[cube_idx]:
                self.log(f'Create prediction for {cube_idx}: {item[1:]}. axis={item[0]}.')
                axis = item[0]
                ranges = item[1:]
                if len(ranges) != 3:
                    ranges = (None, None, None)
                locations = [slice(item[0], item[1]) if item else slice(None) for item in ranges]
                ranges = fill_defaults(ranges, [[0, i] for i in shape])

                filenames = {ext: os.path.join(dirname, self.make_filename(prefix, 'x' if axis==1 else 'i', ext))
                             for ext in ['hdf5', 'sgy', 'blosc', 'qblosc', 'npy', 'meta']}

                if fmt in ['hdf5', 'blosc', 'qblosc']:
                    path = filenames[fmt]
                elif fmt in ['sgy', 'segy']:
                    path = filename[tmp]
                elif fmt == 'npy':
                    path = None

                grid, accumulator = self.make_accumulator(geometry, ranges, crop_shape, strides, axis, path)
                ppl = inference_pipeline << dataset << {'sampler': grid, 'accumulator': accumulator}

                ppl.run(n_iters=grid.n_iters, bar=bar)
                prediction = accumulator.aggregate()

                if fmt == 'npy':
                    return prediction
                elif fmt == 'sgy':
                    copyfile(dataset.geometries[0].path_meta, filename['meta'])
                    dataset.geometries[0].make_sgy(
                        path_hdf5=filename[tmp],
                        path_spec=dataset.geometries[0].segy_path.decode('utf-8'),
                        path_segy=filename['sgy'],
                        remove_hdf5=True, zip_result=True, pbar=True
                    )

    def get_model_config(self, name):
        if name == 'UNet':
            return {**self.BASE_MODEL_CONFIG, **self.UNET_CONFIG}
        raise ValueError(f'Unknown model name: {name}')

    def get_model_class(self, name):
        if name == 'UNet':
            return EncoderDecoder
        return TorchModel

    def dump_model(self, model, path):
        model.save(os.path.join(self.config['savedir'], path))

    def amplitudes_path(self, cube):
        ext = self.config['dataset/ext']
        filename = self.config['dataset/path'] + 'CUBE_' + cube + f'/amplitudes_{cube}.{ext}'
        if os.path.exists(filename):
            return filename
        raise ValueError(f"File doesn't exist: {filename}")

    def cube_name_from_alias(self, path):
        return os.path.splitext(self.amplitudes_path(path).split('/')[-1])[0]

    def cube_name_from_path(self, path):
        return os.path.splitext(path.split('/')[-1])[0]

    def create_filename(self, prefix, orientation, ext):
        return (prefix + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_{}.{}').format(orientation, ext)

    @classmethod
    def adjust_shape(cls, crop_shape, angle, scale):
        crop_shape = np.array(crop_shape)
        load_shape = adjust_shape_3d(crop_shape[[1, 2, 0]], angle, scale=scale)
        return (load_shape[2], load_shape[0], load_shape[1])

    def smooth_borders(self, crops, step):
        mask = self.border_smoothing_mask(crops.shape[-3:], step)
        mask = np.expand_dims(mask, axis=0)
        if len(crops.shape) == 5:
            mask = np.expand_dims(mask, axis=0)
        crops = crops * mask
        return crops

    def border_smoothing_mask(self, shape, step):
        mask = np.ones(shape)
        axes = [(1, 2), (0, 2), (0, 1)]
        if isinstance(step, (int, float)):
            step = [step] * 3
        for i in range(len(step)):
            if isinstance(step[i], float):
                step[i] = int(shape[i] * step[i])
            length = shape[i]
            if length >= 2 * step[i]:
                _mask = np.ones(length, dtype='float32')
                _mask[:step[i]] = np.linspace(0, 1, step[i]+1)[1:]
                _mask[:-step[i]-1:-1] = np.linspace(0, 1, step[i]+1)[1:]
                _mask = np.expand_dims(_mask, axes[i])
                mask = mask * _mask
        return mask

    def make_filename(self, prefix, orientation, ext):
        return (prefix + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_{}.{}').format(orientation, ext)

    def skeletonize_faults(self, prediction, axis=0, threshold=0.1, bar=True):
        prediction_cube = SeismicGeometry(prediction) if isinstance(prediction, str) else prediction
        processed_faults = np.zeros(prediction_cube.cube_shape)
        for i in tqdm.tqdm_notebook(range(prediction_cube.cube_shape[axis]), disable=(not bar)):
            slices = [slice(None)] * 2
            slices[axis] = i
            slices = tuple(slices)
            struct = generate_binary_structure(2, 10)

            prediction = prediction_cube.load_slide(i, axis=axis)
            dilation = binary_dilation(prediction > threshold, struct)
            holes = binary_fill_holes(dilation, struct)
            erosion = binary_erosion(holes, generate_binary_structure(2, 1))

            processed_faults[slices] = binary_dilation(skeletonize(erosion, method='lee'))

        return Fault.from_mask(processed_faults, prediction_cube, chunk_size=100, pbar=bar)

# from visdom import Visdom
# import numpy as np

# class VisdomMonitor:
#     def __init__(self, env_name):
#         self.viz = Visdom()
#         self.env = env_name
#         self.plots = {}

#     def plot(self, x, y, var_name):
#         if var_name not in self.plots:
#             if len(x) == 1:
#                 x = [x[0], x[0]]
#                 y = [y[0], y[0]]
#             self.plots[var_name] = self.viz.line(X = x, Y = y, opts=dict(title=var_name, xlabel='Iteration'), env=self.env)
#         else:
#             self.viz.line(X = x, Y = y, win=self.plots[var_name], opts=dict(title=var_name, xlabel='Iteration'), update='replace', env=self.env)

#     def imshow(self, image, var_name):
#         image = image[0].astype('uint8')
#         if var_name not in self.plots:
#             self.plots[var_name] = self.viz.image(image, opts=dict(caption=self.env), env=self.env)
#         else:
#             self.viz.image(image, win=self.plots[var_name], env=self.env, opts=dict(caption=self.env))

def sum_with_axes(array, axes=None):
    if axes is None:
        return array.sum()
    if isinstance(axes, int):
        axes = [axes]
    res = array
    axes = sorted(axes)
    for i, axis in enumerate(axes):
        res = res.sum(axis=axis-i)
    return res

def mean(array, axes=None):
    if axes is None:
        return array.mean()
    if isinstance(axes, int):
        axes = [axes]
    res = array
    axes = sorted(axes)
    for i, axis in enumerate(axes):
        res = res.mean(axis=axis-i)
    return res

def similarity_metric(semblance, masks, threshold=None):
    SHIFTS = [-20, -15, -5, 5, 15, 20]
    if threshold:
        masks = masks > threshold
    if semblance.ndim == 2:
        semblance = np.expand_dims(semblance, axis=0)
    if semblance.ndim == 3:
        semblance = np.expand_dims(semblance, axis=0)

    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=0)
    if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=0)

    res = []
    m = sum_with_axes(masks * (1 - semblance), axes=[1,2,3])
    weights = np.ones((len(SHIFTS), 1))
    weights = weights / weights.sum()
    for i in SHIFTS:
        random_mask = shift(masks, shift=i)
        rm = sum_with_axes(random_mask * (1 - semblance), axes=[1,2,3])
        ratio = m/rm
        res += [np.log(ratio)]
    res = np.stack(res, axis=0)
    res = (res * weights).sum(axis=0)
    res = np.clip(res, -2, 2)
    return res

def shift(array, shift=20):
    result = np.zeros_like(array)
    for i, _array in enumerate(array):
        if shift > 0:
            result[i][:, shift:] = _array[:, :-shift]
        elif shift < 0:
            result[i][:, :shift] = _array[:, -shift:]
        else:
            result[i] = _array
    return result