""" A convenient holder for faults detection steps:
    - creating dataset with desired properties
    - training a model
    - making an inference on selected data
"""
import os
import glob
import datetime

import numpy as np
import torch

from ...batchflow import Config, Pipeline
from ...batchflow import B, C, D, P, R, V
from ...batchflow.models.torch import TorchModel, ResBlock
from .base import BaseController
from ..cubeset import SeismicCubeset
from ..fault import Fault
from ..layers import InputLayer

class FaultController(BaseController):
    DEFAULTS = Config({
        **BaseController.DEFAULTS,
        # Data
        'dataset': {
            'train_cubes': [],
            'transposed_cubes': [],
            'label_dir': '/INPUTS/FAULTS/NPY_WIDTH_{}/*',
            'width': 3,
        },

        # Loading parameters
        'load_crop_shape':  None,
        'adjusted_crop_shape': None,

        # Model parameters
        'train': {
            # Augmentation parameters
            'angle': 25,
            'scale': (0.7, 1.5),
            'crop_shape': [1, 128, 512],
            'filters': [64, 96, 128, 192, 256],
            'stats': 'item',
            'phase': True,
            'continuous_phase': False,
            'model_class': TorchModel,
            'model': 'UNet',
            'loss': 'bce',
            'output': 'sigmoid',
            'slicing': 'native',
            'stats': 'item',
        },

        'inference': {
            'crop_shape': [1, 128, 512],
            'inference_batch_size': 32,
            'inference_chunk_shape': (100, None, None),
            'smooth_borders': False,
            'stride': 0.5,
            'orientation': 'ilines',
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
            'continuous': C('continuous_phase')
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

    def __init__(self, config=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = config or {}
        self.config = {**self.DEFAULTS, **config, **kwargs}

    def make_dataset(self, ratios=None, **kwargs):
        config = {**self.config['dataset'], **kwargs}
        width = config['width']
        label_dir = config['label_dir']
        paths = [amplitudes_path(item) for item in config['train_cubes']]

        dataset = SeismicCubeset(index=paths)
        dataset.load(label_dir=label_dir.format(width), labels_class=Fault, transform=True, verify=True)
        dataset.modify_sampler(dst='train_sampler', finish=True, low=0.0, high=1.0)

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

        dataset.create_sampler(p=list(weights))
        dataset.modify_sampler(dst='train_sampler', finish=True, low=0.0, high=1.0)

        return dataset

    def load_pipeline(self, create_masks=True, train=True):
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
        shape = {self._cube_name_from_path(k): C('crop_shape') for k in self.cubes_paths}
        shape.update({self._cube_name_from_path(k): C('crop_shape')[[1, 0, 2]] for k in self.transpose_paths})

        if train:
            ppl = Pipeline().make_locations(points=D('train_sampler')(C('batch_size')), shape=shape, side_view=C('side_view'))
        else:
            ppl = Pipeline().make_locations(points=D('grid_gen')(), shape=C('test_load_shape'))

        ppl += Pipeline().load_cubes(dst='images', slicing=C('slicing'))

        if create_masks:
            ppl +=  Pipeline().create_masks(dst='masks')
            components = ['images', 'masks']
        else:
            components = ['images']

        shape = C('crop_shape') if train else C('adjusted_crop_shape')
        ppl += (Pipeline()
            .adaptive_reshape(src=components, shape=shape)
            .normalize(mode='q', stats=C('stats'), src='images')
        )
        return ppl

    def augmentation_pipeline(self):
        return (Pipeline()
            .transpose(src=['images', 'masks'], order=(1, 2, 0))
            .flip(axis=1, src=['images', 'masks'], seed=P(R('uniform', 0, 1)), p=0.3)
            .additive_noise(scale=0.005, src='images', dst='images', p=0.3)
            .rotate(angle=P(R('uniform', -C('angle'), C('angle'))), src=['images', 'masks'], p=0.3)
            .scale_2d(scale=P(R('uniform', C('scale')[0], C('scale')[1])), src=['images', 'masks'], p=0.3)
            .transpose(src=['images', 'masks'], order=(2, 0, 1))
            .central_crop(C('crop_shape'), src=['images', 'masks'])
            .cutout_2d(src=['images', 'masks'], patch_shape=np.array((1, 40, 40)), n=3, p=0.2)
        )

    def train_pipeline(self):
        model_class = F(self._model_class)(C('model'))
        model_config = F(self._get_model_config)(C('model'))
        return (Pipeline()
            .init_variable('loss_history', [])
            .init_model('dynamic', model_class, 'model', model_config)
            .add_channels(src=['images', 'masks'])
            .train_model('model',
                         fetches=['loss', C('output')],
                         images=B('images'),
                         masks=B('masks'),
                         save_to=[V('loss_history', mode='w'), B('predictions')])
        )

    def get_inference_template(self, train_pipeline=None, model_path=None, create_masks=False, smooth_borders=False):
        if train_pipeline:
            test_pipeline = Pipeline().import_model('model', train_pipeline)
        else:
            test_pipeline = Pipeline().load_model(mode='dynamic', model_class=TorchModel, name='model', path=model_path)

        test_pipeline += self.load_pipeline(create_masks=create_masks, train=False)

        if create_masks:
            comp = ['images', 'masks']
        else:
            comp = ['images']

        test_pipeline += (
            Pipeline()
            .adaptive_reshape(src=comp, shape=C('crop_shape'))
            .add_channels(src=comp)
            .init_variable('predictions', [])
            .init_variable('target', [])
            .predict_model('model', B('images'), fetches=C('output'), save_to=B('predictions'))
            .run_later(D('size'))
        )

        if smooth_borders:
            if isinstance(smooth_borders, bool):
                step = 0.1
            else:
                step = smooth_borders
            test_pipeline += Pipeline().update(B('predictions') , F(self.smooth_borders)(B('predictions'), step))

        if create_masks:
            test_pipeline += Pipeline().update(V('target', mode='e'), B('masks'))
        test_pipeline += Pipeline().update(V('predictions', mode='e'), B('predictions'))
        return test_pipeline

    def inference(self, dataset, model, **kwargs):
        """ Make inference on a supplied dataset with a provided model.

        Works by making inference on chunks, splitted into crops.
        Resulting predictions (horizons) are stitched together.
        """
        # Prepare parameters
        config = Config({**self.config['inference'], **kwargs})
        orientation = config.pop('orientation')
        self.log(f'Starting {orientation} inference')

        # Log: pipeline_config to a file
        self.log_to_file(pformat(config.config, depth=2), '末 inference_config.txt')

        # Start resource tracking
        if self.monitor:
            monitor = Monitor(['uss', 'gpu', 'gpu_memory'], frequency=0.5, gpu_list=self.gpu_list)
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

    def get_model_config(self, name):
        if name == 'UNet':
            return {**self.BASE_MODEL_CONFIG, **self.UNET_CONFIG}
        raise ValueError(f'Unknown model name: {name}')

    def get_model_class(self, name):
        if name == 'UNet':
            return EncoderDecoder
        return TorchModel

def cube_name_from_alias(path):
    return os.path.splitext(amplitudes_path(path).split('/')[-1])[0]

def cube_name_from_path(path):
    return os.path.splitext(path.split('/')[-1])[0]

def amplitudes_path(cube):
    return glob.glob(DATA_PATH + 'CUBE_' + cube + '/amplitudes*.hdf5')[0]

def create_filename(self, prefix, orientation, ext):
    return (prefix + datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + '_{}.{}').format(orientation, ext)