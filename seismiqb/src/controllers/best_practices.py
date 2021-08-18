""" Collection of good architectures for the tasks of horizon detection. """
import torch
from torch import nn
from torch.nn import functional as F

from ...batchflow.batchflow.models.torch import ResBlock
from ...batchflow.models.torch.losses.binary import Dice
from ..utils.layers import DepthSoftmax


MODEL_CONFIG = {
    # Model layout
    'initial_block': {
        'base_block': ResBlock,
        'filters': 16,
        'kernel_size': 5,
        'downsample': False,
        'attention': 'scse'
    },

    'body/encoder': {
        'num_stages': 4,
        'order': 'sbd',
        'blocks': {
            'base': ResBlock,
            'n_reps': 1,
            'filters': [32, 64, 128, 256],
            'attention': 'scse',
        },
    },
    'body/embedding': {
        'base': ResBlock,
        'n_reps': 1,
        'filters': 256,
        'attention': 'scse',
    },
    'body/decoder': {
        'num_stages': 4,
        'upsample': {
            'layout': 'tna',
            'kernel_size': 2,
        },
        'blocks': {
            'base': ResBlock,
            'filters': [128, 64, 32, 16],
            'attention': 'scse',
        },
    },

    'head': {
        'base_block': ResBlock,
        'filters': [16, 8],
        'attention': 'scse'
    },

    'output': 'sigmoid',
    # Train configuration
    'loss': 'bdice',
    'optimizer': {'name': 'Adam', 'lr': 0.01,},
    'decay': {'name': 'exp', 'gamma': 0.1, 'frequency': 150},
    'microbatch': 8,
    }

MODEL_CONFIG_DETECTION = {**MODEL_CONFIG}

MODEL_CONFIG_EXTENSION = {
    **MODEL_CONFIG,
    'initial_block': {},
    'optimizer': {'name': 'Adam', 'lr': 0.005,},
    'microbatch': 64,
    'order': ['initial_block', 'body', 'head', ('head_2', 'head_2', 'head')],
    'head/attention': None,
    'head/classes': 8,
    'head_2': {
        'layout': 'ca',
        'classes': 1,
        'filters': 1,
        'activation': DepthSoftmax
    },
    'loss': Dice(apply_sigmoid=False)
}
MODEL_CONFIG_ENHANCE = {
    **MODEL_CONFIG,
    'initial_block': {},
    'optimizer': {'name': 'Adam', 'lr': 0.005,}
}
