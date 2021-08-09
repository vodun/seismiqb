""" Collection of good architectures for the tasks of horizon detection. """
import torch
from torch import nn
from torch.nn import functional as F

from ...batchflow.batchflow.models.torch import ResBlock
from ...batchflow.models.torch.losses.binary import Dice


class DepthSoftmax(nn.Module):
    """ Softmax activation for depth dimension.

    Parameters
    ----------
    width : int
        The predicted horizon width. Default is 3.
    """
    def __init__(self, width=3):
        super().__init__()
        self.width_weights = torch.ones((1, 1, 1, width))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        x = torch.nn.functional.softmax(x, dim=-1)
        width_weights = self.width_weights.to(device=x.device, dtype=x.dtype)
        x = F.conv2d(x, width_weights, padding=(0, 1))
        return x.float()

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
