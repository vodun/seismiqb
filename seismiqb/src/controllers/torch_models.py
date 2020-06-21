""" Custom model for using two separate branches.
TODO: make separate configs for branches
TODO: make it work with N branches and N configs
"""
import torch
import torch.nn as nn

from ...batchflow.batchflow.models.torch.layers import ConvBlock
from ...batchflow.batchflow.models.torch import ResBlock, EncoderDecoder
from ...batchflow.batchflow.models.utils import unpack_args



class Dice(nn.Module):
    """ !!. """
    def forward(self, input, target):
        input = torch.sigmoid(input)
        dice_coeff = 2. * (input * target).sum() / (input.sum() + target.sum() + 1e-7)
        return 1 - dice_coeff



MODEL_CONFIG = {
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
    'loss': Dice,
    'optimizer': {'name': 'Adam', 'lr': 0.01,},
    "decay": {'name': 'exp', 'gamma': 0.1},
    }

class MyEncoderModule(nn.ModuleDict):
    """ Encoder: create compressed representation of an input by reducing its spatial dimensions. """
    def __init__(self, inputs=None, return_all=True, **kwargs):
        super().__init__()
        self.return_all = return_all
        self._make_modules(inputs, **kwargs)

    def forward(self, inputs):
        x, y = inputs
        outputs = []
        for letter, (layer_name, layer) in zip(self.layout, self.items()):
            # print('FW', letter, layer_name, x.shape, y.shape)
            if letter in ['b', 'd', 'p']:
                if layer_name.endswith('x'):
                    x = layer(x)
                else:
                    y = layer(y)
                    if letter == 'b':
                        x = x + y
            elif letter in ['s']:
                if layer_name.endswith('x'):
                    outputs.append(x)
        outputs.append(x)
        if self.return_all:
            return outputs
        return outputs[-1]

    def _make_modules(self, inputs, **kwargs):
        num_stages = kwargs.pop('num_stages')
        encoder_layout = ''.join([item[0] for item in kwargs.pop('order')])

        block_args = kwargs.pop('blocks')
        downsample_args = kwargs.pop('downsample')
        self.layout = ''

        for i in range(num_stages):
            for letter in encoder_layout:
                for j, prefix in zip([0, 1], 'xy'):
                    # print('MM', i, letter, prefix, inputs[j].shape)
                    if letter in ['b']:
                        args = {**kwargs, **block_args, **unpack_args(block_args, i, num_stages)}

                        layer = ConvBlock(inputs=inputs[j], **args)
                        inputs[j] = layer(inputs[j])
                        layer_desc = 'block-{}'.format(i)

                    elif letter in ['d', 'p']:
                        args = {**kwargs, **downsample_args, **unpack_args(downsample_args, i, num_stages)}

                        layer = ConvBlock(inputs=inputs[j], **args)
                        inputs[j] = layer(inputs[j])
                        layer_desc = 'downsample-{}'.format(i)

                    elif letter in ['s']:
                        layer = nn.Identity()
                        layer_desc = 'skip-{}'.format(i)
                    else:
                        raise ValueError('Unknown letter in order {}, use one of "b", "d", "p", "s"'
                                         .format(letter))

                    self.update([(layer_desc + prefix, layer)])
                    self.layout += letter


class ExtensionModel(EncoderDecoder):
    @classmethod
    def encoder(cls, inputs, **kwargs):
        """ Create encoder either from base model or block args. """
        return MyEncoderModule([inputs[0], inputs[1]], **kwargs)
