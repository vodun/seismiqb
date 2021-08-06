""" Custom model for using two separate branches.
TODO: make separate configs for branches
TODO: make it work with N branches and N configs
"""
from torch import nn

from ...batchflow.batchflow.models.torch.layers import ConvBlock
from ...batchflow.batchflow.models.torch import EncoderDecoder
from ...batchflow.batchflow.models.utils import unpack_args


class MyEncoderModule(nn.ModuleDict):
    """ Encoder: create compressed representation of an input by reducing its spatial dimensions. """
    def __init__(self, inputs=None, return_all=True, **kwargs):
        super().__init__()
        self.return_all = return_all
        self._make_modules(inputs, **kwargs)

    def forward(self, inputs):
        """ Docstring. """
        x, y = inputs
        outputs = []
        for letter, (layer_name, layer) in zip(self.layout, self.items()):
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
    """ EncoderDecoder with multiple encoding branches. """
    @classmethod
    def encoder(cls, inputs, **kwargs):
        """ Create encoder either from base model or block args. """
        return MyEncoderModule([inputs[0], inputs[1]], **kwargs)
