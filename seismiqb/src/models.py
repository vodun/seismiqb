import tensorflow as tf

from ..batchflow.models.tf import EncoderDecoder, ResNet, TFModel
from ..batchflow.models.tf.layers import conv_block, ConvBlock
from ..batchflow.models.utils import unpack_args


class ExtentionModel(EncoderDecoder):
    @classmethod
    def encoder(cls, inputs, name='encoder', **kwargs):
        images, prior_masks = inputs

        order = cls.pop('order', kwargs)
        parallel = kwargs.pop('parallel')
            
        steps, downsample, block_args = cls.pop(['num_stages', 'downsample', 'blocks'], kwargs)
        order = ''.join([item[0] for item in order])

        base_block = block_args.get('base')
        with tf.variable_scope(name):
            x = images
            y = prior_masks
            encoder_outputs = []
            parallel_outputs = []

            for i in range(steps):
                with tf.variable_scope('encoder-'+str(i)):
                    # Make all the args
                    args = {**kwargs, **block_args, **unpack_args(block_args, i, steps)}
                    if downsample:
                        downsample_args = {**kwargs, **downsample, **unpack_args(downsample, i, steps)}

                    for letter in order:
                        if letter == 'b':
                            x = base_block(x, name='block', **args)
                            y = base_block(y, name='p_block', **args)
                            x = tf.add_n([x, y], name='sum')
                        elif letter == 's':
                            parallel_outputs.append(y)
                            encoder_outputs.append(x)
                        elif letter in ['d', 'p']:
                            if downsample.get('layout') is not None:
                                x = conv_block(x, name='downsample', **downsample_args)
                                y = conv_block(y, name='p_downsample', **downsample_args)
                        else:
                            raise ValueError('Unknown letter in order {}, use one of "b", "d", "p", "s"'
                                             .format(letter))
            parallel_outputs.append(y)
            encoder_outputs.append(x)

        return encoder_outputs

    @classmethod
    def head(cls, inputs, targets, name='head', **kwargs):
        kwargs = cls.fill_params('head', **kwargs)

        with tf.variable_scope(name):
            x = TFModel.head(inputs, name, **kwargs)
            channels = 1 # remove hardcode
            args = {**kwargs, **dict(layout='c', kernel_size=1, filters=channels)}
            x = ConvBlock(name='conv1x1', **args)(x)
            x = tf.expand_dims(x, axis=-1, name='expand')
        return x
