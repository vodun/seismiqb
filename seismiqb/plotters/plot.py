""" Plotter with redefined defaults. """
import batchflow



class plot(batchflow.plot):
    """ Wrapper over `batchflow.plot` with custom defaults.

    Images are displayed transposed, text labels are bigger, ticks labels are displayed from all four image sides.
    """
    __doc__ += '------------------------\n    ' + batchflow.plot.__doc__[1:] # doc inheritance

    COMMON_DEFAULTS = {
        **batchflow.plot.COMMON_DEFAULTS,
        'suptitle_size': 30,
    }

    IMAGE_DEFAULTS = {
        **batchflow.plot.IMAGE_DEFAULTS,
        'labeltop': True,
        'labelright': True,
        'xlabel_size': 22,
        'ylabel_size': 22,
        'transpose': (1, 0, 2)
    }
