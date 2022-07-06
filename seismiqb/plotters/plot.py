""" Plotter with redefined defaults. """
import batchflow



class plot(batchflow.plot):
    """ Wrapper over original `plot` with custom defaults. """
    def __init__(self, *args, **kwargs):
        kwargs = {'transpose': (1, 0, 2), **kwargs}
        super().__init__(*args, **kwargs)
