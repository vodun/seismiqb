""" Plot class inheritance with custom defaults. """

from ..batchflow import plot as batchflow_plot



defaults = {
    'order_axes': (1, 0, 2),
    'scale': 1.5
}

class plot(batchflow_plot.plot):
    """ Wrapper over original `plot` with specific defaults. """
    def __init__(self, *args, **kwargs):
        kwargs = {**defaults, **kwargs}
        super().__init__(*args, **kwargs)
