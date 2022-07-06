""" Predefined colormaps. """
import numpy as np

from matplotlib.colors import LinearSegmentedColormap, ListedColormap, ColorConverter
from matplotlib.cm import register_cmap, get_cmap



def clist2cdict(clist):
    """ Convert list of colors to dict of colors valid for `LinearSegmentedColormap`. """
    cdict = {'red': [], 'green': [], 'blue': []}
    domain = np.linspace(0, 1, len(clist))
    for x, color in zip(domain, clist):
        cdict['red'].append([x, color[0], color[0]])
        cdict['green'].append([x, color[1], color[1]])
        cdict['blue'].append([x, color[2], color[2]])
    return cdict


DEPTHS_CMAP = get_cmap('summer_r')
register_cmap('Depths', DEPTHS_CMAP)


METRIC_CDICT = {
    'red': [[0.0, None, 1.0], [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 0.0, None]],
    'green': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 1.0, 1.0], [1.0, 0.5, None]],
    'blue': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, None]]
}
METRIC_CMAP = LinearSegmentedColormap('Metric', METRIC_CDICT)
METRIC_CMAP.set_bad(color='black')
register_cmap(name='Metric', cmap=METRIC_CMAP)


SAMPLER_CMAP = ListedColormap([ColorConverter().to_rgb('blue'),
                                ColorConverter().to_rgb('red'),
                                ColorConverter().to_rgb('purple')])
register_cmap(name='Sampler', cmap=SAMPLER_CMAP)


SEISMIC_CLIST = [
    (0.15, 0.22, 0.48),
    (0.5, 0.7, 0.8),
    (0.9, 0.9, 0.9),
    (0.75, 0.25, 0.25),
    (0.75, 0.0, 0.25)
]
SEISMIC_CMAP = LinearSegmentedColormap('Seismic', clist2cdict(SEISMIC_CLIST))
register_cmap('Seismic', SEISMIC_CMAP)


SEISMIC2_CDICT = {
    'red': [[0.0, None, 0.0], [0.25, 0.5, 0.5], [0.5, 1., 1], [0.75, 0.75, 0.75], [1.0, 1, None]],
    'green': [[0.0, None, 0.0], [0.25, 0.5, 0.5], [0.5, 1., 1], [0.75, 0.25, 0.25], [1.0, 0., None]],
    'blue': [[0.0, None, 1.0], [0.25, 0.5, 0.5], [0.5, 1., 1], [0.75, 0., 0.0], [1.0, 0.0, None]],
}
SEISMIC2_CMAP = LinearSegmentedColormap('Seismic2', SEISMIC2_CDICT)
register_cmap(name='Seismic2', cmap=SEISMIC2_CMAP)
