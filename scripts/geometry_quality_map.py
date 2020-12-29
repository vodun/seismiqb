""" Compute fine map to show local geological hardness of the data. """
import os
import sys
import warnings
warnings.filterwarnings("ignore")

from utils import str2bool, make_config, safe_mkdir

sys.path.append('..')
from seismiqb import SeismicGeometry, plot_image, save_point_cloud



# Help message
MSG = """Compute map of local geological hardness for seismic cube.
Passed cube must be in HDF5 format, e.g. created by `convert_to_hdf5.py` script.
Under the hood, we compare each trace in the cube against 100 reference ones.
As the traces are quite long (up to 3000 elements), we use their compressed representation.
"""

# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the seismic cube in HDF5 format', str, None),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('metrics', 'metrics to compute for quality map creation', str, ['support_hellinger']),
    ('add-prefix', 'whether to prepend cube name to the saved file names', str2bool, True),
    ('save-txt', 'whether to save point cloud of quality map to disk', str2bool, False),
]


if __name__ == '__main__':
    config = make_config(MSG, ARGS, os.path.basename(__file__).split('.')[0])
    if config['savedir'] == '_placeholder_':
        config['savedir'] = os.path.dirname(config['cube-path'])


    geometry = SeismicGeometry(config['cube-path'])
    geometry.make_quality_map([0.1, 0.15, 0.2, 0.4, 0.5], config['metrics'])

    prefix = '' if config['add-prefix'] is False else geometry.short_name + '_'
    safe_mkdir(config['savedir'])
    plot_image(geometry.quality_map, cmap='Reds',
               xlabel='INLINE_3D', ylabel='CROSSLINE_3D', title='',
               savepath=os.path.join(config['savedir'], f'{prefix}quality_map.png'))

    if config['save-txt']:
        save_point_cloud(geometry.quality_map,
                         os.path.join(config['savedir'], f'{prefix}quality_map.txt'),
                         geometry)
