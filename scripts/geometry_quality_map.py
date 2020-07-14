""" !!. """
import os
import sys

from utils import make_config, save_point_cloud, safe_mkdir

sys.path.append('..')
from seismiqb import SeismicGeometry, plot_image

# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the seismic cube in HDF5 format', str, None),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('metrics', 'metrics to compute for quality map creation', str, ['support_hellinger']),
    ('save-txt', 'whether to save point cloud quality map to disk', bool, False),
]


if __name__ == '__main__':
    config = make_config('Create quality map for seismic cube.', ARGS,
                         os.path.basename(__file__).split('.')[0])
    if config['savedir'] == '_placeholder_':
        config['savedir'] = os.path.dirname(config['cube-path'])

    print('\nPASSED ARGUMENTS:')
    for argname, desc, _, _ in ARGS:
        print(f'{argname.upper()} ({desc}) : {config[argname]}')
    print('#'*110, '\n')

    geometry = SeismicGeometry(config['cube-path'])
    geometry.make_quality_map([0.1, 0.15, 0.2, 0.4, 0.5], config['metrics'])

    safe_mkdir(config['savedir'])
    plot_image(geometry.quality_map, cmap='Reds',
               xlabel='INLINE_3D', ylabel='CROSSLINE_3D',
               savepath=os.path.join(config['savedir'], 'quality_map.png'))

    if config['save-txt']:
        save_point_cloud(geometry.quality_map,
                        os.path.join(config['savedir'], 'quality_map.txt'),
                        geometry)
