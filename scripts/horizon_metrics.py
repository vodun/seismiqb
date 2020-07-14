""" !!. """
import os
import sys
from copy import copy
from glob import glob

from utils import make_config, save_point_cloud, safe_mkdir

sys.path.append('..')
from seismiqb import SeismicGeometry, Horizon, HorizonMetrics, plot_image



# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the seismic cube in HDF5 format', str, None),
    ('horizon-path', 'path to the horizon in a seismic cube in CHARISMA format', [str], None),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('metrics', 'which metrics to compute', str, ['support_corrs', 'local_corrs']),
    ('add-prefix', 'whether to prepend horizon name to the saved file names', bool, True),
    ('save-txt', 'whether to save point cloud metrics to disk', bool, False),
]


# Parameters for `local` metrics like `local_correlation`
LOCAL_KWARGS = {
    'agg': None,
    'kernel_size': 7,
    'reduce_func': 'mean',
}

# Parameters for `support` metrics like `support_corrs`
SUPPORT_KWARGS = {
    'agg': 'nanmean',
    'supports': 100,
}


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
    safe_mkdir(config['savedir'])

    for horizon_path in (config['horizon-path']):
        horizon = Horizon(horizon_path, geometry=geometry)
        hm = HorizonMetrics(horizon)

        prefix = '' if config['add-prefix'] is False else horizon.name + '_'

        for metric_name in config['metrics']:
            kwargs = copy(LOCAL_KWARGS) if metric_name.startswith('local') else copy(SUPPORT_KWARGS)
            kwargs = {} if metric_name.startswith('insta') else kwargs
            savepath = os.path.join(config['savedir'], f'{prefix}{metric_name}')

            metric = hm.evaluate(metric_name, **kwargs,
                                 plot=True, show_plot=False, plot_kwargs={'figsize': (20, 20)},
                                 savepath=savepath + '.png')
            if config['save-txt']:
                save_point_cloud(metric, savepath + '.txt', geometry)
