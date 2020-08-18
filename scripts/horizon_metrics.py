""" Compute quality map(s) for horizon(s). """
import os
import sys
from copy import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from utils import str2bool, make_config, save_point_cloud, safe_mkdir

sys.path.append('..')
from seismiqb import SeismicGeometry, Horizon, HorizonMetrics



# Help message
MSG = """Compute quality map(s) for a given horizon(s).
Under the hood, an array of amplitudes is cut from the cube along the horizon in a fixed window.
After that, `local` metrics are computed by comparing each trace to its neighbours in
a square window (e.g. in a 9x9 square) by using a fixed function (e.g. correlation coefficient).
`support` metrics use multiple reference traces to compare every other one to them.
"""

# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the seismic cube in HDF5 format', str, None),
    ('horizon-path', 'path to the horizon in a seismic cube in CHARISMA format', [str], None),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('metrics', 'which metrics to compute', str, ['support_corrs', 'local_corrs']),
    ('add-prefix', 'whether to prepend horizon name to the saved file names', str2bool, True),
    ('save-txt', 'whether to save point cloud of metrics to disk', str2bool, False),
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
    config = make_config(MSG, ARGS, os.path.basename(__file__).split('.')[0])
    if config['savedir'] == '_placeholder_':
        config['savedir'] = os.path.dirname(config['cube-path'])


    geometry = SeismicGeometry(config['cube-path'])
    safe_mkdir(config['savedir'])

    for horizon_path in config['horizon-path']:
        horizon = Horizon(horizon_path, geometry=geometry)
        hm = HorizonMetrics(horizon)

        prefix = '' if config['add-prefix'] is False else horizon.name + '_'
        with open(os.path.join(config['savedir'], f'{prefix}metrics_info.txt'), 'w') as result_txt:
            _ = horizon.evaluate(compute_metric=False, printer=lambda msg: print(msg, file=result_txt))

            for metric_name in config['metrics']:
                kwargs = copy(LOCAL_KWARGS) if metric_name.startswith('local') else copy(SUPPORT_KWARGS)
                kwargs = {} if metric_name.startswith('insta') else kwargs
                savepath = os.path.join(config['savedir'], f'{prefix}{metric_name}')

                metric = hm.evaluate(metric_name, **kwargs,
                                     plot=True, show_plot=False, plot_kwargs={'figsize': (20, 20)},
                                     savepath=savepath + '.png')
                if config['save-txt']:
                    save_point_cloud(metric, savepath + '.txt', geometry)
                print(f'{metric_name} avg value: {""*20} {np.nanmean(metric):5.5}', file=result_txt)
