""" Compute quality map(s) for horizon(s). """
import os
import sys
from copy import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import make_config, save_point_cloud, safe_mkdir

sys.path.append('..')
from seismiqb import SeismicGeometry, Horizon, HorizonMetrics
from seismiqb import METRIC_CMAP, enlarge_carcass_metric, plot_image



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
    ('carcass-path', 'path to the horizon in a seismic cube in CHARISMA format', [str], []),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('metrics', 'which metrics to compute', str, ['support_corrs', 'local_corrs']),
    ('add-prefix', 'whether to prepend horizon name to the saved file names', bool, True),
    ('save-files', 'whether to save horizons/carcasses to disk', bool, True),
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
    config = make_config(MSG, ARGS, os.path.basename(__file__).split('.')[0])
    if config['savedir'] == '_placeholder_':
        config['savedir'] = os.path.dirname(config['cube-path'])


    geometry = SeismicGeometry(config['cube-path'])
    safe_mkdir(config['savedir'])

    horizons = [Horizon(path, geometry=geometry) for path in config['horizon-path']]
    carcasses = [Horizon(path, geometry=geometry) for path in config['carcass-path']]
    dataframe = []

    for horizon in horizons:
        print(f'Working with {horizon.name}...')
        hm = HorizonMetrics(horizon)
        prefix = '' if config['add-prefix'] is False else horizon.name + '_'

        row_dict = {
            'horizon': horizon.name,
            'path': os.path.join(*horizon.path.split('/')[-2:]),
            'length': len(horizon),
            'coverage': horizon.coverage,
            'description': 'No description',
        }

        # Evaluate the horizon
        for metric_name in config['metrics']:
            kwargs = copy(LOCAL_KWARGS) if metric_name.startswith('local') else copy(SUPPORT_KWARGS)
            kwargs = {} if metric_name.startswith('insta') else kwargs
            savepath = os.path.join(config['savedir'], f'{prefix}{metric_name}')

            metric = hm.evaluate(metric_name, **kwargs,
                                 plot=True, show_plot=False, title='', figsize=(20, 20),
                                 savepath=savepath + '.png')

            if config['save-txt']:
                save_point_cloud(metric, savepath + '.txt', geometry)
            row_dict[f'horizon_{metric_name}'] = np.nanmean(metric)

        if config['save-files']:
            horizon.dump(os.path.join(config['savedir'], horizon.name), add_height=False)

        # Try to detect the carcass of a horizon and evaluate it
        if carcasses:
            lst = [(carcass, Horizon.check_proximity(horizon, carcass))
                   for carcass in carcasses]
            lst.sort(key=lambda x: abs(x[1].get('mean', 999999)))
            carcass, overlap_info = lst[0]
            if overlap_info['mean'] < 10:
                row_dict = {
                    **row_dict,
                    'carcass_name': carcass.name,
                    'carcass_path': os.path.join(*carcass.path.split('/')[-2:]),
                }


                cm = HorizonMetrics(carcass)
                for metric_name in config['metrics']:
                    kwargs = copy(LOCAL_KWARGS) if metric_name.startswith('local') else copy(SUPPORT_KWARGS)
                    kwargs = {} if metric_name.startswith('insta') else kwargs
                    savepath = os.path.join(config['savedir'], f'{prefix}carcass_{metric_name}')

                    metric = cm.evaluate(metric_name, **kwargs)
                    plot_image(enlarge_carcass_metric(metric, geometry), figsize=(20, 20),
                            cmap=METRIC_CMAP, zmin=-1, zmax=1, fill_color='black',
                            savepath=savepath + '.png')

                    row_dict[f'carcass_{metric_name}'] = np.nanmean(metric)
                if config['save-files']:
                    horizon.dump(os.path.join(config['savedir'], f'{horizon.name}_carcass'), add_height=False)
            else:
                row_dict = {
                    **row_dict,
                    'carcass_name': '',
                    'carcass_path': '',
                    **{metric_name: 0 for metric_name in config['metrics']},
                }

        dataframe.append(row_dict)
        pd.DataFrame(dataframe).to_csv(os.path.join(config['savedir'], 'report.csv'), sep=',', index=False)
