""" Build a complete report on the results of carcass interpolation. """
import os
import sys
from copy import copy
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from utils import str2bool, make_config, save_point_cloud, safe_mkdir

sys.path.append('..')
from seismiqb import SeismicGeometry, Horizon, HorizonMetrics
from seismiqb import METRIC_CMAP, enlarge_carcass_metric, plot_image



# Help message
MSG = """Compute a complete quality report on a set of horizons, and, if needed, set of other horizons.
Each of the horizons is matched to exactly one of others, if those provided.
For each of the evaluated surfaces, a number of simple statistics like length and coverage are computed,
as well as metric maps. For carcass-like surfaces, metrics are enlarged for visual clarity.
The report is saved into passed directory and contains a CSV table with numeric results,
metric images, and, if needed, horizons and metric point clouds.
"""

# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the seismic cube in HDF5 format', str, None),
    ('horizon-path', 'path to the horizon in a seismic cube in CHARISMA format', [str], None),
    ('other-path', 'path to the horizon in a seismic cube in CHARISMA format', [str], []),
    ('savedir', 'path to save files to', str, '_placeholder_'),
    ('metrics', 'which metrics to compute', str, ['support_corrs']),
    ('add-prefix', 'whether to prepend horizon name to the saved file names', str2bool, True),
    ('save-files', 'whether to save horizons/carcasses to disk', str2bool, True),
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

    horizons = [Horizon(path, geometry=geometry) for path in config['horizon-path']]
    others = [Horizon(path, geometry=geometry) for path in config['other-path']]
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
            'average_depth': horizon.h_mean,
            'description': 'No description',
        }
        horizon.show(savepath=os.path.join(config['savedir'], f'{prefix}depthmap.png'))

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

        # Try to detect the carcass/prediction of a horizon and evaluate it
        if others:
            lst = [(other, Horizon.check_proximity(horizon, other))
                   for other in others]
            lst.sort(key=lambda x: abs(x[1].get('mean', 999999)))
            other, overlap_info = lst[0]
            if overlap_info['mean'] < 10:
                row_dict = {
                    **row_dict,
                    'other_name': other.name,
                    'other_path': os.path.join(*other.path.split('/')[-2:]),
                    'average_l1': overlap_info['mean'],
                    'other_coverage': other.coverage,
                }
                other_prefix = 'carcass' if other.is_carcass else '@'
                other.show(savepath=os.path.join(config['savedir'], f'{prefix}{other_prefix}_depthmap.png'))

                om = HorizonMetrics(other)
                for metric_name in config['metrics']:
                    kwargs = copy(LOCAL_KWARGS) if metric_name.startswith('local') else copy(SUPPORT_KWARGS)
                    kwargs = {} if metric_name.startswith('insta') else kwargs
                    savepath = os.path.join(config['savedir'], f'{prefix}{other_prefix}_{metric_name}')

                    metric = om.evaluate(metric_name, **kwargs)
                    metric = enlarge_carcass_metric(metric, geometry) if other.is_carcass else metric
                    plot_image(metric, figsize=(20, 20),
                               cmap=METRIC_CMAP, zmin=-1, zmax=1, fill_color='black',
                               xlabel='INLINE_3D', ylabel='CROSSLINE_3D',
                               savepath=savepath + '.png')

                    row_dict[f'{other_prefix}_{metric_name}'] = np.nanmean(metric)
                if config['save-files']:
                    other.dump(os.path.join(config['savedir'], f'{horizon.name}_{other_prefix}'), add_height=False)
            else:
                row_dict = {
                    **row_dict,
                    'other_name': '',
                    'other_path': '',
                    **{metric_name: 0 for metric_name in config['metrics']},
                }

        dataframe.append(row_dict)
        pd.DataFrame(dataframe).to_csv(os.path.join(config['savedir'], 'report.csv'), sep=',', index=False)
