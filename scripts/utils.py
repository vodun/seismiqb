import os
import argparse
import json

import numpy as np
import pandas as pd


def make_config(description, args, filename):
    parser = argparse.ArgumentParser(description=description)
    for argname, desc, dtype, default in args:
        if default is None:
            nargs = '*' if isinstance(dtype, list) else '?'
            dtype = dtype[0] if isinstance(dtype, list) else dtype
            parser.add_argument(f'{argname}', metavar=argname, nargs=nargs, type=dtype,
                                default=default, help=desc)
        else:
            nargs = '*' if isinstance(default, list) else '?'
            parser.add_argument(f'--{argname}', metavar=argname, nargs=nargs,
                                default=default, type=dtype, help=desc)

    config_cl = parser.parse_args()
    config = {key.replace('_', '-') : value for key, value in vars(config_cl).items()}

    config_path = filename + '.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config_json = json.load(file)
        config = {**config_json, **config}

    for argname, desc, dtype, default in args:
        if argname not in config or (default is None and config[argname] == default):
            config[argname] = input(f'Enter {dtype} value for {argname} ({desc}):')

    return config


def save_point_cloud(metric, save_path, geometry=None):
    idx_1, idx_2 = np.asarray(~np.isnan(metric)).nonzero()
    points = np.hstack([idx_1.reshape(-1, 1),
                        idx_2.reshape(-1, 1),
                        metric[idx_1, idx_2].reshape(-1, 1)])

    if geometry is not None:
        points[:, 0] += geometry.ilines_offset
        points[:, 1] += geometry.xlines_offset

    df = pd.DataFrame(points, columns=['iline', 'xline', 'metric_value'])
    df.sort_values(['iline', 'xline'], inplace=True)
    df.to_csv(save_path, sep=' ', columns=['iline', 'xline', 'metric_value'],
              index=False, header=False)


def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
