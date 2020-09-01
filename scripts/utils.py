""" Helper functions for scripts: mainly, parsing supplied CLI and JSON configurations. """
import os
import argparse
import json

import numpy as np
import pandas as pd



def str2bool(string):
    """ Convert string or booleans to True/False. """
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def make_config(description, args, filename, show=True):
    """ Assemble script configuration from command line arguments, JSON file and inputs.

    Parameters
    ----------
    description : str
        Help message for the script.
    args : sequence
        Each element must be in format (argname, description, dtype, default_value).
    filename : str
        Callee filename.
    show : bool
        Whether to print the resulting config.
    """
    # Parse the command line arguments and create convenient help (called by -h)
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawTextHelpFormatter)
    for argname, desc, dtype, default in args:
        if default is None:
            nargs = '*' if isinstance(dtype, list) else '?'
            dtype = dtype[0] if isinstance(dtype, list) else dtype
            parser.add_argument(f'{argname}', metavar=argname, nargs=nargs, type=dtype,
                                default=default, help=desc)
        else:
            nargs = '*' if isinstance(default, list) else '?'
            dtype = dtype[0] if isinstance(dtype, list) else dtype
            parser.add_argument(f'--{argname}', metavar=argname, nargs=nargs, type=dtype,
                                default=default, help=desc)

    config_cl = parser.parse_args()
    config = {key.replace('_', '-') : value for key, value in vars(config_cl).items()}

    # Read the JSON file with the same name as the callee file
    config_path = filename + '.json'
    if os.path.exists(config_path):
        with open(config_path, 'r') as file:
            config_json = json.load(file)
        config = {**config_json, **config}

    # Ask user to input unfilled required arguments
    for argname, desc, dtype, default in args:
        if argname not in config or (default is None and config[argname] == default):
            config[argname] = input(f'Enter {dtype} value for {argname} ({desc}):')

    if show:
        print('\nPASSED ARGUMENTS:')
        for argname, desc, _, _ in args:
            print(f'{argname.upper()} ({desc}) : {config[argname]}')
        print('#'*110, '\n')
    return config


def save_point_cloud(metric, save_path, geometry=None):
    """ Save 2D metric map as a .txt point cloud. Can be opened by GENERAL format reader in geological software. """
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
    """ Make directory, if it does not exists. """
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
