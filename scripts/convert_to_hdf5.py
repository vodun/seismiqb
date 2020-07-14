""" !!. """
import os
import sys

from utils import make_config

sys.path.append('..')
from seismiqb import SeismicGeometry


# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the SEG-Y cube to convert to HDF5', str, None),
]


if __name__ == '__main__':
    config = make_config('Convert SEG-Y cube to a HDF5 one.', ARGS,
                         os.path.basename(__file__).split('.')[0])


    print('\nPASSED ARGUMENTS:')
    for argname, desc, _, _ in ARGS:
        print(f'{argname.upper()} ({desc}) : {config[argname]}')
    print('#'*110, '\n')

    geometry = SeismicGeometry(
        config['cube-path'],
        headers=SeismicGeometry.HEADERS_POST_FULL,
        index_headers=SeismicGeometry.INDEX_POST,
        collect_stats=True, spatial=True,
    )
    geometry.make_hdf5()
