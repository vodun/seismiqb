""" Convert seismic amplitude cubes from SEG-Y format to HDF5. """
import os
import sys

from utils import make_config

sys.path.append('..')
from seismiqb import SeismicGeometry



# Help message
MSG = """Convert SEG-Y cube to HDF5.
Input SEG-Y file must have correctly filled `INLINE_3D` and `CROSSLINE_3D` headers.
A lot of various statistics about traces are also inferred and stored in the resulting file,
so this script takes some time.
"""

# Argname, description, dtype, default
ARGS = [
    ('cube-path', 'path to the SEG-Y cube to convert to HDF5', str, None),
]


if __name__ == '__main__':
    import time
    start = time.time()
    config = make_config(MSG, ARGS, os.path.basename(__file__).split('.')[0])

    geometry = SeismicGeometry(
        config['cube-path'],
        headers=SeismicGeometry.HEADERS_POST_FULL,
        index_headers=SeismicGeometry.INDEX_POST,
        collect_stats=True, spatial=True,
    )
    geometry.make_hdf5()
