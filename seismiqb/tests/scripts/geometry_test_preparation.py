""" Data preparation for tests for SeismicGeometry class. """
import os
import sys

import numpy as np

sys.path.append('../../..')
from seismiqb import SeismicGeometry
from seismiqb.src.geometry import export

# Constants and parameters:
FORMATS = ['hdf5', 'qhdf5', 'blosc', 'qblosc']

def run_preparation(CUBE_NAME, TEST_FOLDER, DATESTAMP, CUBE_SHAPE=(1000, 200, 400), SEED=42):
    CUBE_PATH = TEST_FOLDER + 'tmp/' + CUBE_NAME

    # Create and save a fake cube
    rng = np.random.default_rng(SEED)
    data_array = rng.normal(0, 1000, CUBE_SHAPE).astype(np.float32)
    
    with open(TEST_FOLDER + f'tmp/test_array_{DATESTAMP}.npy', 'wb') as outfile:
        np.save(outfile, data_array)
        
    export.make_segy_from_array(array=data_array, path_segy=CUBE_PATH, zip_segy=False, sample_rate=2., delay=50, pbar='t')

    # Check the cube in SGY:
    geometry_sgy = SeismicGeometry(path=CUBE_PATH, process=True, collect_stats=True, spatial=True, pbar='t')
    
    # Check data loading
    _ = SeismicGeometry(
        path=CUBE_PATH,
        headers=SeismicGeometry.HEADERS_POST_FULL,
        index_headers=SeismicGeometry.INDEX_CDP
    )
    
    _ = SeismicGeometry(
        path=CUBE_PATH,
        headers=SeismicGeometry.HEADERS_POST_FULL,
        index_headers=SeismicGeometry.INDEX_POST
    )
    
    # Convert the cube into other data formats
    for f in FORMATS:
        _ = geometry_sgy.convert(format=f, quantize=False, store_meta=False, pbar='t')
