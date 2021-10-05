""" Script for running tests for SeismicGeometry class. """
import os
import sys
import shutil
from datetime import date

import glob
import json
import numpy as np
from tqdm import tqdm

sys.path.append('../../..')
from seismiqb import SeismicGeometry
from seismiqb.tests.scripts.geometry_test_data_format import run_tests
from seismiqb.tests.scripts.geometry_test_preparation import run_preparation

DATESTAMP = date.today().strftime("%Y-%m-%d")

# Constants and parameters:
TEST_FOLDER = '../geometry_test_files/'
CUBE_NAME = f'test_cube_{DATESTAMP}.sgy'
FORMATS = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']
FIGSIZE = (12, 6)
N_SLIDE = 1000
N_CROP = 300
SEED = 42

# Storage preparation:
# The `tmp` dir contains cube files: cube in different formats and meta
# The `notebooks` dir contains notebooks results (notebooks copies with outputs)
if os.path.exists(TEST_FOLDER + 'tmp/'):
    shutil.rmtree(TEST_FOLDER + 'tmp/')
os.makedirs(TEST_FOLDER + 'tmp/')

# if previous run failed than we need to delete corresponding timings
failed_timings_file = glob.glob(TEST_FOLDER + 'timings*fail*.json')

if failed_timings_file:
    os.remove(failed_timings_file[0])

DROP_EXTRA_FILES = True # drop files reffering to successful tests

msg = '\n' + DATESTAMP + '\n\n'

# Run the script with preparation for tests.
# It contains: data creation, data loading checking and cube conversion into different formats
run_preparation(CUBE_NAME=CUBE_NAME, TEST_FOLDER=TEST_FOLDER, DATESTAMP=DATESTAMP, CUBE_SHAPE=(1000, 200, 400), SEED=42)

msg += 'Data was successfully prepared.\n'
print(msg)

# Run the test script for the cube in each data format.
# It contains: checking data; attributes, slides, crops loading test, data loading timings and visualization tests.
with open(TEST_FOLDER + f'tmp/test_array_{DATESTAMP}.npy', 'rb') as infile:
    data_array = np.load(infile)

timings_file = glob.glob(TEST_FOLDER + 'timings*.json')[0]
with open(timings_file, "r", encoding="utf-8") as infile:
    standard_timings = json.load(infile)

timings = {}
all_OK = True

for f in tqdm(FORMATS):
    try:
        current_message = ''

        CURRENT_CUBE_NAME = CUBE_NAME.replace('sgy', f)
        CUBE_PATH = TEST_FOLDER + 'tmp/' + CURRENT_CUBE_NAME

        geometry = SeismicGeometry(CUBE_PATH)
        expected_timings = standard_timings[geometry.format]

        run_tests(geometry=geometry, data_array=data_array,
                  test_folder=TEST_FOLDER, expected_timings=expected_timings,
                  figsize=FIGSIZE, n_slide=N_SLIDE, n_crop=N_CROP, seed=SEED, datestamp=DATESTAMP)

        # Saving logs
        with open(TEST_FOLDER + f'tmp/timings_{f}_{DATESTAMP}.json', "r", encoding="utf-8") as infile:
            timings.update(json.load(infile))
            current_message += f'Tests for {f.upper()} cube were executed correctly.\n'
    except Exception as exc_inst:
        all_OK = False
        current_message += f'An ERROR occured in {f.upper()} tests.\n'

    print(current_message)

# Dump timings and remove extra files
if all_OK:
    timings['state'] = 'OK'

    if DROP_EXTRA_FILES:
        timings_file = glob.glob(TEST_FOLDER + 'timings*.json')[0]
        os.remove(timings_file)

        shutil.rmtree(TEST_FOLDER + 'tmp/')

    with open(TEST_FOLDER + f'timings_{DATESTAMP}.json', "w", encoding="utf-8") as outfile:
        json.dump(timings, outfile)
else:
    timings['state'] = 'FAIL'

    if DROP_EXTRA_FILES:
        timings_files = glob.glob(TEST_FOLDER + 'tmp/timings*')
        for file_name in timings_files:
            os.remove(file_name)

    with open(TEST_FOLDER + f'timings_fail_{DATESTAMP}.json', "w", encoding="utf-8") as outfile:
        json.dump(timings, outfile)
