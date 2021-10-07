""" Script for running notebook with SeismicGeometry tests."""
import glob
import os
from datetime import date
from ..batchflow.utils_notebook import run_notebook


# Constants
DATESTAMP = date.today().strftime("%Y-%m-%d")
DROP_EXTRA_FILES = False # True
PATH_TO_SEISMIQB = os.getenv("PATH_TO_SEISMIQB", '../..')
SHOW_TEST_ERROR_INFO = True
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
TEST_FOLDER = TESTS_SCRIPTS_DIR + 'notebooks/geometry_test_files/'

# Delete old test notebook results
previous_output_files = glob.glob(TESTS_SCRIPTS_DIR + 'notebooks/geometry_test_out_*.ipynb')
for file in previous_output_files:
    os.remove(file)

out_path_ipynb = TESTS_SCRIPTS_DIR + f'notebooks/geometry_test_out_{DATESTAMP}.ipynb'

# Tests execution
exec_info = run_notebook(
    path=TESTS_SCRIPTS_DIR+'notebooks/geometry_test.ipynb',
    nb_kwargs={
        'TEST_FOLDER': TEST_FOLDER,
        'NOTEBOOKS_FOLDER': TESTS_SCRIPTS_DIR+'notebooks/',
        'DATESTAMP': DATESTAMP,
        'DROP_EXTRA_FILES': DROP_EXTRA_FILES,
        'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,
        'PATH_TO_SEISMIQB': PATH_TO_SEISMIQB
    },
    insert_pos=0,
    out_path_ipynb=out_path_ipynb,
    display_links=False
)

# Output message and extra file deleting
if exec_info is True:
    print('Tests were executed correctly.\n')
    if DROP_EXTRA_FILES:
        os.remove(out_path_ipynb)
else:
    print(f'An ERROR occured in cell number {exec_info}:\n{out_path_ipynb}\n')
