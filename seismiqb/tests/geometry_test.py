""" Script for running the controller notebook for SeismicGeometry tests.

The behaviour of the test is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
TEST_DIR : str
    Path to the directory with test results.
    Used for saving and opening log files (timings, message).

And you can manage test running with parameters:

REMOVE_EXTRA_FILES : bool
    Whether to drop files extra files after execution.
    Extra files are temporary files and execution savings that relate to successful tests.
SHOW_MESSAGE : bool
    Whether to show a detailed tests execution message.
SHOW_TEST_ERROR_INFO : bool
    Whether to show error traceback in outputs.
    Notice that it only works with SHOW_MESSAGE = True.
GITHUB_MODE : bool
    Whether to execute tests in GitHub mode.
    If True, then all files are saved in temporary directories.
    If False, then all files are saved in local directories.
"""
from glob import glob
import json
import os
import pprint
from datetime import date

from .utils import extract_traceback
from ..batchflow.utils_notebook import run_notebook

# Workspace constants
DATESTAMP = date.today().strftime("%Y-%m-%d")
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
TEST_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_files/')

# Execution parameters
REMOVE_EXTRA_FILES = True
SHOW_MESSAGE = True
SHOW_TEST_ERROR_INFO = True
GITHUB_MODE = True

def test_geometry(capsys, tmpdir):
    """ Run SeismicGeometry test notebook.

    This test runs ./notebooks/geometry_test.ipynb test file and show execution message and
    the most important timings for SeismicGeometry tests.

    Under the hood, this notebook create a fake seismic cube, saves it in different data formats
    and for each format run SeismicGeometry tests.
    """
    # Delete old test notebook results
    if GITHUB_MODE:
        SAVING_DIR = tmpdir.mkdir("notebooks").mkdir("geometry_test_files")
        out_path_ipynb = SAVING_DIR.join(f"geometry_test_out_{DATESTAMP}.ipynb")

    else:
        # Clear outdatted files
        previous_output_files = glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

        # Path to a new test noteboook result
        SAVING_DIR = TEST_DIR
        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/geometry_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test.ipynb'),
        nb_kwargs={
            # Workspace constants
            'DATESTAMP': DATESTAMP,
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'TEST_DIR': TEST_DIR,
            'SAVING_DIR': SAVING_DIR,

            # Execution parameters
            'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,
            'GITHUB_MODE': GITHUB_MODE
        },
        insert_pos=1,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    if exec_info is True:
        # Open message
        message_path = glob(os.path.join(SAVING_DIR, 'message_*.txt'))[-1]

        with open(message_path, "r", encoding="utf-8") as infile:
            msg = infile.readlines()

        # Open timings
        timings_path = glob(os.path.join(SAVING_DIR, 'timings_*.json'))[-1]

        with open(timings_path, "r", encoding="utf-8") as infile:
            timings = json.load(infile)

    else:
        msg = ['SeismicGeometry tests execution failed.']
        timings= {'state': 'FAIL'}

        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg.append(extract_traceback(path_ipynb=out_path_ipynb))

    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            print('\n'.join(msg))

        pp = pprint.PrettyPrinter()
        pp.pprint(timings)
        print('\n')

        # End of the running message
        if timings['state']=='OK':
            print('Tests for SeismicGeometry were executed successfully.\n')
        else:
            assert False, 'SeismicGeometry tests failed.\n'
