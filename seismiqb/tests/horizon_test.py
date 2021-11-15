""" Script for running notebook with SeismicGeometry tests."""
import glob
import os
import shutil
from datetime import date

from .utils import extract_traceback
from ..batchflow.utils_notebook import run_notebook

# Constants
# Workspace
DATESTAMP = date.today().strftime("%Y-%m-%d")
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

# Execution
DROP_EXTRA_FILES = True
SHOW_MESSAGE = True
SHOW_TEST_ERROR_INFO = True
GITHUB_MODE = True

def test_horizon(capsys, tmpdir):
    """ Run Horizon test notebook.

    This test runs ./notebooks/horizon_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube with horizon, saves it
    and runs Horizon tests notebooks (base, manipulations, attributes).
    """
    # Delete old test notebook results
    if GITHUB_MODE:
        SAVING_DIR = tmpdir.mkdir("notebooks").mkdir("horizon_test_files")
        out_path_ipynb = SAVING_DIR.join(f"horizon_test_out_{DATESTAMP}.ipynb")

    else:
        # Clear outdatted files
        previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

        # Path to a new test noteboook result
        SAVING_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test_files/')
        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/horizon_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test.ipynb'),
        nb_kwargs={
            # Workspace
            'DATESTAMP': DATESTAMP,
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'SAVING_DIR': SAVING_DIR,

            # Synthetic creation
            'SYNTHETIC_MODE': True,
            'CUBE_PATH': None,
            'HORIZON_PATH': None,
            'CUBE_SHAPE': (500, 500, 200),
            'GRID_SHAPE': (10, 10),
            'NUM_REFLECTIONS': 60,
            'SEED': 42,

            # Visualization
            'FIGSIZE': (12, 7),
            'SHOW_FIGURES': False, # Whether to show additional figures

            # Execution
            'DROP_EXTRA_FILES': DROP_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,
            'GITHUB_MODE': GITHUB_MODE
        },
        insert_pos=1,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    if exec_info is True:
        # Open message
        message_path = glob.glob(os.path.join(SAVING_DIR, 'message_*.txt'))[-1]

        with open(message_path, "r", encoding="utf-8") as infile:
            msg = infile.readlines()
    else:
        msg = ['Horizon tests execution failed.\n']
        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg.append(extract_traceback(path_ipynb=out_path_ipynb))

    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            for line in msg:
                print(line)
        else:
            for line in msg:
                pass


        # End of the running message
        if exec_info is True and line.find('success'):
            print()

            # Clear directory with extra files
            if not GITHUB_MODE and DROP_EXTRA_FILES:
                try:
                    shutil.rmtree(SAVING_DIR)
                except OSError as e:
                    print(f"Can't delete the directory {SAVING_DIR} : {e.strerror}")

        else:
            assert False, 'Horizon tests failed.\n'
