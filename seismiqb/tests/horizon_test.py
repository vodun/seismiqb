""" Script for running the controller notebook for Horizon tests.

The behaviour of the test is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
SAVING_DIR : str
    Path to the directory for saving results and temporary files
    (executed notebooks, logs, data files like cubes, horizons etc.).

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

You can also manage notebook execution kwargs which relates to cube and horizon for the test:

SYNTHETIC_MODE : bool
    Whether to create a synthetic data (cube and horizon) or use existed, provided by paths.
CUBE_PATH : str or None
    Path to an existed seismic cube.
    Notice that it is only used with SYNTHETIC_MODE = False.
HORIZON_PATH : str or None
    Path to an existed seismic horizon.
    Notice that it is only used with SYNTHETIC_MODE = False.
CUBE_SHAPE : sequence of three integers
    Shape of a synthetic cube.
GRID_SHAPE: sequence of two integers
    Sets the shape of grid of support points for surfaces' interpolation (surfaces represent horizons).
SEED: int or None
    Seed used for creation of random generator (check out `np.random.default_rng`).

Visualizations in saved execution notebooks are controlled with:

FIGSIZE : sequence of two integers
    Figures width and height in inches.
SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.
"""
from glob import glob
import os
from shutil import rmtree
from datetime import date

from .utils import extract_traceback
from ..batchflow.utils_notebook import run_notebook

# Workspace constants
DATESTAMP = date.today().strftime("%Y-%m-%d")
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

# Execution parameters
REMOVE_EXTRA_FILES = True
SHOW_MESSAGE = True
SHOW_TEST_ERROR_INFO = True
GITHUB_MODE = True

def test_horizon(capsys, tmpdir):
    """ Run Horizon test notebook.

    This test runs ./notebooks/horizon_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube with horizon, saves them
    and runs Horizon tests notebooks (base, manipulations, attributes).
    """
    # Delete old test notebook results
    if GITHUB_MODE:
        SAVING_DIR = tmpdir.mkdir("notebooks").mkdir("horizon_test_files")
        out_path_ipynb = SAVING_DIR.join(f"horizon_test_out_{DATESTAMP}.ipynb")

    else:
        # Clear outdatted files
        previous_output_files = glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

        # Path to a new test noteboook result
        SAVING_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test_files/')
        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/horizon_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/horizon_test.ipynb'),
        nb_kwargs={
            # Workspace constants
            'DATESTAMP': DATESTAMP,
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'SAVING_DIR': SAVING_DIR,

            # Synthetic creation parameters
            'SYNTHETIC_MODE': True,
            'CUBE_PATH': None,
            'HORIZON_PATH': None,
            'CUBE_SHAPE': (500, 500, 200),
            'GRID_SHAPE': (10, 10),
            'SEED': 42,

            # Visualization parameters
            'FIGSIZE': (12, 7),
            'SHOW_FIGURES': False, # Whether to show additional figures

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

    else:
        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg = extract_traceback(path_ipynb=out_path_ipynb)

        msg.append('Horizon tests execution failed.')

    last_msg_line = msg[-1]

    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            print('\n'.join(msg))

        # End of the running message
        if exec_info is True and last_msg_line.find('fail')==-1:
            print()

            # Clear directory with extra files
            if not GITHUB_MODE and REMOVE_EXTRA_FILES:
                try:
                    rmtree(SAVING_DIR)
                except OSError as e:
                    print(f"Can't delete the directory {SAVING_DIR} : {e.strerror}")

        else:
            assert False, 'Horizon tests failed.\n'
