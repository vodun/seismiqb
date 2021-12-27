""" Script for running controller notebooks for tests.

The behaviour of tests is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
LOGS_DIR : str
    Path to the directory with test logs (timings, message).
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
OUTPUT_DIR : str
    Path to the directory for saving results and temporary files
    (executed notebooks, logs, data files like cubes, etc.).

And you can manage tests running with parameters:

USE_TMP_OUTPUT_DIR: bool
    Whether to use pytest tmpdir as a workspace.
    If True, then all files are saved in temporary directories.
    If False, then all files are saved in local directories.
REMOVE_OUTDATED_FILES: bool
    Whether to remove outdated files which relate to previous executions.
REMOVE_EXTRA_FILES : bool
    Whether to remove extra files after execution.
    Extra files are temporary files and execution savings that relate to successful tests.
SHOW_MESSAGE : bool
    Whether to show a detailed tests execution message.
SHOW_TEST_ERROR_INFO : bool
    Whether to show error traceback in outputs.
    Notice that it only works with SHOW_MESSAGE = True.
    
Visualizations in saved execution notebooks are controlled with:

FIGSIZE : sequence of two integers
    Figures width and height in inches.
SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.

Text outputs in executed notebooks controlled with:

VERBOSE : bool
    Whether to print information about successful tests during the execution of the cycles.
"""
import os
from datetime import date

from .utils import local_workspace_preparation, run_test_notebook
from ..batchflow.utils_notebook import run_notebook


def test_geometry(
    capsys, tmpdir,
    OUTPUT_DIR=None, USE_TMP_OUTPUT_DIR=True,
    REMOVE_OUTDATED_FILES=True, REMOVE_EXTRA_FILES=True,
    SHOW_MESSAGE=True, SHOW_TEST_ERROR_INFO=True
):
    """ Run SeismicGeometry test notebook.

    This test runs ./notebooks/geometry_test.ipynb test file and show execution message and
    the most important timings for SeismicGeometry tests.

    Under the hood, this notebook create a fake seismic cube, saves it in different data formats
    and for each format run SeismicGeometry tests.
    """
    # Get workspace constants
    DATESTAMP = date.today().strftime("%Y-%m-%d")
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
    LOGS_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_files/')

    if USE_TMP_OUTPUT_DIR:
        # Create tmp workspace
        OUTPUT_DIR = tmpdir.mkdir("notebooks").mkdir("geometry_test_files")
        _ = OUTPUT_DIR.mkdir("notebooks")
        _ = OUTPUT_DIR.mkdir("tmp")

        out_path_ipynb = OUTPUT_DIR.join(f"geometry_test_out_{DATESTAMP}.ipynb")
    else:
        OUTPUT_DIR, out_path_ipynb = local_workspace_preparation(REMOVE_OUTDATED_FILES=REMOVE_OUTDATED_FILES,
                                                                 DATESTAMP=DATESTAMP, TESTS_SCRIPTS_DIR=TESTS_SCRIPTS_DIR,
                                                                 notebook_prefix='geometry', local_output_dir=LOGS_DIR)

    nb_kwargs = {
        # Workspace constants
        'DATESTAMP': DATESTAMP,
        'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
        'LOGS_DIR': LOGS_DIR,
        'OUTPUT_DIR': OUTPUT_DIR,

        # Execution parameters
        'USE_TMP_OUTPUT_DIR': USE_TMP_OUTPUT_DIR,
        'REMOVE_OUTDATED_FILES': REMOVE_OUTDATED_FILES,
        'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,
        'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO
    }


    run_test_notebook(OUTPUT_DIR=OUTPUT_DIR, out_path_ipynb=out_path_ipynb, TESTS_SCRIPTS_DIR=TESTS_SCRIPTS_DIR,
                      notebook_path='notebooks/geometry_test.ipynb', nb_kwargs=nb_kwargs,
                      REMOVE_EXTRA_FILES=REMOVE_EXTRA_FILES, test_name='SeismicGeometry',
                      SHOW_MESSAGE=SHOW_MESSAGE, SHOW_TEST_ERROR_INFO=SHOW_TEST_ERROR_INFO,
                      messages_paths_regexp=['message_*.txt', 'timings_*.json'], capsys=capsys)

def test_charisma(
    capsys, tmpdir,
    OUTPUT_DIR=None, USE_TMP_OUTPUT_DIR=True,
    REMOVE_OUTDATED_FILES=True, REMOVE_EXTRA_FILES=True,
    SHOW_MESSAGE=True, SHOW_TEST_ERROR_INFO=True
):
    """ Run CharismaMixin tests notebook.

    This test runs ./notebooks/charisma_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube (Field), saves it and checks
    matrices savings and loadings in CHARISMA data format.
    
    You can manage the test notebook execution kwargs which relates to cube and horizon with parameters:

    CUBE_SHAPE : sequence of three integers
        Shape of a synthetic cube.
    SEED: int or None
        Seed used for creation of random generator (check out `np.random.default_rng`).
    """
    # Get workspace constants
    DATESTAMP = date.today().strftime("%Y-%m-%d")
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

    if USE_TMP_OUTPUT_DIR:
        # Create tmp workspace
        OUTPUT_DIR = tmpdir.mkdir('notebooks')
        _ = OUTPUT_DIR.mkdir('charisma_tests_files')
        
        out_path_ipynb = OUTPUT_DIR.join(f"charisma_test_out_{DATESTAMP}.ipynb")
    else:
        OUTPUT_DIR, out_path_ipynb = local_workspace_preparation(REMOVE_OUTDATED_FILES=REMOVE_OUTDATED_FILES,
                                                                 DATESTAMP=DATESTAMP, TESTS_SCRIPTS_DIR=TESTS_SCRIPTS_DIR,
                                                                 notebook_prefix='charisma')

    nb_kwargs={
        # Workspace constants
        'DATESTAMP': DATESTAMP,
        'OUTPUT_DIR': OUTPUT_DIR,

        # Execution parameters
        'USE_TMP_OUTPUT_DIR': USE_TMP_OUTPUT_DIR,
        'REMOVE_OUTDATED_FILES': REMOVE_OUTDATED_FILES,
        'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,

         # Data creation parameters
        'CUBE_SHAPE': (100, 100, 100),
        'SEED': 10
    }

    run_test_notebook(OUTPUT_DIR=OUTPUT_DIR, out_path_ipynb=out_path_ipynb, TESTS_SCRIPTS_DIR=TESTS_SCRIPTS_DIR,
                      notebook_path='notebooks/charisma_test.ipynb', nb_kwargs=nb_kwargs,
                      REMOVE_EXTRA_FILES=REMOVE_EXTRA_FILES, test_name='CharismaMixin',
                      SHOW_MESSAGE=SHOW_MESSAGE, SHOW_TEST_ERROR_INFO=SHOW_TEST_ERROR_INFO,
                      capsys=capsys, messages_paths_regexp=[])

def test_horizon(
    capsys, tmpdir,
    OUTPUT_DIR=None, USE_TMP_OUTPUT_DIR=True,
    REMOVE_OUTDATED_FILES=True, REMOVE_EXTRA_FILES=True,
    SHOW_MESSAGE=True, SHOW_TEST_ERROR_INFO=True,
    VERBOSE=True
):
    """ Run Horizon test notebook.

    This test runs ./notebooks/horizon_test.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube with horizon, saves them
    and runs Horizon tests notebooks (base, manipulations, attributes).
    
    You can manage the test notebook execution kwargs which relates to cube and horizon with parameters:

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
    """
    # Get workspace constants
    DATESTAMP = date.today().strftime("%Y-%m-%d")
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

    # Workspace preparation
    if USE_TMP_OUTPUT_DIR:
        # Create tmp workspace
        OUTPUT_DIR = tmpdir.mkdir("notebooks").mkdir("horizon_test_files")
        _ = OUTPUT_DIR.mkdir('tmp')

        out_path_ipynb = OUTPUT_DIR.join(f"horizon_test_out_{DATESTAMP}.ipynb")

    else:        
        OUTPUT_DIR, out_path_ipynb = local_workspace_preparation(REMOVE_OUTDATED_FILES=REMOVE_OUTDATED_FILES,
                                                                 DATESTAMP=DATESTAMP, TESTS_SCRIPTS_DIR=TESTS_SCRIPTS_DIR,
                                                                 notebook_prefix='horizon')

    nb_kwargs={
        # Workspace constants
        'DATESTAMP': DATESTAMP,
        'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
        'OUTPUT_DIR': OUTPUT_DIR,

        # Execution parameters
        'USE_TMP_OUTPUT_DIR': USE_TMP_OUTPUT_DIR,
        'REMOVE_OUTDATED_FILES': REMOVE_OUTDATED_FILES,
        'REMOVE_EXTRA_FILES': REMOVE_EXTRA_FILES,
        'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,

        # Synthetic creation parameters
        'SYNTHETIC_MODE': True,
        'CUBE_PATH': None,
        'HORIZON_PATH': None,
        'CUBE_SHAPE': (500, 500, 200),
        'GRID_SHAPE': (10, 10),
        'SEED': 42,

        # Visualization parameters
        'FIGSIZE': (12, 7),
        'SHOW_FIGURES': False,

        # Output parameters
        'VERBOSE': True
    }
    
    run_test_notebook(OUTPUT_DIR=OUTPUT_DIR, out_path_ipynb=out_path_ipynb, TESTS_SCRIPTS_DIR=TESTS_SCRIPTS_DIR,
                      notebook_path='notebooks/horizon_test.ipynb', nb_kwargs=nb_kwargs,
                      REMOVE_EXTRA_FILES=REMOVE_EXTRA_FILES, test_name='Horizon',
                      SHOW_MESSAGE=SHOW_MESSAGE, SHOW_TEST_ERROR_INFO=SHOW_TEST_ERROR_INFO,
                      capsys=capsys, messages_paths_regexp=['message_*.txt'])
