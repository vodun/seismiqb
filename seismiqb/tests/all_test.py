""" Test script for running controller notebooks for tests.

The behavior of tests is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
TESTS_ROOT_DIR : str
    Path to the directory for saving results and temporary files for all tests
    (executed notebooks, logs, data files like cubes, etc.).
TEST_OUTPUTS : str or iterable of str
    List of notebook locals names that return to output.

And you can manage tests running with parameters:

REMOVE_OUTDATED_FILES: bool
    Whether to remove outdated files which relate to previous executions.
REMOVE_EXTRA_FILES : bool
    Whether to remove extra files after execution.
    Extra files are temporary files and execution saved files that relate to successful tests.
SHOW_MESSAGE : bool
    Whether to show a detailed tests execution message.
SHOW_TEST_ERROR_INFO : bool
    Whether to show error traceback in outputs.
    Notice that it only works with SHOW_MESSAGE = True.
SAVE_LOGS_REG_EXP : list of str
    A list of regular expressions for files which should be saved after a test execution.

Outputs in saved execution notebooks are controlled with:

SCALE : int
    Figures scale.
SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.
VERBOSE : bool
    Whether to print information about successful tests during the execution of the cycles.
"""
import os
from datetime import date
import json
import re
import shutil
import tempfile
import pytest

from .run_notebook import run_notebook


# Initialize base tests variables
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

common_params = {
    # Workspace constants
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'TESTS_SCRIPTS_DIR': TESTS_SCRIPTS_DIR,
    'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),

    # Execution parameters
    'REMOVE_OUTDATED_FILES': os.getenv('SEISMIQB_TEST_REMOVE_OUTDATED_FILES') or True,
    'REMOVE_EXTRA_FILES': os.getenv('SEISMIQB_TEST_REMOVE_EXTRA_FILES') or True,
    'SHOW_MESSAGE': os.getenv('SEISMIQB_TEST_SHOW_MESSAGE') or True,
    'SHOW_TEST_ERROR_INFO': os.getenv('SEISMIQB_TEST_SHOW_ERROR_INFO') or True,

    # Visualization parameters
    'SCALE': os.getenv('SEISMIQB_TEST_SCALE') or 1,
    'SHOW_FIGURES': os.getenv('SEISMIQB_TEST_SHOW_FIGURES') or False,

    # Output parameters
    'VERBOSE': os.getenv('SEISMIQB_TEST_VERBOSE') or True,
}


# Initialize tests configs
geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']
notebooks_params = (
    # (notebook filename, test params)
    # Note: params for each notebook in the test will be saved for it and for next notebooks in the test

    # CharismaMixin test
    ('charisma_test', {}),

    # SeismicGeometry test
    ('geometry_test_preparation', {'FORMATS': geometry_formats}),
    *[('geometry_test_data_format', {'TEST_OUTPUTS': ['timings'], 'FORMAT': f}) for f in geometry_formats],

    # Horizon test
    ('horizon_test_preparation', {}),
    ('horizon_test_base', {}),
    ('horizon_test_attributes', {}),
    ('horizon_test_manipulations', {}),
    ('horizon_test_extraction', {'TEST_OUTPUTS': ['message']})
)

# Create directory for temporary files and results
common_params['TESTS_ROOT_DIR'] = tempfile.mkdtemp(prefix='tests_root_dir_', dir='./')
pytest.failed = False


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys):
    """..!!.."""
    filename, params = notebook_kwargs
    config = params.copy()
    _ = config.pop('TEST_OUTPUTS', None)
    params.update(common_params)

    # Run test notebook
    prepare_suffix = lambda v: re.sub(r'[^\w^ ]', '', v).replace(' ', '_') # remove symbols and replace spaces
    suffix = "_".join(f"{prepare_suffix(str(v))}" for v in config.values())

    path_ipynb = os.path.join(params['NOTEBOOKS_DIR'], f"{filename}.ipynb")
    out_path_ipynb = os.path.join(params['TESTS_ROOT_DIR'], f"{filename}_out_{suffix}_{params['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=params, outputs=params.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    pytest.failed = pytest.failed or exec_res['failed']

    # Clear extra files
    if not exec_res['failed'] and params['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)

    if (notebook_kwargs == notebooks_params[-1]) and not pytest.failed and common_params['REMOVE_EXTRA_FILES']:
        shutil.rmtree(common_params['TESTS_ROOT_DIR'])

    # Terminal output
    with capsys.disabled():
        # Extract traceback if failed
        if exec_res['failed'] and params['SHOW_TEST_ERROR_INFO']:
            print(exec_res.get('traceback', ''))

        # Test outputs
        if params['SHOW_MESSAGE']:
            for k, v in exec_res['outputs'].items():
                message = v if isinstance(v, str) else json.dumps(v, indent=4)
                print(f"{k}:\n {message}\n")

        # End of the running message
        notebook_info = f"{params['DATESTAMP']} \'{filename}\'{' with config=' + str(config) if config else ''} was"
        if not exec_res['failed']:
            print(f"{notebook_info} executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"
