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
USE_TMP_OUTPUT_DIR: bool
    Whether to use pytest tmpdir as a workspace.
    If True, then all files are saved in temporary directories.
    If False, then all files are saved in local directories.
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
import pytest

from .run_notebook import run_notebook


# Initialize base tests variables
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

tests_params = {
    # Workspace constants
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'TESTS_SCRIPTS_DIR': TESTS_SCRIPTS_DIR,
    'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
    'USE_TMP_OUTPUT_DIR': os.getenv('SEISMIQB_TEST_USE_TMP_OUTPUT_DIR') or True,

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


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, tmpdir_factory, capsys):
    """..!!.."""
    filename, params = notebook_kwargs
    config = params.copy()
    _ = config.pop('TEST_OUTPUTS', None)

    # Create `root_dir` for all `test_run_notebook` calls
    if not tests_params.get('TESTS_ROOT_DIR', ''):
        if tests_params['USE_TMP_OUTPUT_DIR']:
            tests_params['TESTS_ROOT_DIR'] = tmpdir_factory.mktemp(f"tests_root_dir")
        else:
            tests_params['TESTS_ROOT_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'], "tests_root_dir")
            os.makedirs(tests_params['TESTS_ROOT_DIR'], exist_ok=True)

    params.update(tests_params)

    # Run test notebook
    remove_symbols = lambda v: re.sub(r' ', '_', re.sub(r'[^\w^ ]', '', str(v)))
    suffix = "_".join(f"{k}_{remove_symbols(v)}" for k,v in config.items())

    path_ipynb = os.path.join(params['NOTEBOOKS_DIR'], f"{filename}.ipynb")
    out_path_ipynb = os.path.join(params['TESTS_ROOT_DIR'], f"{filename}_out_{suffix}_{params['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=params, outputs=params.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    if not exec_res['failed'] and params['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)

    # Terminal output
    with capsys.disabled():
        # Extract traceback if failed
        if exec_res['failed'] and params['SHOW_TEST_ERROR_INFO']:
            print(exec_res.get('traceback', ''))

        # Test outputs
        if params['SHOW_MESSAGE']:
            for k, v in exec_res['outputs'].items():
                print(f"{k}:\n")
                if isinstance(v, str):
                    print(v)
                else:
                    print(json.dumps(v, indent=4), '\n')

        # End of the running message
        notebook_info = f"{params['DATESTAMP']} \'{filename}\'{' with config=' + str(config) if config else ''} was"
        if not exec_res['failed']:
            print(f"{notebook_info} executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"
