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
OUTPUT_DIR : str
    Path to the directory for saving results and temporary files
    (executed notebooks, logs, data files like cubes, etc.).
LOGS_DIR : str
    Path to the directory for saving log files (executed notebooks, timings, messages).
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

Visualizations in saved execution notebooks are controlled with:

SCALE : int
    Figures scale.
SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.

Text outputs in executed notebooks controlled with:

VERBOSE : bool
    Whether to print information about successful tests during the execution of the cycles.
"""
import os
from datetime import date
import json
import re
import pytest

from .run_notebook import run_notebook
from .utils import remove_paths


# Initialize base tests variables
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

tests_params = {
    # Workspace constants
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'TESTS_SCRIPTS_DIR': TESTS_SCRIPTS_DIR,
    'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
    'USE_TMP_OUTPUT_DIR': False, #os.getenv('SEISMIQB_TEST_USE_TMP_OUTPUT_DIR') or True,

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
notebooks_kwargs = (
    # (notebook file, test kwargs)
    # Note: kwargs for each notebook in the test will be saved for it and for next notebooks in the test

    # CharismaMixin test
    ('charisma_test.ipynb', {'IS_FINAL': True}),

    # SeismicGeometry test
    ('geometry_test_preparation.ipynb', {'TEST_OUTPUTS': ['states', 'timings'], 'FORMATS': geometry_formats}),
    *[('geometry_test_data_format.ipynb', {'FORMAT': f}) for f in geometry_formats],
    ('geometry_test_final.ipynb', {'IS_FINAL': True}),

    # Horizon test
    ('horizon_test_preparation.ipynb', {'TEST_OUTPUTS': ['message']}),
    ('horizon_test_base.ipynb', {}),
    ('horizon_test_attributes.ipynb', {}),
    ('horizon_test_manipulations.ipynb', {}),
    ('horizon_test_extraction.ipynb', {'IS_FINAL': True})
)


# Global variables (for sharing between notebooks in one test)
tests_workspaces_kwargs = {}
global_tests_states = {}


@pytest.mark.parametrize("notebook_kwargs", notebooks_kwargs)
def test_run_notebook(notebook_kwargs, tmpdir_factory, capsys):
    """..!!.."""
    # Variables and params initialization
    notebook_basename, test_kwargs_ = notebook_kwargs
    is_final_notebook = test_kwargs_.pop('IS_FINAL', False)
    test_name = notebook_basename.split('_')[0]
    config = test_kwargs_.copy()
    _ = config.pop('TEST_OUTPUTS', None)


    # `test_kwargs` initialization
    if 'test_kwargs' not in global_tests_states.get(test_name, {}):
        # Init `test_kwargs`
        test_kwargs = test_kwargs_
        test_kwargs.update(tests_params)
    else:
        # Extract and update `test_kwargs`
        test_kwargs = global_tests_states[test_name]['test_kwargs']
        test_kwargs.update(test_kwargs_)

    # Actualize `test_kwargs`` depend on previous notebooks in the test (for sharing variables, logs, etc.)
    if test_name in global_tests_states:
        test_kwargs.update(global_tests_states[test_name].get('outputs', {})) # For saving shared logs
    else:
        global_tests_states[test_name] = {}

    # Initialize test workspace (for the first notebook in the test)
    if test_name not in tests_workspaces_kwargs:
        tests_workspaces_kwargs[test_name] = {}

        if tests_params['USE_TMP_OUTPUT_DIR']:
            tests_workspaces_kwargs[test_name]['OUTPUT_DIR'] = tmpdir_factory.mktemp(f"{test_name}_test_files")
            tests_workspaces_kwargs[test_name]['LOGS_DIR'] = tmpdir_factory.mktemp("logs")
        else:
            tests_workspaces_kwargs[test_name]['OUTPUT_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'],
                                                    f"notebooks/{test_name}_test_files")
            tests_workspaces_kwargs[test_name]['LOGS_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'], 'logs/')

    test_kwargs.update(tests_workspaces_kwargs[test_name])

    # Run test notebook
    path_ipynb = os.path.join(test_kwargs['NOTEBOOKS_DIR'], notebook_basename)
    file_name = os.path.splitext(notebook_basename)[0]
    suffix = "_" + "_".join(str(k) + "_" + re.sub(r' ', '_', re.sub(r'[^\w^ ]', '', str(v))) for k,v in config.items())
    out_path_ipynb = os.path.join(test_kwargs['LOGS_DIR'], f"{file_name}_out{suffix}_{test_kwargs['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=test_kwargs, outputs=test_kwargs.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    if not exec_res['failed'] and test_kwargs['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)

    # Update global test state
    test_failed = global_tests_states[test_name].get('failed', False) & exec_res['failed']
    test_outputs = global_tests_states[test_name].get('outputs', {})
    test_outputs.update(exec_res.get('outputs', {}))

    test_state = {
        'test_kwargs': test_kwargs,
        'failed': test_failed,
        'outputs': test_outputs
    }
    global_tests_states[test_name].update(test_state)

    # Clear 'OUTPUT_DIR' after the last notebook in the test
    if is_final_notebook and tests_params['REMOVE_EXTRA_FILES'] and not tests_params['USE_TMP_OUTPUT_DIR']:
        remove_paths(paths=tests_workspaces_kwargs[test_name]['OUTPUT_DIR'])

    # Terminal output
    with capsys.disabled():
        print('\n' + test_kwargs['DATESTAMP'])

        # Extract traceback if failed
        if exec_res['failed'] and tests_params['SHOW_TEST_ERROR_INFO']:
            print(exec_res.get('traceback', ''))

        # Notebook outputs
        if test_kwargs['SHOW_MESSAGE']:
            for k, v in exec_res['outputs'].items():
                if isinstance(v, str):
                    print(f"{k}:\n\n{v}")
                else:
                    print(f"{k}:\n")
                    print(json.dumps(v, indent=4))

        # End of the running message
        notebook_info = f"\'{notebook_basename}\'{' with config=' + str(config) if config else ''} was"
        if not exec_res['failed']:
            print(f"{notebook_info} executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"
