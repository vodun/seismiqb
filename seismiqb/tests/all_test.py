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
from glob import glob
import json
from itertools import product
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

# Tests configurations
geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']

# Parameters for all tests stages (preparation, main, final)
all_tests_kwargs = [
    # (test_name, test_kwargs)
    ('geometry', {'TEST_OUTPUTS': ['states', 'timings'],
                  'FORMATS': geometry_formats}),
    ('charisma', {}),
    ('horizon', {'TEST_OUTPUTS': ['states', 'message']})
]

# Iterables for main tests
main_tests_kwargs = {'geometry': {'FORMAT': geometry_formats}}
    

@pytest.fixture(params=all_tests_kwargs)
def test_prepartion(request, capsys, tmpdir):
    """..!!.."""
    test_name, test_kwargs = request.param

    # Workspace preparation
    if tests_params['USE_TMP_OUTPUT_DIR']:
        test_kwargs['OUTPUT_DIR'] = tmpdir.mkdir(f"{test_name}_test_files")
        test_kwargs['LOGS_DIR'] = tmpdir.mkdir("logs")

    else:
        test_kwargs['OUTPUT_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'],
                                                    f"notebooks/{test_name}_test_files")
        test_kwargs['LOGS_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'], 'logs/')

    test_kwargs.update(tests_params)

    # Run preparation notebook if exists
    test_notebooks_paths = glob(test_kwargs['NOTEBOOKS_DIR'] + test_name + '*.ipynb')
    test_notebooks_paths = [x for x in test_notebooks_paths if x.find('preparation') != -1]

    exec_res = {'failed': False} # Some tests haven't preparation notebooks
    for path_ipynb in test_notebooks_paths:
        exec_res = run_test_notebook(path_ipynb, test_kwargs, capsys)

    return test_name, test_kwargs, exec_res['failed'], exec_res.get('outputs', {})


@pytest.fixture
def test_run_main_notebook(test_prepartion, capsys):
    """..!!.."""
    # Extract params from iterables and run main test notebook with params
    test_name, test_kwargs, test_failed, test_outputs = test_prepartion
    test_kwargs.update(test_outputs) # For saving shared logs

    test_notebooks_paths = glob(test_kwargs['NOTEBOOKS_DIR'] + test_name + '*.ipynb')
    test_notebooks_paths = [x for x in test_notebooks_paths if (x.find('preparation') == -1) and (x.find('final') == -1)]

    if not test_failed:
        # Extract iterable params configurations
        iterables = main_tests_kwargs.get(test_name, {})
        params_values = product(*iterables.values())

        for values in params_values:
            params = {k: v for k, v in zip(iterables.keys(), values)}
            suffix = "_" + "_".join(str(v) for v in values)
            test_kwargs.update(params)

            # Run main test notebooks with params and save test state
            if params:
                with capsys.disabled():
                    print(f"Running `{test_name}` main test notebooks with `{params}`.")

            for path_ipynb in test_notebooks_paths:
                exec_res = run_test_notebook(path_ipynb=path_ipynb, test_kwargs=test_kwargs, capsys=capsys, suffix=suffix)

                test_failed = test_failed or exec_res['failed']
                test_outputs.update(exec_res.get('outputs', {}))

                test_kwargs.update(test_outputs)

            for k in iterables.keys(): # Delete iterable keys from globals
                _ = test_kwargs.pop(k, None)

    return test_name, test_kwargs, test_failed, test_outputs


def test_run_final_notebook(test_run_main_notebook, capsys):
    """..!!.."""
    test_name, test_kwargs, test_failed, test_outputs = test_run_main_notebook

    # Run a final notebook if exists
    if not test_failed:
        test_notebooks_paths = glob(test_kwargs['NOTEBOOKS_DIR'] + test_name + '*.ipynb')
        test_notebooks_paths = [x for x in test_notebooks_paths if x.find('final') != -1]

        for path_ipynb in test_notebooks_paths:
            exec_res = run_test_notebook(path_ipynb, test_kwargs, capsys)

            test_failed = test_failed or exec_res['failed']
            test_outputs.update(exec_res.get('outputs', {}))

        if test_kwargs['REMOVE_EXTRA_FILES'] and not test_kwargs['USE_TMP_OUTPUT_DIR']:
            remove_paths(paths=test_kwargs['OUTPUT_DIR'])

    # Provide outputs to the terminal
    with capsys.disabled():
        print('\n' + test_kwargs['DATESTAMP'])

        # Tests output
        if test_kwargs['SHOW_MESSAGE']:
            for k, v in test_outputs.items():
                if isinstance(v, str):
                    print(f"{k}:\n\n{v}")
                else:
                    print(f"{k}:\n")
                    print(json.dumps(v, indent=4))

        # End of the running message
        if not test_failed:
            print(f"\'{test_name}\' tests were executed successfully.\n")
        else:
            assert False, f"\'{test_name}\' tests failed.\n"


# Helper method
def run_test_notebook(path_ipynb, test_kwargs, capsys, suffix=""):
    """..!!.."""
    # Run test notebook 
    file_name = os.path.splitext(os.path.split(path_ipynb)[1])[0]
    out_path_ipynb = os.path.join(test_kwargs['LOGS_DIR'], f"{file_name}_out{suffix}_{test_kwargs['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=test_kwargs, outputs=test_kwargs.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)
    
    # Logs postprocessing: drop out_path_ipynb if all OK
    if not exec_res['failed'] and test_kwargs['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)
        
    # Provide output message to the terminal
    with capsys.disabled():
        # End of the running message
        if not exec_res['failed']:
            print(f"\'{file_name}\' notebook was executed successfully.\n")
        else:
            if test_kwargs['SHOW_TEST_ERROR_INFO']:
                print(exec_res['traceback'])

    return exec_res
