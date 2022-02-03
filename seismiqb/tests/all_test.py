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
import pytest

from .run_notebook import run_notebook


testdata = [
    ('geometry', {'TEST_OUTPUTS': ['message', 'timings']}),
    ('charisma', {}),
    ('horizon', {'TEST_OUTPUTS': 'message'})
]

@pytest.mark.parametrize("test_name,test_kwargs", testdata)
def test_notebook(test_name, test_kwargs, capsys, tmpdir):
    """ Prepare a test workspace, execute a test notebook, extract and print an output message.

    Parameters
    ----------
    test_name : str
        A test name and a prefix of a test notebook.
    test_kwargs : dict
        A test notebook kwargs.
    capsys : :class:`_pytest.capture.CaptureFixture`
        Pytest capturing of the stdout/stderr output.
    tmpdir : :class:`py._path.local.LocalPath`
        The tmpdir fixture from the pytest module.
    """
    # Initialize base test variables
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

    test_outputs = test_kwargs.pop('TEST_OUTPUTS', None)

    test_variables = {
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

    test_variables.update(test_kwargs)

    # Workspace preparation
    # Saving logs: in a local case we save logs in the `tests/logs` directory
    # In tmpdir case we save logs in a temporary directory
    if test_variables['USE_TMP_OUTPUT_DIR']:
        test_variables['OUTPUT_DIR'] = tmpdir.mkdir(f"{test_name}_test_files")
        test_variables['LOGS_DIR'] = tmpdir.mkdir("logs")

    else:
        test_variables['OUTPUT_DIR'] = os.path.join(test_variables['TESTS_SCRIPTS_DIR'],
                                                    f"notebooks/{test_name}_test_files")
        test_variables['LOGS_DIR'] = os.path.join(TESTS_SCRIPTS_DIR, 'logs/')

    # Run and save the test notebook
    file_name = test_name + '_test'
    path_ipynb = os.path.join(test_variables['NOTEBOOKS_DIR'], f"{file_name}.ipynb")
    out_path_ipynb = os.path.join(test_variables['LOGS_DIR'], f"{file_name}_out_{test_variables['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=test_variables, outputs=test_outputs,
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    # Logs postprocessing: drop out_path_ipynb if all OK
    if not exec_res['failed'] and test_variables['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)

    # Provide output message to the terminal
    with capsys.disabled():
        print(test_variables['DATESTAMP'] + '\n\n')

        # Tests output
        if test_variables['SHOW_MESSAGE']:
            for k, v in exec_res.get('outputs', {}).items():
                if isinstance(v, str):
                    print(f"{k}:\n\n{v}")
                else:
                    print(f"{k}:\n")
                    print(json.dumps(v, indent=4))

        # End of the running message
        if not exec_res['failed']:
            print(f"All \'{test_name}\' tests were executed successfully.\n")
        else:
            if test_variables['SHOW_TEST_ERROR_INFO']:
                print(exec_res['traceback'])
            assert False, f"\'{test_name}\' tests failed.\n"
