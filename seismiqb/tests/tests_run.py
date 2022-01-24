""" Script for running controller notebooks for tests.

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

from .utils import execute_test_notebook


def run_test_notebook(test_name, test_kwargs, capsys, tmpdir):
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

    test_outputs_names = test_kwargs.pop('TEST_OUTPUTS', None)

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
        test_variables['OUTPUT_DIR'] = tmpdir.mkdir("notebooks").mkdir(f"{test_name}_test_files")
        test_variables['LOGS_DIR'] = tmpdir.mkdir("logs")

    else:
        test_variables['OUTPUT_DIR'] = os.path.join(test_variables['TESTS_SCRIPTS_DIR'],
                                                    f"notebooks/{test_name}_test_files")
        test_variables['LOGS_DIR'] = os.path.join(TESTS_SCRIPTS_DIR, 'logs/')

    # Run and save the test notebook
    file_name = test_name + '_test'
    path_ipynb = os.path.join(test_variables['NOTEBOOKS_DIR'], f"{file_name}.ipynb")
    out_path_ipynb = os.path.join(test_variables['LOGS_DIR'], f"{file_name}_out_{test_variables['DATESTAMP']}.ipynb")


    failed, traceback, test_outputs = execute_test_notebook(path_ipynb=path_ipynb,
                                                            inputs=test_variables, outputs=test_outputs_names,
                                                            out_path_ipynb=out_path_ipynb,
                                                            show_test_error_info=test_variables['SHOW_TEST_ERROR_INFO'],
                                                            remove_extra_files=test_variables['REMOVE_EXTRA_FILES'])
    # Logs postprocessing
    if not failed:
        if test_variables['REMOVE_EXTRA_FILES'] and not test_variables['USE_TMP_OUTPUT_DIR']:
            # Extract logs paths that need to be saved
            SAVE_LOGS_PATHS = []
            SAVE_LOGS_REG_EXP = test_kwargs.get('SAVE_LOGS_REG_EXP', [])

            if SAVE_LOGS_REG_EXP:
                for reg_exp in SAVE_LOGS_REG_EXP:
                    paths = glob(os.path.join(test_variables['LOGS_DIR'], reg_exp))
                    SAVE_LOGS_PATHS.extend(paths)

            # Remove unnecessary logs
            logs_paths = os.listdir(test_variables['LOGS_DIR'])

            for logs_path in logs_paths:
                if logs_path.find(test_name) != -1:
                    logs_path = os.path.join(test_variables['LOGS_DIR'], logs_path)
                    if not logs_path in SAVE_LOGS_PATHS:
                        os.remove(logs_path)

    # Provide output message to the terminal
    with capsys.disabled():
        print(test_variables['DATESTAMP'] + '\n\n')
        # Tests output
        if test_variables['SHOW_MESSAGE']:
            for k, v in test_outputs.items():
                if isinstance(v, str):
                    print(k, ':\n', v)
                else:
                    print(k, ':\n')
                    print(json.dumps(v, indent=4))

        # End of the running message
        if not failed:
            print(f"All \'{test_name}\' tests were executed successfully.\n")
        else:
            if test_variables['SHOW_TEST_ERROR_INFO']:
                print(traceback)
            assert False, f"\'{test_name}\' tests failed.\n"
