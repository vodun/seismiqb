""" Script for running controller notebooks for tests.

The behaviour of tests is parametrized by the following constants:

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

And you can manage tests running with parameters:

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

from .utils import extract_traceback, execute_test_notebook
from ..batchflow.utils_notebook import run_notebook    


def run_test_notebook(test_name, test_kwargs, capsys, tmpdir, messages_paths_regexp=None):
    """ Prepare a test workspace, execute a test notebook, extract and print an output message.

    Parameters:
    ----------
    test_name : str
        A test name and a prefix of a test notebook.
    test_kwargs : dict
        A test notebook kwargs.
    capsys : :class:`_pytest.capture.CaptureFixture`
        Pytest capturing of the stdout/stderr output.
    tmpdir : :class:`py._path.local.LocalPath`
        The tmpdir fixture from the pytest module.
    messages_paths_regexp : list of str
        A list with regular expressions of paths to files with output messagies,
        e.g. apath to log file and a path to timings file.
    """
    if messages_paths_regexp is None:
        messages_paths_regexp = []

    # Initialize base test variables
    TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

    test_variables = {
        # Workspace constants
        'DATESTAMP': date.today().strftime("%Y-%m-%d"),
        'TESTS_SCRIPTS_DIR': TESTS_SCRIPTS_DIR,
        'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
        'USE_TMP_OUTPUT_DIR': os.getenv('SEISMIQB_TEST_USE_TMP_OUTPUT_DIR', True),

        # Execution parameters
        'REMOVE_OUTDATED_FILES': os.getenv('SEISMIQB_TEST_REMOVE_OUTDATED_FILES', True),
        'REMOVE_EXTRA_FILES': os.getenv('SEISMIQB_TEST_REMOVE_EXTRA_FILES', True),
        'SHOW_MESSAGE': os.getenv('SEISMIQB_TEST_SHOW_MESSAGE', True),
        'SHOW_TEST_ERROR_INFO': os.getenv('SEISMIQB_TEST_SHOW_ERROR_INFO', True),
        
        # Visualization parameters
        'FIGSIZE': os.getenv('SEISMIQB_TEST_FIGSIZE', (12, 7)),
        'SHOW_FIGURES': os.getenv('SEISMIQB_TEST_SHOW_FIGURES', False),

        # Output parameters
        'VERBOSE': os.getenv('SEISMIQB_TEST_VERBOSE', True),
    }

    test_variables.update(test_kwargs)


    # Workspace preparation
    # Saving logs: in a local case we save logs in the `tests/logs` directory
    # In tmpdir case we save logs in a temporary directory
    if test_variables['USE_TMP_OUTPUT_DIR']:
        test_variables['OUTPUT_DIR'] = tmpdir.mkdir("notebooks").mkdir(f"{test_name}_test_files")
        test_variables['LOGS_DIR'] = tmpdir.mkdir("logs")

    else:
        test_variables['OUTPUT_DIR'] = os.path.join(test_variables['TESTS_SCRIPTS_DIR'], f"notebooks/{test_name}_test_files")
        test_variables['LOGS_DIR'] = os.path.join(TESTS_SCRIPTS_DIR, 'logs/')

    # Run and save the test notebook
    message = ""

    file_name = test_name + '_test'
    path_ipynb = os.path.join(test_variables['NOTEBOOKS_DIR'], f"{file_name}.ipynb")
    out_path_ipynb = os.path.join(test_variables['LOGS_DIR'], f"{file_name}_out_{test_variables['DATESTAMP']}.ipynb")
    
    traceback_message, failed = execute_test_notebook(path_ipynb=path_ipynb, nb_kwargs=test_variables,
                                                      out_path_ipynb=out_path_ipynb,
                                                      show_test_error_info=test_variables['SHOW_TEST_ERROR_INFO'],
                                                      remove_extra_files=test_variables['REMOVE_EXTRA_FILES'])

    # Extract logs paths that need to be saved
    SAVE_LOGS_PATHS = []
    SAVE_LOGS_REG_EXP = test_kwargs.get('SAVE_LOGS_REG_EXP', [])

    if SAVE_LOGS_REG_EXP:
        for reg_exp in SAVE_LOGS_REG_EXP:
            paths = glob(os.path.join(test_variables['LOGS_DIR'], reg_exp))
            SAVE_LOGS_PATHS.extend(paths)

    if not failed:
        # Open all files with messages and save data into the variable and remove unneccessary logs
        for message_path_regexp in messages_paths_regexp:
            messages_paths = glob(os.path.join(test_variables['LOGS_DIR'], message_path_regexp))

            for message_path in messages_paths:
                if test_variables['SHOW_MESSAGE']:
                    extension = message_path.split('.')[-1]

                    # Open message and store it in a pretty string
                    with open(message_path, "r", encoding="utf-8") as infile:
                        if extension == 'json':
                            current_message = json.load(infile)
                            current_message = json.dumps(current_message, indent=4)
                        else:
                            current_message = infile.readlines()
                            current_message = ''.join(current_message)

                    message += current_message

                if not (message_path in SAVE_LOGS_PATHS):
                    os.remove(message_path)

        failed = (message.find('fail') != -1) or failed

    else:
        if test_variables['SHOW_TEST_ERROR_INFO']:
            # Add error traceback into the message
            message += traceback_message

        message += f"\n{test_name} tests execution failed."

    # Provide output message to the terminal
    with capsys.disabled():
        # Tests output
        if test_variables['SHOW_MESSAGE']:
            print(message)

        # End of the running message
        if not failed:
            print(f"All \'{test_name}\' tests were executed successfully.\n")
        else:
            assert False, f"\'{test_name}\' tests failed.\n"
