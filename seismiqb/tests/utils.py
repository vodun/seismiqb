""" Tests helper functions."""
from glob import glob
import json
import os
import nbformat

from ..batchflow.utils_notebook import run_notebook

def find_traceback_in_outputs(cell_info):
    """ Find cell output with a traceback and extract the traceback. """
    outputs = cell_info.get('outputs', [])
    traceback_message = ""
    has_error_traceback = False

    for output in outputs:
        traceback = output.get('traceback', [])

        if traceback:
            has_error_traceback = True

            for line in traceback:
                traceback_message += line
            break

    return has_error_traceback, traceback_message

def extract_traceback(path_ipynb, cell_num=None):
    """ Extracts error traceback from tests notebooks.

    Parameters
    ----------
    path_ipynb: str
        Path to a notebook from which extract error traceback.
    cell_num: int or None
        A number of an error cell.
        If None, than we will find cell_num iterating over notebook.
        If notebook doesn't contain markdown cells than `cell_num` equals to `exec_info`
        from `seismiqb.batchflow.utils_notebook.run_notebook` in error case.
    """
    traceback_message = "TRACEBACK: \n"
    failed = False
    out_notebook = nbformat.read(path_ipynb, as_version=4)

    if cell_num is not None:
        # Get a traceback from cell directly
        cell_info = out_notebook['cells'][cell_num]

        has_error_traceback, current_traceback_message = find_traceback_in_outputs(cell_info=cell_info)

        traceback_message += current_traceback_message
        failed = failed or has_error_traceback

    else:
        # Find a cell with a traceback
        for cell_info in out_notebook['cells']:
            has_error_traceback, current_traceback_message = find_traceback_in_outputs(cell_info=cell_info)

            traceback_message += current_traceback_message
            failed = failed or has_error_traceback

    return failed, traceback_message

def local_workspace_preparation(REMOVE_OUTDATED_FILES, DATESTAMP, TESTS_SCRIPTS_DIR,
                                notebook_prefix, local_output_dir=None):
    """ Prepare local workspace for tests: remove outdated files, set OUTPUT_DIR and out_path_ipynb links.

    Parameters:
    ----------
    REMOVE_OUTDATED_FILES : bool
        Whether to remove outdated files which relate to previous executions.
    DATESTAMP : str
        Execution date in "YYYY-MM-DD" format.
        Used for saving notebooks executions and temporary files.
    TESTS_SCRIPTS_DIR : str
        Path to the directory with test .py scripts.
        Used as an entry point to the working directory.
    notebook_prefix : str
        Name prefix for a test controller notebook.
    local_output_dir : str
        Preferrable local path to the directory for saving results and temporary files
        (executed notebooks, logs, data files like cubes, etc.).
    """
    # Remove outdated executed controller notebook (It is saved near to the original one)
    if REMOVE_OUTDATED_FILES:
        previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/{notebook_prefix}_test_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

    # Create main paths links
    if local_output_dir is None:
        OUTPUT_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks', f'{notebook_prefix}_tests_files')
    else:
        OUTPUT_DIR = local_output_dir

    out_path_ipynb = os.path.join(OUTPUT_DIR, f'{notebook_prefix}_test_out_{DATESTAMP}.ipynb')

    return OUTPUT_DIR, out_path_ipynb

def run_test_notebook(OUTPUT_DIR, out_path_ipynb, TESTS_SCRIPTS_DIR, notebook_path, nb_kwargs,
                      REMOVE_EXTRA_FILES, test_name, SHOW_MESSAGE, SHOW_TEST_ERROR_INFO,
                      capsys, messages_paths_regexp=None):
    """ Execute a test notebook, extract and print an output message.

    Parameters:
    ----------
    OUTPUT_DIR : str
        Path to the directory for saving results and temporary files
        (executed notebooks, logs, data files like cubes, etc.).
    out_path_ipynb : str
        Path where to save an executed test notebook.
    TESTS_SCRIPTS_DIR : str
        Path to the directory with test .py scripts.
        Used as an entry point to the working directory.
    notebook_path : str
        Path to a test notebook to execute.
    nb_kwargs : dict
        A test notebook kwargs.
    messages_paths_regexp : list of str
        A list with regular expressions of paths to files with output messagies,
        e.g. apath to log file and a path to timings file.
    REMOVE_EXTRA_FILES : bool
        Whether to remove extra files after execution.
        Extra files are temporary files and execution savings that relate to successful tests.
    test_name : str
        A test name which printed in a message.
    SHOW_MESSAGE : bool
        Whether to show a detailed tests execution message.
    SHOW_TEST_ERROR_INFO : bool
        Whether to show error traceback in outputs.
        Notice that it only works with SHOW_MESSAGE = True.
    capsys
        Pytest capturing of the stdout/stderr output.
    """
    if messages_paths_regexp is None:
        messages_paths_regexp = []

    # Run and save the test notebook
    exec_info = run_notebook(
            path=os.path.join(TESTS_SCRIPTS_DIR, notebook_path),
            nb_kwargs=nb_kwargs,
            insert_pos=2,
            out_path_ipynb=out_path_ipynb,
            display_links=False
    )

    # Tests exit
    message = ""

    failed, traceback_message = extract_traceback(path_ipynb=out_path_ipynb)
    failed = failed or (exec_info is not True)

    if not failed:
        if SHOW_MESSAGE:
            # Open all files with messages and save data into the variable
            for message_path_regexp in messages_paths_regexp:
                message_path = glob(os.path.join(OUTPUT_DIR, message_path_regexp))[-1]
                extension = message_path.split('.')[-1]

                # Open message and save it as a pretty string
                with open(message_path, "r", encoding="utf-8") as infile:
                    if extension == 'json':
                        current_message = json.load(infile)
                        current_message = json.dumps(current_message, indent=4)
                    else:
                        current_message = infile.readlines()
                        current_message = ''.join(current_message)

                message += current_message

            failed = (message.find('fail') != -1) or failed

        # If everything is OK we can delete the test notebook
        if REMOVE_EXTRA_FILES:
            try:
                os.remove(out_path_ipynb)
            except OSError as e:
                print(f"Can't delete the file: {out_path_ipynb} : {e.strerror}")

    else:
        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            message += traceback_message

        message += f'\n{test_name} tests execution failed.'

    # Provide output message to the terminal
    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            print(message)

        # End of the running message
        if not failed:
            print(f'Tests for {test_name} were executed successfully.\n')
        else:
            assert False, f'{test_name} tests failed.\n'
