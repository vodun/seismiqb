""" Tests helper functions."""
import os
import shutil
import nbformat

from ..batchflow.utils_notebook import run_notebook


def remove_outdated(output_dir, logs_paths=None):
    """ Remove savings from a previous run. """
    # Remove an old output directory
    if os.path.exists(output_dir):
        try:
            shutil.rmtree(output_dir)
        except OSError as e:
            print(f"Can't delete the directory {output_dir} : {e.strerror}")

    if logs_paths is not None:
        for path in logs_paths:
            if os.path.exists(path):
                os.remove(path)

def prepare_local(output_dir, remove_outdated_files, logs_paths=None):
    """ Prepare a local workpspace: remove outdated files and create output directory if needed. """
    if remove_outdated_files:
        remove_outdated(output_dir=output_dir, logs_paths=logs_paths)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


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


def execute_test_notebook(test_name, nb_kwargs, datestamp, notebook_output_dir, notebooks_dir, show_test_error_info):
    """ Execute a test notebook and construct an exit message."""
    out_path_ipynb = os.path.join(notebook_output_dir, f'{test_name}_test_out_{datestamp}.ipynb')

    exec_info = run_notebook(
        path=os.path.join(notebooks_dir, f'{test_name}_test.ipynb'),
        nb_kwargs=nb_kwargs,
        insert_pos=2,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    failed, traceback_msg = extract_traceback(path_ipynb=out_path_ipynb)
    failed = failed or (exec_info is not True)

    if not failed:
        message = f"\'{test_name}\' tests were executed successfully.\n"

    else:
        message = f"\'{test_name}\'' tests execution failed.\n"

        if show_test_error_info:
            # Add error traceback into the message
            message += traceback_msg
            message += '\n'

    return message, failed
