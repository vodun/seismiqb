""" Tests helper functions."""
import os
import shutil
import nbformat

from ..batchflow.utils_notebook import run_notebook


def remove_savings(dirs_to_remove=None, paths_to_remove=None):
    """ Remove savings from a previous run.

    Parameters:
    ----------
    dirs_to_remove: list of str
        A list of paths to directories to remove.
    paths_to_remove: list of str
        A list of paths to files to remove.
    """
    if dirs_to_remove is None:
        dirs_to_remove = []
    if paths_to_remove is None:
        paths_to_remove = []

    for path in paths_to_remove:
        if os.path.exists(path):
            os.remove(path)

    for directory in dirs_to_remove:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print(f"Can't delete the directory {directory} : {e.strerror}")


def prepare_local(output_dir, remove_outdated_files, dirs_to_remove=None, paths_to_remove=None):
    """ Prepare a local workpspace: remove outdated files and create output directory if needed.

    Parameters:
    ----------
    output_dir : str
        Path to the directory for saving results and temporary files
        (executed notebooks, logs, data files like cubes, etc.).
    remove_outdated_files : bool
        Whether to remove outdated files which relate to previous executions.
    dirs_to_remove: list of str
        A list of paths to directories to remove.
    paths_to_remove : list of str
        A list of paths to files to remove.
    """
    if remove_outdated_files:
        remove_savings(dirs_to_remove=dirs_to_remove, paths_to_remove=paths_to_remove)

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


def execute_test_notebook(path_ipynb, nb_kwargs, out_path_ipynb,
                          show_test_error_info, remove_extra_files):
    """ Execute a test notebook and construct an exit message.

    Parameters:
    ----------
    path_ipynb : str
        Path to a test notebook.
    nb_kwargs : dict
        Test notebook kwargs.
    out_path_ipynb : str
        Path to save an executed notebook.
    show_test_error_info : bool
        Whether to show error traceback in outputs.
        Notice that it only works with SHOW_MESSAGE = True.
    remove_extra_files : bool
        Whether to remove extra files after execution.
        Extra files are temporary files and execution savings that relate to successful tests.
    """
    file_name = path_ipynb.split('/')[-1].split('.')[0]
    test_name = file_name.replace('_test', '')

    exec_info = run_notebook(
        path=path_ipynb,
        nb_kwargs=nb_kwargs,
        insert_pos=2,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    failed, traceback_msg = extract_traceback(path_ipynb=out_path_ipynb)
    failed = failed or (exec_info is not True)

    if not failed:
        message = f"\'{test_name}\' tests were executed successfully.\n"

        # If everything is OK we can delete the test notebook
        if remove_extra_files:
            try:
                os.remove(out_path_ipynb)
                message += f'Notebook {out_path_ipynb} executed correctly and was deleted.\n'
            except OSError as e:
                print(f"Can't delete the file: {out_path_ipynb} : {e.strerror}")

    else:
        message = f"\'{test_name}\'' tests execution failed.\n"

        if show_test_error_info:
            # Add error traceback into the message
            message += traceback_msg
            message += '\n'

        message += f'An ERROR occured in cell number {exec_info} in {out_path_ipynb}\n'

    return message, failed
