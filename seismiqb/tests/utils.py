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


def prepare_local(output_dir, dirs_to_remove=None, paths_to_remove=None):
    """ Prepare a local workpspace: remove outdated files and (re)create output directory if needed.

    Parameters:
    ----------
    output_dir : str
        Path to the directory for saving results and temporary files
        (executed notebooks, logs, data files like cubes, etc.).
    dirs_to_remove, paths_to_remove
        Passed directly to :meth:`remove_savings`.
    """
    if dirs_to_remove is None:
        dirs_to_remove = [output_dir]
    elif output_dir not in dirs_to_remove:
        dirs_to_remove.append(output_dir)

    remove_savings(dirs_to_remove=dirs_to_remove, paths_to_remove=paths_to_remove)

    os.makedirs(output_dir)


def execute_test_notebook(path_ipynb, nb_kwargs, out_path_ipynb,
                          show_test_error_info, remove_extra_files, nb_outputs=None):
    """ Execute a test notebook and construct an exit message.

    Parameters:
    ----------
    path_ipynb : str
        Path to a test notebook.
    nb_kwargs,  nb_outputs, out_path_ipynb
        Passed directly to :meth:`..batchflow.utils_notebook.run_notebook`.
    show_test_error_info : bool
        Whether to show error traceback in outputs.
        Notice that it only works with SHOW_MESSAGE = True.
    remove_extra_files : bool
        Whether to remove extra files after execution.
        Extra files are temporary files and execution savings that relate to successful tests.
    """
    file_name = path_ipynb.split('/')[-1].split('.')[0]
    test_name = file_name.replace('_test', '')

    exec_res = run_notebook(
        path=path_ipynb,
        nb_kwargs=nb_kwargs,
        nb_outputs=nb_outputs,
        insert_pos=2,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    if not exec_res['failed']:
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
            message += exec_res.get('traceback', "")
            message += '\n'

        message += f"An ERROR occured in cell number {exec_res.get('failed cell number', None)} in {out_path_ipynb}\n"

    nb_outputs = exec_res.get('nb_outputs', {})
    return exec_res['failed'], message, nb_outputs
