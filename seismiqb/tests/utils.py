""" Tests helper functions."""
import os
import shutil

from ..batchflow.run_notebook import run_notebook


def remove_paths(dirs_to_remove=None, files_to_remove=None):
    """ Remove saves from a previous run.

    Parameters
    ----------
    dirs_to_remove: list of str
        A list of paths to directories to remove.
    files_to_remove: list of str
        A list of paths to files to remove.
    """
    if dirs_to_remove is None:  dirs_to_remove = []
    if files_to_remove is None: files_to_remove = []

    for path in files_to_remove:
        if os.path.exists(path):
            os.remove(path)

    for directory in dirs_to_remove:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print(f"Can't delete the directory {directory} : {e.strerror}")


def execute_test_notebook(path_ipynb, inputs, out_path_ipynb,
                          show_test_error_info, remove_extra_files, outputs=None):
    """ Execute a test notebook and construct an exit message.

    Parameters
    ----------
    path_ipynb : str
        Path to a test notebook.
    inputs,  outputs, out_path_ipynb
        Passed directly to :meth:`..batchflow.run_notebook.run_notebook`.
    show_test_error_info : bool
        Whether to show error traceback in outputs.
        Notice that it only works with SHOW_MESSAGE = True.
    remove_extra_files : bool
        Whether to remove extra files after execution.
        Extra files are temporary files and execution saved files that relate to successful tests.
    """
    file_name = path_ipynb.split('/')[-1].split('.')[0]
    test_name = file_name.replace('_test', '')
    out_path_db = os.path.splitext(out_path_ipynb)[0] + '_db'

    exec_res = run_notebook(
        path=path_ipynb,
        inputs=inputs,
        outputs=outputs,
        inputs_pos=2,
        out_path_db=out_path_db,
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

        message += f"An ERROR occurred in cell number {exec_res.get('failed cell number', None)} in {out_path_ipynb}\n"

    outputs = exec_res.get('outputs', {})
    return exec_res['failed'], message, outputs
