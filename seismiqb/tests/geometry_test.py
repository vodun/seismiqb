""" Script for running notebook with SeismicGeometry tests."""
import glob
import json
import os
import pprint
from datetime import date

# from ..batchflow.utils_notebook import run_notebook
import sys
import re
import json
import time
import warnings
import numpy as np


# Constants
DATESTAMP = date.today().strftime("%Y-%m-%d")
DROP_EXTRA_FILES = True
SHOW_TEST_ERROR_INFO = True
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
TEST_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_files/')
SHOW_MESSAGE = True
GITHUB_MODE = True

def run_notebook(path, nb_kwargs=None, insert_pos=1, kernel_name=None, timeout=-1,
                 working_dir='./', execute_kwargs=None,
                 save_ipynb=True, out_path_ipynb=None, save_html=False, out_path_html=None, suffix='_out',
                 add_timestamp=True, hide_input=False, display_links=True,
                 raise_exception=False, show_error_info=True, return_nb=False):
    """ Run a notebook and save the output.
    Additionally, allows to pass `nb_kwargs` arguments, that are used for notebook execution. Under the hood,
    we place all of them into a separate cell, inserted in the notebook; hence, all of the keys must be valid Python
    names, and values should be valid for re-creating objects.
    Heavily inspired by https://github.com/tritemio/nbrun.

    Parameters
    ----------
    path : str
        Path to the notebook to execute.
    nb_kwargs : dict, optional
        Additional arguments for notebook execution. Converted into cell of variable assignments and inserted
        into the notebook on `insert_pos` place.
    insert_pos : int
        Position to insert the cell with argument assignments into the notebook.
    kernel_name : str, optional
        Name of the kernel to execute the notebook.
    timeout : int
        Maximum execution time for each cell. -1 means no constraint.
    working_dir : str
        The working folder of starting the kernel.
    execute_kwargs : dict, optional
        Other parameters of `:class:ExecutePreprocessor`.
    save_ipynb : bool
        Whether to save the output .ipynb file.
    out_path_ipynb : str, optional
        Path to save the output .ipynb file. If not provided and `save_ipynb` is set to True, we add `suffix` to `path`.
    save_html : bool
        Whether to convert the output notebook to .html.
    out_path_html : str, optional
        Path to save the output .html file.
        If not provided and `save_html` is set to True, we add `suffix` to `path` and change extension.
    suffix : str
        Appended to output file names if paths are not explicitly provided.
    add_timestamp : bool
        Whether to add cell with execution information in the beginning of the output notebook.
    hide_input : bool
        Whether to hide the code cells in the output notebook.
    display_links : bool
        Whether to display links to the output notebook and html at execution.
    raise_exception : bool
        Whether to re-raise exceptions from the notebook.
    show_error_info : bool
        Whether to show a message with information about an error in the output notebook (if an error exists).
    return_nb : bool
        Whether to return the notebook object from this function.
    """
    # pylint: disable=bare-except, lost-exception
    from IPython.display import display, FileLink
    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor
    from nbconvert import HTMLExporter

    # Prepare paths
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path {path} not found.')

    if save_ipynb and out_path_ipynb is None:
        out_path_ipynb = os.path.splitext(path)[0] + suffix + '.ipynb'
    if save_html and out_path_html is None:
        out_path_html = os.path.splitext(path)[0] + suffix + '.html'

    # Execution arguments
    execute_kwargs = execute_kwargs or {}
    execute_kwargs.update(timeout=timeout)
    if kernel_name:
        execute_kwargs.update(kernel_name=kernel_name)
    executor = ExecutePreprocessor(**execute_kwargs)

    # Read the master notebook, prepare and insert kwargs cell
    notebook = nbformat.read(path, as_version=4)
    exec_info = True
    if hide_input:
        notebook["metadata"].update({"hide_input": True})
    if nb_kwargs:
        header = '# Cell inserted during automated execution.'
        lines = (f"{key} = {repr(value)}"
                 for key, value in nb_kwargs.items())
        code = '\n'.join(lines)
        code_cell = '\n'.join((header, code))
        notebook['cells'].insert(insert_pos, nbformat.v4.new_code_cell(code_cell))

    # Execute the notebook
    start_time = time.time()
    try:
        executor.preprocess(notebook, {'metadata': {'path': working_dir}})
    except:
        # Execution failed, print a message with error location and re-raise
        # Find cell with a failure
        exec_info = sys.exc_info()

        # Get notebook cells from an execution traceback and iterate over them
        notebook_cells = exec_info[2].tb_frame.f_locals['notebook']['cells']
        error_cell_number = None
        for cell in notebook_cells:
            try:
                # A cell with a failure has 'output_type' equals to 'error', but cells have
                # variable structure and some of them don't have these target fields
                if cell['outputs'][0]['output_type'] == 'error':
                    error_cell_number = cell['execution_count']
                    break
            except:
                pass

        if show_error_info:
            msg = ('Error executing the notebook "%s".\n'
                   'Notebook arguments: %s\n\n'
                   'See notebook "%s" (cell number %s) for the traceback.' %
                   (path, str(nb_kwargs), out_path_ipynb, error_cell_number))
            print(msg)

        exec_info = error_cell_number

        if raise_exception:
            raise
    finally:
        # Add execution info
        if add_timestamp:
            duration = int(time.time() - start_time)
            timestamp = (f'**Executed:** {time.ctime(start_time)}<br>'
                         f'**Duration:** {duration} seconds.<br>'
                         f'**Autogenerated from:** [{path}]\n\n---')
            timestamp_cell = nbformat.v4.new_markdown_cell(timestamp)
            notebook['cells'].insert(0, timestamp_cell)

        # Save the executed notebook/HTML to disk
        if save_ipynb:
            with open(out_path_ipynb, 'w', encoding='utf-8') as file:
                nbformat.write(notebook, file)

            if display_links:
                display(FileLink(out_path_ipynb))
        if save_html:
            html_exporter = HTMLExporter()
            body, _ = html_exporter.from_notebook_node(notebook)
            with open(out_path_html, 'w') as f:
                f.write(body)
            if display_links:
                display(FileLink(out_path_html))

        if return_nb:
            return (exec_info, notebook)
        return exec_info


def test_geometry(capsys, tmpdir):
    """ Run SeismicGeometry test notebook.

    This test runs ./notebooks/geometry_test.ipynb test file and show execution message and
    the most important timings for SeismicGeometry tests.

    Under the hood, this notebook create a fake seismic cube, saves it in different data formats
    and for each format run SeismicGeometry tests.
    """
    # Delete old test notebook results
    if GITHUB_MODE:
        SAVING_DIR = tmpdir.mkdir("notebooks").mkdir("geometry_test_files")
        out_path_ipynb = SAVING_DIR.join(f"geometry_test_out_{DATESTAMP}.ipynb")

    else:
        # Clear outdatted files
        previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

        # Path to a new test noteboook result
        SAVING_DIR = TEST_DIR
        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/geometry_test_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test.ipynb'),
        nb_kwargs={
            'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'TEST_DIR': TEST_DIR,
            'DATESTAMP': DATESTAMP,
            'DROP_EXTRA_FILES': DROP_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO,
            'SAVING_DIR': SAVING_DIR,
            'GITHUB_MODE': GITHUB_MODE
        },
        insert_pos=1,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    if exec_info is True:
        # Open message
        message_path = glob.glob(os.path.join(SAVING_DIR, 'message_*.txt'))[-1]

        with open(message_path, "r", encoding="utf-8") as infile:
            msg = infile.readlines()

        # Open timings
        timings_path = glob.glob(os.path.join(SAVING_DIR, 'timings_*.json'))[-1]

        with open(timings_path, "r", encoding="utf-8") as infile:
            timings = json.load(infile)

    else:
        msg = ['SeismicGeometry tests execution failed.\n']
        timings= {'state': 'FAIL'}

    with capsys.disabled():
        # Tests output
        if SHOW_MESSAGE:
            for line in msg:
                print(line)

        pp = pprint.PrettyPrinter()
        pp.pprint(timings)
        print('\n')

        # End of the running message
        if timings['state']=='OK':
            print('Tests for SeismicGeometry were executed successfully.\n')
        else:
            assert False, 'SeismicGeometry tests failed.\n'
