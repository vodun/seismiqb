""" Test script for running tests notebooks.

The behavior of tests is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
NOTEBOOKS_DIR : str
    Path to the directory with *.ipynb files with tests.
TESTS_ROOT_DIR : str
    Path to the directory for saving results and temporary files for all tests
    (executed notebooks, logs, data files like cubes, etc.).
    Note that in case of success this directory will be removed (if `REMOVE_ROOT_DIR` is True).
REMOVE_ROOT_DIR : bool
    Whether to remove tests root directory files after execution in case of success.

Outputs in saved execution notebooks and terminal are controlled with:

SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.
VERBOSE : bool
    Whether to print information about successful tests during the execution of the cycles.
"""
import os
from datetime import date
import json
import re
import shutil
import tempfile
import pytest

from .run_notebook import run_notebook


# Initialize base tests variables
pytest.failed = False
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

common_params = {
    # Workspace constants and parameters
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
    'TESTS_ROOT_DIR': tempfile.mkdtemp(prefix='tests_root_dir_', dir='./'),
    'REMOVE_ROOT_DIR': os.getenv('SEISMIQB_TEST_REMOVE_ROOT_DIR') or True,

    # Visualization and output parameters (these variables are used in notebooks)
    'SHOW_FIGURES': os.getenv('SEISMIQB_TEST_SHOW_FIGURES') or False,
    'VERBOSE': os.getenv('SEISMIQB_TEST_VERBOSE') or True
}

# Initialize tests configs
geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']
notebooks_params = (
    # (notebook filename, test params dict)
    # CharismaMixin test
    ('charisma_test', {}),

    # SeismicGeometry test
    ('geometry_test_preparation', {'FORMATS': geometry_formats}),
    *[('geometry_test_data_format', {'TEST_OUTPUTS': ['timings'], 'FORMAT': f}) for f in geometry_formats],

    # Horizon test
    ('horizon_test_preparation', {}),
    ('horizon_test_base', {}),
    ('horizon_test_attributes', {}),
    ('horizon_test_manipulations', {}),
    ('horizon_test_extraction', {'TEST_OUTPUTS': ['message']})
)


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys):
    """ Run tests notebooks using kwargs and print outputs in the terminal."""
    filename, params = notebook_kwargs

    config = params.copy() # Test configuration for saving outputs and printing information properly
    _ = config.pop('TEST_OUTPUTS', None) # Remove non-config variables
    prepare_suffix = lambda v: re.sub(r'[^\w^ ]', '', v).replace(' ', '_') # Remove symbols and replace spaces
    suffix = "_".join(f"{prepare_suffix(str(v))}" for v in config.values()) # For filenames

    params.update(common_params)

    # Run test notebook
    path_ipynb = os.path.join(params['NOTEBOOKS_DIR'], f"{filename}.ipynb")
    out_path_ipynb = os.path.join(params['TESTS_ROOT_DIR'], f"{filename}_out_{suffix}_{params['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=params, outputs=params.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    pytest.failed = pytest.failed or exec_res['failed']

    # Remove test root directory if all tests were successfull
    if (notebook_kwargs == notebooks_params[-1]) and common_params['REMOVE_ROOT_DIR'] and not pytest.failed:
        shutil.rmtree(common_params['TESTS_ROOT_DIR'])

    # Terminal output
    with capsys.disabled():
        # Extract traceback if failed
        if exec_res['failed']:
            print(exec_res.get('traceback', ''))

        # Provide test outputs
        for k, v in exec_res['outputs'].items():
            message = v if isinstance(v, str) else json.dumps(v, indent=4)
            print(f"{k}:\n {message}\n")

        # Print test conclusion
        notebook_info = f"{params['DATESTAMP']} \'{filename}\'{' with config=' + str(config) if config else ''} was"
        if not exec_res['failed']:
            print(f"{notebook_info} executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"
