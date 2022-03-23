""" Test script for running tests notebooks.

The behavior of tests is parametrized by the following constants:

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
from nbtools import run_notebook


# Initialize base tests variables
pytest.failed = False
BASE_DIR =  os.path.normpath(os.getenv("BASE_DIR", os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')))
TESTS_NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'seismiqb/tests/notebooks/') # path to directory with tests notebooks
# TUTORIALS_DIR = os.path.join(BASE_DIR, 'tutorials/') # path to directory with tutorials

common_params = {
    # Workspace constants and parameters
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'TESTS_ROOT_DIR': tempfile.mkdtemp(prefix='tests_root_dir_', dir='./'),
    'REMOVE_ROOT_DIR': os.getenv('SEISMIQB_TEST_REMOVE_ROOT_DIR') or True,

    # Visualization and output parameters (these variables are used in notebooks)
    'SHOW_FIGURES': os.getenv('SEISMIQB_TEST_SHOW_FIGURES') or False,
    'VERBOSE': os.getenv('SEISMIQB_TEST_VERBOSE') or True
}

# Initialize tests configs
geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']
notebooks_params = (
    # (notebook path, test params dict)
    # CharismaMixin test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'charisma_test.ipynb'), {}),

    # SeismicGeometry test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_preparation.ipynb'), {'FORMATS': geometry_formats}),
    *[(os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_data_format.ipynb'),
       {'TEST_OUTPUTS': ['timings'], 'FORMAT': f}) for f in geometry_formats],

    # Horizon test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_preparation.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_base.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_attributes.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_manipulations.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_extraction.ipynb'), {'TEST_OUTPUTS': ['message']}),

    # # Tutorials : example for the future, tutorials notebooks needs some refactoring (data and paths changes)
    # (os.path.join(TUTORIALS_DIR, '01_Geometry_part_1.ipynb'), {})
)


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys):
    """ Run tests notebooks using kwargs and print outputs in the terminal."""
    path_ipynb, params = notebook_kwargs
    filename = os.path.splitext(os.path.basename(path_ipynb))[0]

    test_outputs = params.pop('TEST_OUTPUTS', []) # Non-config param
    config = str(params) # For printing outputs
    filename_suffix = f"{prepare_str(config=config)}" # Replace symbols

    params.update(common_params)

    # Run test notebook
    out_path_ipynb = os.path.join(params['TESTS_ROOT_DIR'],
                                  f"{filename}_out_{filename_suffix}_{params['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=params, outputs=test_outputs,
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    pytest.failed = pytest.failed or exec_res['failed']

    # Remove test root directory if all tests were successfull
    if (notebook_kwargs == notebooks_params[-1]) and params['REMOVE_ROOT_DIR'] and not pytest.failed:
        shutil.rmtree(params['TESTS_ROOT_DIR'])

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
        notebook_info = f"{params['DATESTAMP']} \'{filename}\'{' with config=' + config if config!='{}' else ''} was"
        if not exec_res['failed']:
            print(f"{notebook_info} executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"


# Helper method
def prepare_str(config):
    """ Create a part of a filename suffix from a configuration dict string.

    Under the hood we remove all symbols and replace spaces with underscores.
    """
    config = re.sub(r'[^\w^ ]', '', config) # Remove all symbols except spaces
    config = config.replace(' ', '_')
    return config
