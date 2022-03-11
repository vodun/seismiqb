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
    Note that in case of success this directory will be removed (if REMOVE_EXTRA_FILES is True).

And you can manage tests running with the parameter:

REMOVE_EXTRA_FILES : bool
    Whether to remove extra files after execution.
    Extra files are temporary files and saved files that relate to successful tests.

Outputs in saved execution notebooks are controlled with:

SCALE : int
    Figures scale.
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
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

common_params = {
    # Workspace constants
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),

    # Execution parameters
    'REMOVE_EXTRA_FILES': os.getenv('SEISMIQB_TEST_REMOVE_EXTRA_FILES') or True,

    # Visualization parameters
    'SCALE': os.getenv('SEISMIQB_TEST_SCALE') or 1,
    'SHOW_FIGURES': os.getenv('SEISMIQB_TEST_SHOW_FIGURES') or False,

    # Output parameters
    'VERBOSE': os.getenv('SEISMIQB_TEST_VERBOSE') or True,
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


# Create directory for temporary files and results
common_params['TESTS_ROOT_DIR'] = tempfile.mkdtemp(prefix='tests_root_dir_', dir='./')
pytest.failed = False


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys):
    """ Run tests notebooks using kwargs and print outputs in the terminal."""
    filename, params = notebook_kwargs
    config = params.copy()
    _ = config.pop('TEST_OUTPUTS', None)
    params.update(common_params)

    # Run test notebook
    prepare_suffix = lambda v: re.sub(r'[^\w^ ]', '', v).replace(' ', '_') # remove symbols and replace spaces
    suffix = "_".join(f"{prepare_suffix(str(v))}" for v in config.values())

    path_ipynb = os.path.join(params['NOTEBOOKS_DIR'], f"{filename}.ipynb")
    out_path_ipynb = os.path.join(params['TESTS_ROOT_DIR'], f"{filename}_out_{suffix}_{params['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=params, outputs=params.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    pytest.failed = pytest.failed or exec_res['failed']

    # Clear extra files
    if not exec_res['failed'] and params['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)

    if (notebook_kwargs == notebooks_params[-1]) and common_params['REMOVE_EXTRA_FILES'] and not pytest.failed:
        shutil.rmtree(common_params['TESTS_ROOT_DIR'])

    # Terminal output
    with capsys.disabled():
        # Extract traceback if failed
        if exec_res['failed']:
            print(exec_res.get('traceback', ''))

        # Test outputs
        for k, v in exec_res['outputs'].items():
            message = v if isinstance(v, str) else json.dumps(v, indent=4)
            print(f"{k}:\n {message}\n")

        # End of the running message
        notebook_info = f"{params['DATESTAMP']} \'{filename}\'{' with config=' + str(config) if config else ''} was"
        if not exec_res['failed']:
            print(f"{notebook_info} executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"
