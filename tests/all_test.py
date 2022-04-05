""" Test script for running tests notebooks with provided parameters.

The behavior of all tests is controlled by the following constants that are declared in the `common_params` dict:

TESTS_ROOT_DIR : str
    Path to the directory for saving results and temporary files for all tests
    (executed notebooks, logs, data files like cubes, etc.).
    Note that in case of success this directory will be removed (if `REMOVE_ROOT_DIR` is True).
REMOVE_ROOT_DIR : bool
    Whether to remove tests root directory after execution in case of completion of tests without failures.
SHOW_FIGURES : bool
    Whether to show additional figures in the executed notebooks.
    Showing some figures can be useful for finding out the reason for the failure of tests.
VERBOSE : bool
    Whether to print in the terminal additional information from tests.

To add a new test you just need to add a new tuple (notebook path, test params) in the `notebooks_params` variable.
Also, this variable manages internal parameter values and outputs variables names for each individual test.
For more, read the comment above the `notebooks_params` initialization.

After all parameters initializations the main `test_run_notebook` function is called.
Under the hood, the function parses test kwargs, runs a test notebook with a given configuration,
catches execution information such as traceback, internal variables values and the test result,
and provides them to the terminal output.
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
BASE_DIR =  os.path.normpath(os.getenv("BASE_DIR", os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../seismiqb')))
TESTS_NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'tests/notebooks/') # path to the directory with tests notebooks
# TUTORIALS_DIR = os.path.join(BASE_DIR, 'tutorials/') # path to the directory with tutorials

# Initialize common parameters for all tests notebooks
common_params = {
    # Workspace constants and parameters
    'TESTS_ROOT_DIR': os.getenv('SEISMIQB_TESTS_ROOT_DIR', tempfile.mkdtemp(prefix='tests_root_dir_', dir=BASE_DIR)),
    'REMOVE_ROOT_DIR': os.getenv('SEISMIQB_TESTS_REMOVE_ROOT_DIR', 'True') == 'True',

    # Visualization and output parameters (these variables are used in notebooks)
    'SHOW_FIGURES': os.getenv('SEISMIQB_TESTS_SHOW_FIGURES', 'False') == 'True',
    'VERBOSE': os.getenv('SEISMIQB_TESTS_VERBOSE', 'True') == 'True'
}

# Initialize tests configurations
# The `notebooks_params` declares tests configurations in the (notebook path, params dict) format.
# The params dict contains 'inputs' and 'outputs' keys, where the 'inputs' is a dict with test parameters to pass to
# the test notebook, and the 'outputs' contains names of variables to return from the test notebook.
# Note that 'inputs' and 'outputs' are optional parameters.
geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']
notebooks_params = (
    # Tests configurations:
    # (notebook path, {'inputs': dict (optional), 'outputs': str or list of str (optional)})

    # CharismaMixin test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'charisma_test.ipynb'), {}),

    # SeismicGeometry test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_preparation.ipynb'),
     {'inputs': {'FORMATS': geometry_formats}}),

    *[(os.path.join(TESTS_NOTEBOOKS_DIR, 'geometry_test_data_format.ipynb'),
       {'inputs': {'FORMAT': data_format}, 'outputs': 'timings'}) for data_format in geometry_formats],

    # Horizon test
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_preparation.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_base.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_attributes.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_manipulations.ipynb'), {}),
    (os.path.join(TESTS_NOTEBOOKS_DIR, 'horizon_test_extraction.ipynb'), {'outputs': 'message'}),

    # # Tutorials : example for the future, tutorials notebooks needs some refactoring (data and paths changes)
    # (os.path.join(TUTORIALS_DIR, '01_Geometry_part_1.ipynb'), {})
)


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys):
    """ Run tests notebooks using kwargs and print outputs in the terminal."""
    # Parse kwargs
    path_ipynb, params = notebook_kwargs
    filename = os.path.splitext(os.path.basename(path_ipynb))[0]

    test_outputs = params.pop('outputs', None)
    params = params.pop('inputs', {})
    config = str(params) # For printing output info
    filename_suffix = f"{prepare_str(config=config)}" # Params config with symbols replacement for filenames

    params.update(common_params)

    # Run test notebook
    out_path_ipynb = os.path.join(params['TESTS_ROOT_DIR'],
                                  f"{filename}_out_{filename_suffix}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=params, outputs=test_outputs,
                            inputs_pos=2, working_dir=os.path.dirname(path_ipynb),
                            out_path_ipynb=out_path_ipynb, display_links=False)

    pytest.failed = pytest.failed or exec_res['failed']

    # Remove test root directory if all tests were successfull
    if (notebook_kwargs == notebooks_params[-1]) and params['REMOVE_ROOT_DIR'] and not pytest.failed:
        shutil.rmtree(params['TESTS_ROOT_DIR'])

    # Terminal output
    with capsys.disabled():
        # Extract traceback if failed
        if exec_res['failed']:
            print(exec_res.get('traceback', ''))

        # Print test outputs
        for k, v in exec_res.get('outputs', {}).items():
            message = v if isinstance(v, str) else json.dumps(v, indent=4)
            print(f"{k}:\n {message}\n")

        # Provide test conclusion
        notebook_info = f"`{filename}`{' with config=' + config if config!='{}' else ''}"
        if not exec_res['failed']:
            print(f"{notebook_info} was executed successfully.\n")
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
