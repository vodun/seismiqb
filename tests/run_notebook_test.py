""" Script for running tests notebooks with provided parameters.

Each test execution is controlled by the following constants that are declared in the `common_params` dict:

TESTS_ROOT_DIR : str
    Path to the directory for saving results and temporary files for all tests
    (executed notebooks, logs, data files like cubes, etc.).
    Note that the directory will be removed if `REMOVE_ROOT_DIR` is True and no one test failed.
SHOW_FIGURES : bool
    Whether to show additional figures in the executed notebooks.
    Showing some figures can be useful for finding out the reason for the tests failure.
VERBOSE : bool
    Whether to print in the terminal additional information from tests.

One more noteworthy constant for tests control is:

REMOVE_ROOT_DIR : bool
    Whether to remove `TESTS_ROOT_DIR` after execution in case of all tests completion without failures.

Another important script part is the `notebooks_params` variable which manages notebooks execution order,
internal parameter values and outputs variables names for each individual test.
To add a new test case you just need to add a configuration tuple (notebook_path, params_dict) in it, where
the `params_dict` may have optional keys 'inputs' and 'outputs':
    - 'inputs' is a dict with test parameters to pass to the test notebook execution,
    - 'outputs' contains names of variables to return from the test notebook.

After all parameters initializations the `test_run_notebook` function is called.
Under the hood, the function parses test arguments, runs test notebooks with given configurations,
catches execution information such as traceback and internal variables values, and provides them to the terminal output.
"""
import os
import json
import re
import shutil
import tempfile
import pytest
from nbtools import run_notebook


# Base tests variables for entire test process
pytest.failed = False
BASE_DIR =  os.path.normpath(os.getenv('BASE_DIR', os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../seismiqb')))
REMOVE_ROOT_DIR = bool(int(os.getenv('SEISMIQB_TESTS_REMOVE_ROOT_DIR', '1')))

# Parameters for each test notebooks
common_params = {
    'TESTS_ROOT_DIR': os.getenv('SEISMIQB_TESTS_ROOT_DIR', tempfile.mkdtemp(prefix='tests_root_dir_', dir=BASE_DIR)),
    'SHOW_FIGURES': bool(int(os.getenv('SEISMIQB_TESTS_SHOW_FIGURES', '0'))),
    'VERBOSE': bool(int(os.getenv('SEISMIQB_TESTS_VERBOSE', '1')))
}

TESTS_NOTEBOOKS_DIR = os.path.join(BASE_DIR, 'tests/notebooks/') # path to the directory with tests notebooks
# TUTORIALS_DIR = os.path.join(BASE_DIR, 'tutorials/')             # path to the directory with tutorials

geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']
notebooks_params = (
    # Tests configurations:
    # (notebook_path, {'inputs': dict (optional), 'outputs': str or list of str (optional)})

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

    # TODO: add tutorials
    # (os.path.join(TUTORIALS_DIR, '01_Geometry_part_1.ipynb'), {})
)


@pytest.mark.parametrize("notebook_kwargs", notebooks_params)
def test_run_notebook(notebook_kwargs, capsys, cleanup_fixture):
    """ Run tests notebooks using kwargs and print outputs in the terminal. """
    # Parse kwargs
    path_ipynb, params = notebook_kwargs
    filename = os.path.splitext(os.path.basename(path_ipynb))[0]

    outputs = params.pop('outputs', None)
    inputs = params.pop('inputs', {})
    inputs_repr = str(inputs) # for printing output info
    filename_suffix = f"{make_suffix(inputs_repr=inputs_repr)}" # inputs_repr in a correct format for file naming

    inputs.update(common_params)

    # Run test notebook
    out_path_ipynb = os.path.join(common_params['TESTS_ROOT_DIR'],
                                f"{filename}_out_{filename_suffix}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=inputs, outputs=outputs,
                            inputs_pos=2, working_dir=os.path.dirname(path_ipynb),
                            out_path_ipynb=out_path_ipynb, display_links=False)

    pytest.failed = pytest.failed or exec_res['failed']

    # Terminal output
    with capsys.disabled():
        # Extract traceback
        if exec_res['failed']:
            print(exec_res.get('traceback', ''))

        # Print test outputs
        for k, v in exec_res.get('outputs', {}).items():
            message = v if isinstance(v, str) else json.dumps(v, indent=4)
            print(f"{k}:\n{message}\n")

        # Provide test conclusion
        notebook_info = f"`{filename}`{' with inputs=' + inputs_repr if inputs_repr!='{}' else ''}"
        if not exec_res['failed']:
            print(f"{notebook_info} was executed successfully.\n")
        else:
            assert False, f"{notebook_info} failed.\n"

@pytest.fixture(scope="module")
def cleanup_fixture():
    """ Remove `TESTS_ROOT_DIR` in case of all tests completion without failures. """
    # Run all tests in the module
    yield
    # Remove TESTS_ROOT_DIR if all tests were successfull
    if REMOVE_ROOT_DIR and not pytest.failed:
        shutil.rmtree(common_params['TESTS_ROOT_DIR'])


# Helper method
def make_suffix(inputs_repr):
    """ Create a correct filename suffix from a string. Removes non-letter symbols and replaces spaces with underscores. """
    inputs_repr = re.sub(r'[^\w^ ]', '', inputs_repr) # Remove all non-letter symbols except spaces
    inputs_repr = inputs_repr.replace(' ', '_')
    return inputs_repr
