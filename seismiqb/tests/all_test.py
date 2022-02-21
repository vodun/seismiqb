""" Test script for running controller notebooks for tests.

The behavior of tests is parametrized by the following constants:

DATESTAMP : str
    Execution date in "YYYY-MM-DD" format.
    Used for saving notebooks executions and temporary files.
TESTS_SCRIPTS_DIR : str
    Path to the directory with test .py scripts.
    Used as an entry point to the working directory.
NOTEBOOKS_DIR : str
    Path to the directory with test .ipynb files.
USE_TMP_OUTPUT_DIR: bool
    Whether to use pytest tmpdir as a workspace.
    If True, then all files are saved in temporary directories.
    If False, then all files are saved in local directories.
OUTPUT_DIR : str
    Path to the directory for saving results and temporary files
    (executed notebooks, logs, data files like cubes, etc.).
LOGS_DIR : str
    Path to the directory for saving log files (executed notebooks, timings, messages).
TEST_OUTPUTS : str or iterable of str
    List of notebook locals names that return to output.

And you can manage tests running with parameters:

REMOVE_OUTDATED_FILES: bool
    Whether to remove outdated files which relate to previous executions.
REMOVE_EXTRA_FILES : bool
    Whether to remove extra files after execution.
    Extra files are temporary files and execution saved files that relate to successful tests.
SHOW_MESSAGE : bool
    Whether to show a detailed tests execution message.
SHOW_TEST_ERROR_INFO : bool
    Whether to show error traceback in outputs.
    Notice that it only works with SHOW_MESSAGE = True.
SAVE_LOGS_REG_EXP : list of str
    A list of regular expressions for files which should be saved after a test execution.

Visualizations in saved execution notebooks are controlled with:

SCALE : int
    Figures scale.
SHOW_FIGURES : bool
    Whether to show additional figures.
    Showing some figures can be useful for finding out the reason for the failure of tests.

Text outputs in executed notebooks controlled with:

VERBOSE : bool
    Whether to print information about successful tests during the execution of the cycles.
"""
import os
from datetime import date
from glob import glob
import json
from itertools import product
import pytest

from .run_notebook import run_notebook
from .utils import remove_paths

# Initialize base tests variables
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')

tests_params = {
    # Workspace constants
    'DATESTAMP': date.today().strftime("%Y-%m-%d"),
    'TESTS_SCRIPTS_DIR': TESTS_SCRIPTS_DIR,
    'NOTEBOOKS_DIR': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
    'USE_TMP_OUTPUT_DIR': os.getenv('SEISMIQB_TEST_USE_TMP_OUTPUT_DIR') or True,

    # Execution parameters
    'REMOVE_OUTDATED_FILES': os.getenv('SEISMIQB_TEST_REMOVE_OUTDATED_FILES') or True,
    'REMOVE_EXTRA_FILES': os.getenv('SEISMIQB_TEST_REMOVE_EXTRA_FILES') or True,
    'SHOW_MESSAGE': os.getenv('SEISMIQB_TEST_SHOW_MESSAGE') or True,
    'SHOW_TEST_ERROR_INFO': os.getenv('SEISMIQB_TEST_SHOW_ERROR_INFO') or True,

    # Visualization parameters
    'SCALE': os.getenv('SEISMIQB_TEST_SCALE') or 1,
    'SHOW_FIGURES': os.getenv('SEISMIQB_TEST_SHOW_FIGURES') or False,

    # Output parameters
    'VERBOSE': os.getenv('SEISMIQB_TEST_VERBOSE') or True,
}

# Tests configurations
geometry_formats = ['sgy', 'hdf5', 'qhdf5', 'blosc', 'qblosc']

# Parameters for all tests stages (preparation, main, final)
all_tests_kwargs = {
    'geometry': {'TEST_OUTPUTS': ['states', 'timings'],
                  'FORMATS': geometry_formats},
    'charisma': {},
    'horizon': {'TEST_OUTPUTS': ['message']}
}
tests_names = all_tests_kwargs.keys()

# Iterables
# Helper
def get_tests_notebooks(tests_stage_name):
    """..!!.."""
    if tests_stage_name != 'main':
        notebooks = [x for x in test_notebooks_paths if x.find(tests_stage_name) != -1]
    else:
        notebooks = [x for x in test_notebooks_paths if (x.find('preparation') == -1) and (x.find('final') == -1)]

    params = []

    for test_name in tests_names:
        test_notebooks = [x for x in notebooks if x.find(test_name + '_test')!=-1] or [None]
        params.extend([(test_name, x) for x in test_notebooks])

    return params

test_notebooks_paths = glob(tests_params['NOTEBOOKS_DIR'] + '*.ipynb') # all tests notebooks

# Iterables for preparation tests
preparation_tests_notebooks = get_tests_notebooks(tests_stage_name='preparation')

# Iterables for main tests
main_tests_notebooks_kwargs = {'geometry_test_data_format': {'FORMAT': geometry_formats}}
main_tests_notebooks = get_tests_notebooks(tests_stage_name='main')

# Combine test_name, notebooks and notebooks kwargs into params configuration
main_tests_params = []
for test_name in tests_names:
    notebooks = [x[1] for x in main_tests_notebooks if x[0] == test_name]

    for notebook in notebooks:
        notebook_name = os.path.splitext(os.path.basename(notebook))[0]
        main_test_kwargs = main_tests_notebooks_kwargs.get(notebook_name, None)

        if main_test_kwargs:
            for k, values in main_test_kwargs.items():
                for value in values:
                    main_tests_params.append((test_name, notebook, {k: value}))
        else:
            main_tests_params.append((test_name, notebook, {}))

# Iterables for final tests
final_tests_notebooks = get_tests_notebooks(tests_stage_name='final')

# Globals tests states
pytest.states = {}


# Tests stages fixtures
@pytest.fixture(scope='session', params=preparation_tests_notebooks)
def run_preparation(request, tmpdir_factory):
    """..!!.."""
    test_name, path_ipynb = request.param
    test_kwargs = all_tests_kwargs[test_name]

    # Workspace preparation
    if tests_params['USE_TMP_OUTPUT_DIR']:
        test_kwargs['OUTPUT_DIR'] = tmpdir_factory.mktemp(f"{test_name}_test_files")
        test_kwargs['LOGS_DIR'] = tmpdir_factory.mktemp("logs")

    else:
        test_kwargs['OUTPUT_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'],
                                                 f"notebooks/{test_name}_test_files")
        test_kwargs['LOGS_DIR'] = os.path.join(tests_params['TESTS_SCRIPTS_DIR'], 'logs/')

    test_kwargs.update(tests_params)

    # Run preparation notebook if exists
    exec_res = {'failed': False} # Some tests haven't preparation notebooks
    if path_ipynb:
        exec_res = run_test_notebook(path_ipynb, test_kwargs)

    states = {
        test_name: {
            'test_kwargs': test_kwargs,
            'failed': exec_res['failed'],
            'outputs': exec_res.get('outputs', {})
        }
    }
    pytest.states.update(states)
    return exec_res, test_name, path_ipynb

@pytest.fixture(scope='session', params=main_tests_params)
def run_main_notebook(request):
    """..!!.."""
    # Extract params from iterables and run main test notebook with params
    test_name, path_ipynb, ipynb_kwargs = request.param
    test_kwargs, test_failed, test_outputs = pytest.states[test_name].values()

    test_kwargs.update(test_outputs) # For saving shared logs

    if not test_failed:
        # Extract iterable params configurations
        suffix = "_" + "_".join(str(v) for v in ipynb_kwargs.values())
        test_kwargs.update(ipynb_kwargs)

        # Run main test notebooks with params and save test state
        exec_res = {'failed': False}
        if path_ipynb:
            if ipynb_kwargs:
                print(f"Running `{os.path.basename(path_ipynb)}` test notebook with `{ipynb_kwargs}`.")

            exec_res = run_test_notebook(path_ipynb=path_ipynb, test_kwargs=test_kwargs,
                                        suffix=suffix)

            test_failed = test_failed or exec_res['failed']
            test_outputs.update(exec_res.get('outputs', {}))

            test_kwargs.update(test_outputs)

            pytest.states[test_name].update({
                'failed': test_failed,
                'outputs': test_outputs
            })
    else:
        exec_res = {'failed': True}

    return exec_res, test_name, path_ipynb, ipynb_kwargs

@pytest.fixture(scope='session', params=final_tests_notebooks)
def run_final_notebook(request):
    """..!!.."""
    test_name, path_ipynb = request.param
    test_kwargs, test_failed, test_outputs = pytest.states[test_name].values()

    # Run a final notebook if exists
    exec_res = {'failed': test_failed}
    if not test_failed and path_ipynb:
        exec_res = run_test_notebook(path_ipynb, test_kwargs)

        test_failed = test_failed or exec_res['failed']
        test_outputs.update(exec_res.get('outputs', {}))

        pytest.states[test_name].update({
            'failed': test_failed,
            'outputs': test_outputs
        })

        if test_kwargs['REMOVE_EXTRA_FILES'] and not test_kwargs['USE_TMP_OUTPUT_DIR']:
            remove_paths(paths=test_kwargs['OUTPUT_DIR'])

    return exec_res, test_name, path_ipynb


# Tests calls with terminal outputs
def test_run_all_preparation_notebooks(run_preparation, capsys):
    """..!!.."""
    exec_res, test_name, notebook_path = run_preparation

    message = f"Preparation {'`' + os.path.basename(notebook_path) + '` ' if notebook_path else ''}for {test_name} "\
              f"execution was {'failed' if exec_res['failed'] else 'successfull'}."

    print_exec_info(exec_res=exec_res, message=message, capsys=capsys)

def test_run_all_main_notebooks(run_main_notebook, capsys):
    """..!!.."""
    exec_res, test_name, notebook_path, kwargs = run_main_notebook

    message = f"Main notebook {'`' + os.path.basename(notebook_path) + '` ' if notebook_path else ''}"\
              f"{'with ' + str(kwargs) if kwargs else ''} "\
              f"for {test_name} execution was {'failed' if exec_res['failed'] else 'successfull'}."

    print_exec_info(exec_res=exec_res, message=message, capsys=capsys)

def test_run_all_final_notebooks(run_final_notebook, capsys):
    """..!!.."""
    exec_res, test_name, notebook_path = run_final_notebook

    message = f"Final stage {'`' + os.path.basename(notebook_path) + '` ' if notebook_path else ''}"\
                f"for {test_name} execution was {'failed' if exec_res['failed'] else 'successfull'}."

    print_exec_info(exec_res=exec_res, message=message, capsys=capsys)


@pytest.mark.parametrize("test_name", tests_names)
def test_finalize(test_name, capsys):
    """..!!.."""
    test_kwargs, test_failed, test_outputs = pytest.states[test_name].values()
    # Provide outputs to the terminal
    with capsys.disabled():
        print('\n' + test_kwargs['DATESTAMP'])

        # Tests output
        if test_kwargs['SHOW_MESSAGE']:
            for k, v in test_outputs.items():
                if isinstance(v, str):
                    print(f"{k}:\n\n{v}")
                else:
                    print(f"{k}:\n")
                    print(json.dumps(v, indent=4))

        # End of the running message
        if not test_failed:
            print(f"\'{test_name}\' tests were executed successfully.\n")
        else:
            assert False, f"\'{test_name}\' tests failed.\n"


# Helper method
def run_test_notebook(path_ipynb, test_kwargs, suffix=""):
    """..!!.."""
    # Run test notebook 
    file_name = os.path.splitext(os.path.basename(path_ipynb))[0]
    out_path_ipynb = os.path.join(test_kwargs['LOGS_DIR'], f"{file_name}_out{suffix}_{test_kwargs['DATESTAMP']}.ipynb")

    exec_res = run_notebook(path=path_ipynb, inputs=test_kwargs, outputs=test_kwargs.get('TEST_OUTPUTS', []),
                            inputs_pos=2, out_path_ipynb=out_path_ipynb, display_links=False)

    # Logs postprocessing: remove out_path_ipynb if all OK
    if not exec_res['failed'] and test_kwargs['REMOVE_EXTRA_FILES']:
        os.remove(out_path_ipynb)

    return exec_res

def print_exec_info(exec_res, message, capsys):
    """..!!.."""
    with capsys.disabled():
        print(message)

    if exec_res['failed']:
        if tests_params['SHOW_TEST_ERROR_INFO']:
            with capsys.disabled():
                print(exec_res.get('traceback', ''))
        assert False, message
