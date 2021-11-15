""" Script for running notebook with Field tests."""
import glob
import os
from datetime import date
import nbformat

from ..batchflow.utils_notebook import run_notebook

# Constants
# Workspace
DATESTAMP = date.today().strftime("%Y-%m-%d")
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
TEST_DIR = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks')

# Execution
DROP_EXTRA_FILES = True
SHOW_TEST_ERROR_INFO = True
SHOW_MESSAGE = True
GITHUB_MODE = True


def test_field(capsys, tmpdir):
    """ Run Field test notebook draft.

    This test runs ./notebooks/field_test_draft.ipynb test file and show execution message.

    Under the hood, this notebook create a fake seismic cube (Field), saves it and checks
    matrices savings and loadings in CHARISMA data format.
    """
    # Delete old test notebook results
    if GITHUB_MODE:
        SAVING_DIR = tmpdir.mkdir("notebooks")
        out_path_ipynb = SAVING_DIR.join(f"field_test_draft_out_{DATESTAMP}.ipynb")

    else:
        # Clear outdatted files
        previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/field_test_draft_out_*.ipynb'))

        for file in previous_output_files:
            os.remove(file)

        # Path to a new test noteboook result
        SAVING_DIR = TEST_DIR
        out_path_ipynb = os.path.join(TESTS_SCRIPTS_DIR, f'notebooks/field_test_draft_out_{DATESTAMP}.ipynb')

    # Tests execution
    exec_info = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/field_test_draft.ipynb'),
        nb_kwargs={
            'SEED': 10,
            'SAVING_DIR': SAVING_DIR,
            'DATESTAMP': DATESTAMP,
            'CUBE_SHAPE': (100, 100, 100),
            'DROP_EXTRA_FILES': DROP_EXTRA_FILES,
            'GITHUB_MODE': GITHUB_MODE
        },
        insert_pos=1,
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )

    if exec_info is True:
        msg = ['Draft tests for Field were executed successfully.\n']
    else:
        msg = ['Field draft tests execution failed.\n']

        if SHOW_TEST_ERROR_INFO:
            # Add error traceback into the message
            msg.append(extract_traceback(path_ipynb=out_path_ipynb))


    with capsys.disabled():
        # Output message
        if SHOW_MESSAGE:
            for line in msg:
                print(line)
        else:
            for line in msg:
                pass

        # Test exit
        if exec_info is True and line.find('success'):
            print()
        else:
            assert False, 'Field tests draft failed.\n'
