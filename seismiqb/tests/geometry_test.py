""" Script for running notebook with SeismicGeometry tests."""
import glob
import json
import os
import pytest
import pprint
from datetime import date
from ..batchflow.utils_notebook import run_notebook


# Constants
DATESTAMP = date.today().strftime("%Y-%m-%d")
DROP_EXTRA_FILES = True
SHOW_TEST_ERROR_INFO = True
TESTS_SCRIPTS_DIR = os.getenv("TESTS_SCRIPTS_DIR", os.path.dirname(os.path.realpath(__file__))+'/')
TEST_FOLDER = os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_files/')
SHOW_MESSAGE = True

@pytest.fixture(scope="session")
def tests_notebook(tmpdir_factory):
    """ Run SeismicGeometry test notebook."""
    # Delete old test notebook results
    previous_output_files = glob.glob(os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test_out_*.ipynb'))

    for file in previous_output_files:
        os.remove(file)

    out_path_ipynb = tmpdir_factory.mktemp("notebooks") / "geometry_test_out_{DATESTAMP}.ipynb"

    # Tests execution
    _, tests_notebook = run_notebook(
        path=os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/geometry_test.ipynb'),
        nb_kwargs={
            'TEST_FOLDER': TEST_FOLDER,
            'NOTEBOOKS_FOLDER': os.path.join(TESTS_SCRIPTS_DIR, 'notebooks/'),
            'DATESTAMP': DATESTAMP,
            'DROP_EXTRA_FILES': DROP_EXTRA_FILES,
            'SHOW_TEST_ERROR_INFO': SHOW_TEST_ERROR_INFO
        },
        insert_pos=0,
        out_path_ipynb=out_path_ipynb,
        display_links=False,
        return_nb=True
    )
    return tests_notebook

def test_geometry(capsys, tests_notebook):
    # Get message and timing from the notebook cells (github don't save temporary files)
    print(tests_notebook)
    msg = tests_notebook['cells'][7]['outputs'][0]['text']
    timings = json.loads(tests_notebook['cells'][9]['outputs'][0]['text'].replace('\'', '\"'))

    with capsys.disabled():
        if SHOW_MESSAGE:
            print(msg, '\n')

        pp = pprint.PrettyPrinter()
        pp.pprint(timings)

        # End of the running message
        if timings['state']=='OK':
            print('Tests for SeismicGeometry were executed successfully.\n')
        else:
            print(f'SeismicGeometry tests failed.\n')
            assert False
