""" Script for running notebook with SeismicGeometry tests."""
import sys
from datetime import date

sys.path.append('../../..')
from seismiqb.batchflow.utils_notebook import run_notebook

if __name__ == "__main__":
    TEST_FOLDER = './notebooks/geometry_test_files/'
    NOTEBOOKS_FOLDER = './notebooks/'
    DATESTAMP = date.today().strftime("%Y-%m-%d")

    out_path_ipynb = TEST_FOLDER + f'notebooks/geometry_test_out_{DATESTAMP}.ipynb'

    exec_info = run_notebook(
        path='./notebooks/geometry_test.ipynb',
        nb_kwargs={
            'TEST_FOLDER': TEST_FOLDER,
            'NOTEBOOKS_FOLDER': NOTEBOOKS_FOLDER,
            'DATESTAMP': DATESTAMP
        },
        insert_pos=1, 
        out_path_ipynb=out_path_ipynb,
        display_links=False
    )
