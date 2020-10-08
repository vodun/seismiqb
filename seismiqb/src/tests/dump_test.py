""" Test for dumping data """

import os

import psutil

import pytest
import numpy as np

from seismiqb import SeismicGeometry
# from seismicpro.batchflow import V, B, L, I
# from seismicpro.src import SeismicDataset, FieldIndex, TraceIndex, merge_segy_files

PATH = "/data/acquisition_footprints/7-2_ftp_input.sgy" #"os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../datasets/demo_data/teapot_dome_10.sgy')
PATH2 = "/data/acquisition_footprints/7-2_ftp_output.sgy"


def get_chunk_size(geometries, axis):
    mem_avail = psutil.virtual_memory().available

    if not isinstance(geometries, (list, tuple)):
        geometries = (geometries,)

    mem_needed_for_slide = 0
    for geom in geometries:
        l = list(geom.cube_shape)
        del l[axis]
        t = geom.load_trace(0)
        mem_needed_for_slide += t.itemsize * l[0] * l[1]

    return mem_avail // mem_needed_for_slide


# @pytest.mark.parametrize('path_1', [PATH, PATH2])
# @pytest.mark.parametrize('path_2', [PATH, PATH2])
def compare_segy_files(path_1, path_2):
    """ Checks that two SEG-Y files contain exacly same traces

    Parameters
    ----------
    path_1, path_2 : str
        paths to files
    """

    g1 = SeismicGeometry(path_1)
    g2 = SeismicGeometry(path_2)

    assert np.all(g1.cube_shape == g2.cube_shape)

    n_ilines = g1.cube_shape[0]
    for i in range(n_ilines):
        s1 = g1.load_slide(i, axis=0)
        s2 = g2.load_slide(i, axis=0)

        assert np.allclose(s1, s2)

    

    # n_ilines = g1.cube_shape[0]
    # chunk_size = get_chunk_size((g1, g2), 0)
    # chunk_size = min(chunk_size, n_ilines)

    # i = 0
    # while i < n_ilines:
    #     slices1 = load_slices(g1, axis=0, start=i, end=i + chunk_size)
    #     slices2 = load_slices(g2, axis=0, start=i, end=i + chunk_size)

    #     assert np.allclose(slices1, slices2)

    #     i += chunk_size


def test_load_dump_segy(tmp_path):
    g = SeismicGeometry(PATH)

    hdf5_path = os.path.join(tmp_path, 'tmp.hdf5')
    g.make_hdf5(hdf5_path, store_meta=False)

    out_path = os.path.join(tmp_path, 'out.sgy')

    g.make_sgy(path_hdf5=hdf5_path, path_segy=out_path, zip_result=False, remove_hdf5=True)

    compare_segy_files(PATH, out_path)
