""" Test for dumping data """

import os

import pytest
import numpy as np

from seismiqb import SeismicGeometry

PATH = "/data/acquisition_footprints/7-2_ftp_input.sgy"

def compare_segy_files(path_1, path_2):
    """ Checks that two SEG-Y files contain exactly same traces

    Parameters
    ----------
    path_1, path_2 : str
        paths to files
    """

    g1 = SeismicGeometry(path_1)
    g2 = SeismicGeometry(path_2)

    assert np.all(g1.cube_shape == g2.cube_shape)

    f1 = g1.segyfile
    f2 = g2.segyfile

    l1 = len(f1.header)
    l2 = len(f2.header)

    assert l1 == l2
    for i in range(l1):
        assert f1.header[i] == f2.header[i]

    n_ilines = g1.cube_shape[0]
    for i in range(n_ilines):
        s1 = g1.load_slide(i, axis=0)
        s2 = g2.load_slide(i, axis=0)

        assert np.allclose(s1, s2)


@pytest.mark.slow
@pytest.mark.parametrize("cubes", [
    # 'ixh',
    'i',
    'x', 'h'
])
def test_load_dump_segy(tmp_path, cubes):
    """ Loads SEG-Y file, converts it to hdf5 with specified cubes,
    then dumps to SEG-Y again and compares it to the original file """
    g = SeismicGeometry(PATH)

    hdf5_path = os.path.join(tmp_path, 'tmp.hdf5')
    g.make_hdf5(hdf5_path, store_meta=False, cubes=cubes)

    out_path = os.path.join(tmp_path, 'out.sgy')

    g.make_sgy(path_hdf5=hdf5_path, path_segy=out_path, from_cubes=cubes, zip_result=False, remove_hdf5=True)

    compare_segy_files(PATH, out_path)
