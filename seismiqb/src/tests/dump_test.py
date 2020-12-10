""" Test for dumping data """
# pylint: disable=redefined-outer-name

import os

import pytest
import numpy as np

from seismiqb import SeismicGeometry

PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                    "../../../datasets/demo_data/CUBE_30_SCHLUM/Schlumberger_800_SLB_Force_Fault_Model_BlockId.segy")

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


@pytest.fixture(scope='module')
def make_hdf5_cube(tmp_path_factory):
    """ Creates hdf5 files with cubes in specified projections to use in hdf5-related tests """
    cubes = {}

    def _make_cube(projection):
        if projection not in cubes:
            g = SeismicGeometry(PATH)

            hdf5_path = os.path.join(tmp_path_factory.mktemp('cubes'), projection + '.hdf5')
            g.make_hdf5(hdf5_path, store_meta=True, cubes=projection, bar=False)

            cubes[projection] = hdf5_path

        return cubes[projection]

    return _make_cube


@pytest.mark.slow
@pytest.mark.parametrize("projections", ['ixh', 'i', 'x', 'h'])
def test_load_dump_segy(tmp_path, make_hdf5_cube, projections):
    """ Loads SEG-Y file, converts it to hdf5 with specified projections.
    Loads the new hdf5, then dumps it to SEG-Y again and compares it to the original file """
    hdf5_path = make_hdf5_cube(projections)
    g = SeismicGeometry(hdf5_path)

    out_path = os.path.join(tmp_path, 'out.sgy')
    g.make_sgy(path_hdf5=hdf5_path, path_segy=out_path, zip_result=False, remove_hdf5=False)

    compare_segy_files(PATH, out_path)
