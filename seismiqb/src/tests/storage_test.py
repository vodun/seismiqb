""" Test for seismic data stirages. """
# pylint: disable=missing-docstring
# pylint: disable=redefined-outer-name

import os

import pytest
import numpy as np

from seismiqb import SeismicGeometry, StorageHDF5

PATH = "/data/seismic_data/seismic_interpretation/CUBE_30_SCHLUM/Schlumberger_800_SLB_Force_Fault_Model_BlockId.segy"

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
            g.make_hdf5(hdf5_path, store_meta=True, projections=projection, pbar=False)

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

@pytest.fixture(scope='function')
def make_hdf5_storage(tmp_path, projections, shape=(100, 200, 300)):
    path = tmp_path.joinpath("cube.hdf5")
    if os.path.exists(path):
        os.remove(path)
    cube = StorageHDF5(path, mode='a', shape=shape, projections=projections)
    cube[:, :, :] = np.ones(shape)
    return cube, shape

@pytest.mark.parametrize("projections", [[0, 1, 2], [0, 1], [0, 2], [1, 2], [0], [1], [2]])
def test_hdf5_storage_creation(make_hdf5_storage, projections):
    cube, _ = make_hdf5_storage
    assert tuple(cube.projections) == tuple(projections)
    for axis in projections:
        if axis == 0:
            assert cube.cube_orientation(axis) == cube.file_hdf5['cube']
        elif axis == 1:
            assert cube.cube_orientation(axis) == cube.file_hdf5['cube_x']
        else:
            assert cube.cube_orientation(axis) == cube.file_hdf5['cube_h']

@pytest.mark.parametrize("projections", ['ixh', 'ix', 'ih', 'xh', 'i', 'x', 'h'])
def test_hdf5_storage_load_slide(make_hdf5_storage, projections):
    cube, shape = make_hdf5_storage
    for axis in range(3):
        if axis == 0:
            _shape = (shape[1], shape[2])
        elif axis == 1:
            _shape = (shape[0], shape[2])
        else:
            _shape = (shape[0], shape[1])
        assert cube.load_slide(0, axis=axis).shape == _shape

@pytest.mark.parametrize("projections", ['ixh', 'ix', 'ih', 'xh', 'i', 'x', 'h'])
def test_hdf5_storage_slicing(make_hdf5_storage, projections):
    cube, _ = make_hdf5_storage
    slices = (slice(20, 30), slice(30, 50), slice(50, 80))
    assert cube[slices[0], slices[1], slices[2]].shape == (10, 20, 30)

@pytest.mark.slow
@pytest.mark.parametrize("projections, axis",
    [('ixh', 0), ('ix', 0), ('ih', 0), ('xh', 1), ('i', 0), ('x', 1), ('h', 2)]
)
def test_optimal_cube(make_hdf5_storage, projections, axis):
    locations = slice(10), slice(20), slice(30)
    cube, _ = make_hdf5_storage
    assert cube.get_optimal_projection(locations) == axis
