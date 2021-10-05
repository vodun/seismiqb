""" Tests for SeismicGeometry class. """
import os
import sys

import glob
import json
import pytest
import numpy as np

sys.path.append('../../..')
from seismiqb import SeismicGeometry

ATTRIBUTES_NAMES = ['snr', 'std_matrix', 'quality_map', 'quality_grid']

def check_data(geometry, data_array):
    geometry_data = geometry[:, :, :].ravel()

    if not geometry.quantized:
        assert (geometry_data == data_array.ravel()).all()
    else:
        corr = np.corrcoef(geometry_data, data_array.ravel())[0, 1]
        assert corr >=0.9
        
def test_attributes_loading(geometry):
    geometry.make_quality_map(
        [0.1], ['support_js', 'support_hellinger'], safe_strip=0
    ) # safe_strip=0 because data is a noise and otherwise haven't good traces

    for attr_name in ATTRIBUTES_NAMES:
        attr = geometry.__getattr__(key=attr_name)
        assert attr is not None

        attr = geometry.load_attribute(src=attr_name)
        assert attr is not None
        
def test_slide_loading(geometry, data_array):
    axis = np.random.randint(3)
    loc = np.random.randint(geometry.cube_shape[axis])
    data_slice = [slice(None) for i in range(axis)]
    data_slice.append(loc)

    geometry_slide = geometry.load_slide(loc=loc, axis=axis)
    data_slide = data_array[tuple(data_slice)]

    if not geometry.quantized:
        assert (geometry_slide == data_slide).all()
    else:
        corr = np.corrcoef(geometry_slide.ravel(), data_slide.ravel())[0, 1]
        assert corr >=0.9

def test_crop_loading(geometry, data_array):
    point = np.random.randint(geometry.cube_shape) // 2
    shape = np.random.randint((5, 5, 5), (200, 200, 200))
    locations = [slice(start_, np.clip(start_+shape_, 0, max_shape))
                        for start_, shape_, max_shape in zip(point, shape, geometry.cube_shape)]

    geometry_crop = geometry.load_crop(locations=locations)
    data_crop = data_array[tuple(locations)]

    if not geometry.quantized:
        assert (geometry_crop == data_crop).all()
    else:
        corr = np.corrcoef(geometry_crop.ravel(), data_crop.ravel())[0, 1]
        assert corr >=0.9
        
def check_visualizations(geometry, figsize):
    axis = np.random.randint(3)
    loc = np.random.randint(geometry.cube_shape[axis])

    geometry.show_slide(loc=loc, axis=axis, figsize=figsize)  
    geometry.show_histogram(hist_log=True, figsize=figsize)

    geometry.show(matrix='mean_matrix', figsize=figsize)
    geometry.show(matrix='snr', figsize=figsize)

    geometry.show_quality_map(figsize=figsize)    
    geometry.show_quality_grid(figsize=figsize)
    
def test_loading_timings(geometry, expected_timings, n_slide, n_crop, seed, test_folder, datestamp):
    timings_ = geometry.benchmark(n_slide=n_slide, n_crop=n_crop, use_cache=False, seed=seed)
    timings = {geometry.format: timings_}

    with open(test_folder + f'tmp/timings_{geometry.format}_{datestamp}.json', "w") as outfile:
        json.dump(timings, outfile)

    assert timings_['slide']['user'] == pytest.approx(expected_timings['slide']['user'], rel=0.3)
    assert timings_['crop']['user'] == pytest.approx(expected_timings['crop']['user'], rel=0.3)

    assert timings_['slide']['wall'] == pytest.approx(expected_timings['slide']['wall'], rel=0.3)
    assert timings_['crop']['wall'] == pytest.approx(expected_timings['crop']['wall'], rel=0.3)
    
def run_tests(geometry, data_array, expected_timings, figsize, n_slide, n_crop, seed, test_folder, datestamp):
    check_data(geometry=geometry, data_array=data_array)

    test_attributes_loading(geometry=geometry)
    test_slide_loading(geometry=geometry, data_array=data_array)
    test_crop_loading(geometry=geometry, data_array=data_array)
    
    check_visualizations(geometry=geometry, figsize=figsize)

    test_loading_timings(geometry=geometry, expected_timings=expected_timings,
                         n_slide=n_slide, n_crop=n_crop, seed=seed,
                         test_folder=test_folder, datestamp=datestamp)
