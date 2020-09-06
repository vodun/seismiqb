import numpy as np
from memory_profiler import profile


# making test configuration
def make_test_config():
    # test offsets
    horizon_offset = (15, 10)
    array_offset = (15, 17, 118)
    width = 15

    # test arrays
    array = np.arange(1500 * 2000 * 35).reshape((1500, 2000, 35))
    horizon_matrix = np.tile((125, 126, 127, 128, 129, 130), (700, 500))
    args = (horizon_matrix, horizon_offset, array, array_offset, width, -999)
    return args


def get_array_values_np_loop(horizon_matrix, horizon_offset, array, array_offset, width, fill_value):
    """ Function for cutting out array-values along a horizon surface.
    """
    # compute start and end-points of the ilines-xlines overlap between
    # array and horizon_matrix in horizon and array-coordinates
    horizon_offset, array_offset = np.array(horizon_offset), np.array(array_offset)
    horizon_max = horizon_offset[:2] + np.array(horizon_matrix.shape)
    array_max = np.array(array.shape[:2]) + array_offset[:2]
    overlap_shape = np.minimum(horizon_max[:2], array_max[:2]) - np.maximum(horizon_offset[:2], array_offset[:2])
    overlap_start = np.maximum(0, horizon_offset[:2] - array_offset[:2])
    heights_start = np.maximum(array_offset[:2] - horizon_offset[:2], 0)

    # recompute horizon-matrix in array-coordinates
    slc_array = [slice(l, h) for l, h in zip(overlap_start, overlap_start + overlap_shape)]
    slc_horizon = [slice(l, h) for l, h in zip(heights_start, heights_start + overlap_shape)]
    overlap_matrix = np.full(array.shape[:2], fill_value=fill_value, dtype=np.float32)
    overlap_matrix[slc_array] = horizon_matrix[slc_horizon]
    overlap_matrix -= array_offset[-1]

    # make the cut-array and fill it with array-data located on needed heights
    result = np.zeros(array.shape[:2] + (width, ))
    for i, surface_level in enumerate(np.array([overlap_matrix + shift for shift in range(-width // 2 + 1,
                                                                                          width // 2 + 1)])):
        mask = (surface_level >= 0) & (surface_level < array.shape[-1]) & (surface_level !=
                                                                           fill_value - array_offset[-1])
        mask_where = np.where(mask)
        result[mask_where[0], mask_where[1], i] = array[mask_where[0], mask_where[1],
                                                        surface_level[mask_where].astype(np.int)]

    return result

def get_array_values_np_no_loop(horizon_matrix, horizon_shift, array, array_shift, width=5, fill_value=-999):
    """ Get values from an external array along the horizon.
    """
    array_shift = np.array(array_shift)

    # compute start and end-points of the ilines-xlines overlap between
    # array and horizon_matrix in horizon and array-coordinates
    horizon_max = horizon_shift[:2] + np.array(horizon_matrix.shape)
    array_max = np.array(array.shape[:2]) + array_shift[:2]
    overlap_shape = np.minimum(horizon_max[:2], array_max[:2]) - np.maximum(horizon_shift[:2], array_shift[:2])
    overlap_start = np.maximum(0, horizon_shift[:2] - array_shift[:2])
    heights_start = np.maximum(array_shift[:2] - horizon_shift[:2], 0)

    # recompute horizon-matrix in array-coordinates
    slc_array = [slice(l, h) for l, h in zip(overlap_start, overlap_start + overlap_shape)]
    slc_horizon = [slice(l, h) for l, h in zip(heights_start, heights_start + overlap_shape)]
    overlap_matrix = np.full(array.shape[:2], fill_value=fill_value, dtype=np.float32)
    overlap_matrix[slc_array] = horizon_matrix[slc_horizon]
    overlap_matrix -= array_shift[-1]
    overlap_matrix = overlap_matrix[..., np.newaxis]
    overlap_matrix = overlap_matrix + np.arange(-width // 2 + 1, width // 2 + 1)

    # make the cut-array and fill it with array-data located on needed heights
    result = np.full(array.shape[:2] + (width, ), np.nan, dtype=np.float32)
    mask = ((overlap_matrix >= 0) & (overlap_matrix < array.shape[-1])
            & (overlap_matrix != fill_value - array_shift[-1] + np.arange(-width // 2 + 1, width // 2 + 1)))
    mask_where = np.where(mask)
    result[mask_where] = array[mask_where[0], mask_where[1], overlap_matrix[mask_where].astype(np.int)]

    return result


def test_loop():
    args = make_test_config()
    get_array_values_np_loop(*args)

def test_no_loop():
    args = make_test_config()
    get_array_values_np_no_loop(*args)