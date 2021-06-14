""" Methods to save data as seismic cubes in different formats. """
import os
import shutil
from tqdm.auto import tqdm

import numpy as np
import h5pickle as h5py
import segyio

from ..utils import make_axis_grid


class ExportMixin:
    """ Container for methods to save data as seismic cubes in different formats. """
    @classmethod
    def create_file_from_iterable(cls, src, shape, window, stride, dst=None, agg=None, threshold=None):
        """ Aggregate multiple chunks into file with 3D cube.

        Parameters
        ----------
        src : iterable
            Each item is a tuple (position, array) where position is a 3D coordinate of the left upper array corner.
        shape : tuple
            Shape of the resulting array.
        window : tuple
            Chunk shape.
        stride : tuple
            Stride for chunks. Values in overlapped regions will be aggregated.
        dst : str or None, optional
            Path to the resulting hdf5 or blosc file. If None, function will return numpy array with predictions.
        agg : 'mean', 'min' or 'max' or None, optional
            The way to aggregate values in overlapped regions. None means that new chunk will rewrite
            previous value in cube.
        threshold : float or None, optional
            If not None, threshold to transform values into [0, 1]. Default is None.
        """
        shape = np.array(shape)
        window = np.array(window)
        stride = np.array(stride)

        if dst is None:
            dst = np.zeros(shape)
        else:
            ext = os.path.splitext(dst)[1][1:]
            if ext == 'hdf5':
                file, dtype, transform = h5py.File(dst, 'a'), np.float32, lambda array: array
            else:
                from .blosc import BloscFile
                if ext == 'blosc':
                    file, dtype, transform = BloscFile(dst, 'w'), np.float32, lambda array: array
                elif ext == 'qblosc':
                    file, dtype, transform = BloscFile(dst, 'w'), np.int8, cls.proba_to_int
            dst = file.create_dataset('cube_i', shape, dtype=dtype)
            cube_x = file.create_dataset('cube_x', shape[[1, 2, 0]], dtype=dtype)
            cube_h = file.create_dataset('cube_h', shape[[2, 0, 1]], dtype=dtype)

        lower_bounds = [make_axis_grid((0, shape[i]), stride[i], shape[i], window[i]) for i in range(3)]
        lower_bounds = np.stack(np.meshgrid(*lower_bounds), axis=-1).reshape(-1, 3)
        upper_bounds = lower_bounds + window
        grid = np.stack([lower_bounds, upper_bounds], axis=-1)

        for position, chunk in src:
            chunk = transform(chunk)
            slices = tuple([slice(position[i], position[i]+chunk.shape[i]) for i in range(3)])
            _chunk = dst[slices]
            if agg in ('max', 'min'):
                chunk = np.maximum(chunk, _chunk) if agg == 'max' else np.minimum(chunk, _chunk)
            elif agg == 'mean':
                grid_mask = np.logical_and(
                    grid[..., 1] >= np.expand_dims(position, axis=0),
                    grid[..., 0] < np.expand_dims(position + window, axis=0)
                ).all(axis=1)
                agg_map = np.zeros_like(chunk)
                for chunk_slc in grid[grid_mask]:
                    _slices = [slice(
                        max(chunk_slc[i, 0], position[i]) - position[i],
                        min(chunk_slc[i, 1], position[i] + window[i]) - position[i]
                    ) for i in range(3)]
                    agg_map[tuple(_slices)] += 1
                chunk = chunk / agg_map + _chunk
                if dtype == np.int8:
                    chunk = np.clip(chunk, -128, 127).astype(np.int8)
            dst[slices] = chunk
        if isinstance(dst, np.ndarray):
            if threshold is not None:
                dst = (dst > threshold).astype(int)
        elif ext == 'hdf5':
            for i in range(0, dst.shape[0], window[0]):
                slide = dst[i:i+window[0]]
                if threshold is not None:
                    slide = (slide > threshold).astype(int)
                    dst[i:i+window[0]] = slide
                cube_x[:, :, i:i+window[0]] = slide.transpose((1, 2, 0))
                cube_h[:, i:i+window[0]] = slide.transpose((2, 0, 1))
        return dst


    def make_sgy(self, path_hdf5=None, path_spec=None, postfix='',
                 remove_hdf5=False, zip_result=True, path_segy=None, pbar=False):
        """ Convert POST-STACK HDF5 cube to SEG-Y format with supplied spec.

        Parameters
        ----------
        path_hdf5 : str
            Path to load hdf5 file from.
        path_spec : str
            Path to load segy file from with geometry spec.
        path_segy : str
            Path to store converted cube. By default, new cube is stored right next to original.
        postfix : str
            Postfix to add to the name of resulting cube.
        """
        path_segy = path_segy or (os.path.splitext(path_hdf5)[0] + postfix + '.sgy')
        if not path_spec:
            if hasattr(self, 'segy_path'):
                path_spec = self.segy_path
            else:
                path_spec = os.path.splitext(self.path) + '.sgy'

        # By default, if path_hdf5 is not provided, `temp.hdf5` next to self.path will be used
        if path_hdf5 is None:
            path_hdf5 = os.path.join(os.path.dirname(self.path), 'temp.hdf5')

        with h5py.File(path_hdf5, 'r') as src:
            cube_hdf5 = src['cube_i']

            from .base import SeismicGeometry #pylint: disable=import-outside-toplevel
            geometry = SeismicGeometry(path_spec)
            segy = geometry.segyfile

            spec = segyio.spec()
            spec.sorting = None if segy.sorting is None else int(segy.sorting)
            spec.format = None if segy.format is None else int(segy.format)
            spec.samples = range(self.depth)

            idx = np.stack(geometry.dataframe.index)
            ilines, xlines = self.load_meta_item('ilines'), self.load_meta_item('xlines')

            i_enc = {num: k for k, num in enumerate(ilines)}
            x_enc = {num: k for k, num in enumerate(xlines)}

            spec.ilines = ilines
            spec.xlines = xlines

            with segyio.create(path_segy, spec) as dst_file:
                # Copy all textual headers, including possible extended
                for i in range(1 + segy.ext_headers):
                    dst_file.text[i] = segy.text[i]
                dst_file.bin = segy.bin

                for c, (i, x) in enumerate(tqdm(idx, disable=(not pbar))):
                    locs = tuple([i_enc[i], x_enc[x], slice(None)])
                    dst_file.header[c] = segy.header[c]
                    dst_file.trace[c] = cube_hdf5[locs]
                dst_file.bin = segy.bin
                dst_file.bin[segyio.BinField.Traces] = len(idx)

        if remove_hdf5:
            os.remove(path_hdf5)

        if zip_result:
            dir_name = os.path.dirname(os.path.abspath(path_segy))
            file_name = os.path.basename(path_segy)
            shutil.make_archive(os.path.splitext(path_segy)[0], 'zip', dir_name, file_name)

    @classmethod
    def proba_to_int(cls, array):
        return (array * 255 - 128).astype(np.int8)