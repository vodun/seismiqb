""" Mixin for geometry conversions. """
import os

import numpy as np
import h5pickle as h5py

from batchflow import Notifier


class ConversionMixin:
    """ !!. """
    AXIS_TO_NAME = {0: 'projection_i', 1: 'projection_x', 2: 'projection_d'} # names of projections
    AXIS_TO_ORDER = {0: [0, 1, 2], 1: [1, 0, 2], 2: [2, 0, 1]}               # re-order axis so that `axis` is the first
    AXIS_TO_TRANSPOSE = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}           # revert the previous re-ordering


    # Quantization
    def compute_quantization_parameters(self, ranges=0.99, clip=True, center=False,
                                        seed=42, n_quantile_traces=100_000):
        """ Make bins for int8 quantization and convert value-stats.

        Parameters
        ----------
        ranges : str or sequence of two numbers
            Ranges to quantize data to. Available options are:
                - `q95`, `q99`, `q999` to clip data to respective quantiles.
                - `same` keep the same range of data.
        clip : bool
            Whether to clip data to selected ranges.
        center : bool
            Whether to make data have 0-mean before quantization.
        """
        if isinstance(ranges, float):
            qleft, qright = self.get_quantile([ranges, 1 - ranges])
            value = min(abs(qleft), abs(qright))
            ranges = (-value, +value)

        if center:
            ranges = tuple(item - self.v_mean for item in ranges)
        bins = np.histogram_bin_edges(None, bins=254, range=ranges).astype(np.float32)

        #
        def quantize(array):
            if center:
                array -= self.mean
            if clip:
                array = np.clip(array, *ranges)
            array = np.digitize(array, bins) - 128
            return array.astype(np.int8)

        def dequantize(array):
            array += 128
            array = bins[array]
            if center:
                array += self.mean
            return array.astype(np.float32)

        # Load subset of data to compute quantiles
        alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].ravel()
        indices = np.random.default_rng(seed=seed).choice(alive_traces_indices, size=n_quantile_traces)
        data = self.load_by_indices(indices)
        quantized_data = quantize(data)

        mean, std = quantized_data.mean(), quantized_data.std()
        quantile_values = np.quantile(quantized_data, q=self.quantile_support)
        quantile_values[0], quantile_values[-1] = -127, +128

        # Estimate quantization error
        dequantized_data = dequantize(quantized_data)
        quantization_error = np.mean(np.abs(dequantized_data - data)) / self.std

        return {
            'center': center, 'clip': clip,
            'ranges': ranges, 'bins': bins,
            'transform': quantize, 'dequantize': dequantize,
            'quantization_error': quantization_error,

            'min': -127, 'max': +128,
            'mean': mean, 'std': std,
            'quantile_values': quantile_values,
        }

    # Convert SEG-Y
    def convert(self, format='hdf5', path=None, postfix='', projections='ixd',
                quantize=False, ranges='q99', clip=True, center=False,
                store_meta=True, pbar=True, dataset_kwargs=None, **kwargs):
        """ Convert SEG-Y file to a more effective storage.

        Parameters
        ----------
        format : {'hdf5', 'qhdf5', 'blosc', 'qblosc}
            Format of storage to convert to: `blosc` takes less space, but a touch slower, than `hdf5`.
            Prefix `q` sets the `quantize` parameter to True.
        path : str
            If provided, then path to save file to.
            Otherwise, file is saved under the same name with different extension.
        postfix : str
            Optional string to add before extension. Used only if the `path` is not provided.
        projections : str
            Which projections of data to store: `i` for iline one, `x` for the crossline, `h` for depth.
        quantize : bool
            Whether to binarize data to `int8` dtype. `ranges`, `clip` and `center` define parameters of quantization.
            Binarization is done uniformly over selected `ranges` of values.
            If True, then `q` is appended to extension.
        ranges : str
            Ranges to quantize data to. Available options are:
                - `q95`, `q99`, `q999` to clip data to respective quantiles.
                - `same` keep the same range of data.
        clip : bool
            Whether to clip data to selected ranges.
        center : bool
            Whether to make data have 0-mean before quantization.
        store_meta : bool
            Whether to store meta near the save file.
        pbar : bool
            Whether to show progress bar during conversion.
        kwargs : dict
            Other parameters, passed directly to the file constructor of chosen format.
            If format is `blosc`:
                - `cname` for algorithm of compression. Default is `lz4hc`.
                - `clevel` for level of compression. Default is 6.
                - `shuffle` for bitshuffle during compression. Default is False.
        """
        #pylint: disable=import-outside-toplevel
        # Select format
        if format.startswith('q'):
            quantize = True
            format = format[1:]

        if format == 'hdf5':
            constructor, mode = h5py.File, 'w-'

        dataset_kwargs = dataset_kwargs or {}

        # Quantization
        if quantize:
            self.compute_quantization_parameters(ranges=ranges, clip=clip, center=center)
            dtype, transform = np.int8, self.quantize
        else:
            dtype, transform = np.float32, lambda array: array

        if path is None:
            fmt_prefix = 'q' if quantize else ''

            if postfix == '' and len(projections) < 3:
                postfix = '_' + projections

            path = os.path.join(os.path.dirname(self.path), f'{self.short_name}{postfix}.{fmt_prefix}{format}')

        # Remove file, if exists
        if os.path.exists(path):
            os.remove(path)

        # Create file and datasets inside
        with constructor(path, mode=mode, **kwargs) as file:
            total = (('i' in projections) * self.shape[0] +
                     ('x' in projections) * self.shape[1] +
                     ('d' in projections) * self.shape[2])
            progress_bar = Notifier(pbar, total=total)
            name = os.path.basename(path)

            for p in projections:
                axis = self.parse_axis(p)
                projection_name = self.AXIS_TO_NAME[axis]
                order = self.AXIS_TO_ORDER[axis]
                projection_shape = self.shape[order]
                projection = file.create_dataset(projection_name, shape=projection_shape, dtype=dtype,
                                                 chunks=(1, *projection_shape[1:]),
                                                 **dataset_kwargs)

                progress_bar.set_description(f'Converting to {name}, projection {p}')
                for idx in range(self.shape[axis]):
                    slide = self.load_slide(idx, axis=axis)
                    slide = transform(slide)
                    projection[idx, :, :] = slide

                    progress_bar.update()
            progress_bar.close()

        if store_meta:
            if format == 'hdf5':
                self.dump_meta(path=path)

        return path
        # from .base import SeismicGeometry
        # return SeismicGeometry(path)