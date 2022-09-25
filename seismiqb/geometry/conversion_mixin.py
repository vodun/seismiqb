""" Mixin for geometry conversions. """
import os

import numpy as np
import h5pickle as h5py
import hdf5plugin

from batchflow import Notifier


class Quantizer:
    """ Class to hold parameters oand methods for (de)quantization. """
    def __init__(self, ranges, clip=True, center=False, mean=None, dtype=np.int8):
        self.ranges = ranges
        self.clip, self.center = clip, center
        self.mean = mean
        self.dtype = dtype

        self.bins = np.histogram_bin_edges(None, bins=254, range=ranges).astype(np.float32)

    def quantize(self, array):
        """ Quantize data: find the index of each element in the pre-computed bins and use it as the new value.
        Converts `array` to int8 dtype. Lossy.
        """
        if self.center:
            array -= self.mean
        if self.clip:
            array = np.clip(array, *self.ranges)
        array = np.digitize(array, self.bins) - 128
        return array.astype(self.dtype)

    def dequantize(self, array):
        """ Dequantize data: use each element as the index in the array of pre-computed bins.
        Converts `array` to float32 dtype. Unable to recover full information.
        """
        array += 128
        array = self.bins[array]
        if self.center:
            array += self.mean
        return array.astype(np.float32)

    def __call__(self, array):
        return self.quantize(array)


class ConversionMixin:
    """ Methods for converting data to other formats. """
    #pylint: disable=redefined-builtin
    PROJECTION_NAMES = {0: 'projection_i', 1: 'projection_x', 2: 'projection_d'}    # names of projections
    TO_PROJECTION_TRANSPOSITION = {0: [0, 1, 2], 1: [1, 0, 2], 2: [2, 0, 1]}        # re-order axis to given projection
    FROM_PROJECTION_TRANSPOSITION = {0: [0, 1, 2], 1: [1, 0, 2], 2: [1, 2, 0]}      # revert the previous re-ordering

    @staticmethod
    def compute_axis_transpositions(axis):
        """ Compute transpositions of original (inline, crossline, depth) axes to a given projection.
        Returns a transposition to that projection and from it.
        """
        return ConversionMixin.TO_PROJECTION_TRANSPOSITION[axis], ConversionMixin.FROM_PROJECTION_TRANSPOSITION[axis]


    # Quantization
    def compute_quantization_parameters(self, ranges=0.99, clip=True, center=False, dtype=np.int8,
                                        n_quantile_traces=100_000, seed=42):
        """ Compute parameters, needed for quantizing data to required range.
        Also evaluates quantization error by comparing subset of data with its dequantized quantized version.
        On the same subset, stats like mean, std and quantile values are computed.

        Parameters
        ----------
        ranges : float or sequence of two numbers
            Ranges to quantize data to.
            If float, then used as quantile to clip data to. If two numbers, then this exact range is used.
        clip : bool
            Whether to clip data to selected ranges.
        center : bool
            Whether to make data have 0-mean before quantization.
        n_quantile_traces : int
            Size of the subset to compute quantiles.
        seed : int
            Seed for quantile traces subset selection.

        Returns
        -------
        quantization_parameters : dict
            Dictionary with keys for stats and methods of data transformation.
            `'quantizer'` key is the instance, which can be `called` to quantize arbitrary array.
        """
        # Parse parameters
        if isinstance(ranges, float):
            qleft, qright = self.get_quantile([ranges, 1 - ranges])
            value = min(abs(qleft), abs(qright))
            ranges = (-value, +value)

        if center:
            ranges = tuple(item - self.v_mean for item in ranges)
        quantizer = Quantizer(ranges=ranges, clip=clip, center=center, mean=self.mean, dtype=dtype)

        # Load subset of data to compute quantiles
        alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].ravel()
        indices = np.random.default_rng(seed=seed).choice(alive_traces_indices, size=n_quantile_traces)
        data = self.load_by_indices(indices)
        quantized_data = quantizer.quantize(data)

        mean, std = quantized_data.mean(), quantized_data.std()
        quantile_values = np.quantile(quantized_data, q=self.quantile_support)
        quantile_values[0], quantile_values[-1] = -127, +128

        # Estimate quantization error
        dequantized_data = quantizer.dequantize(quantized_data)
        quantization_error = np.mean(np.abs(dequantized_data - data)) / self.std

        return {
            'ranges': ranges, 'center': center, 'clip': clip,

            'quantizer': quantizer,
            'transform': quantizer.quantize,
            'dequantize': quantizer.dequantize,
            'quantization_error': quantization_error,

            'min': -127, 'max': +127,
            'mean': mean, 'std': std,
            'quantile_values': quantile_values,
        }

    # Convert SEG-Y
    def convert(self, format='hdf5', path=None, postfix='', projections='ixd',
                quantize=False, quantization_parameters=None, dataset_kwargs=None,
                pbar='t', store_meta=True, **kwargs):
        """ Convert SEG-Y file to a more effective storage.

        Parameters
        ----------
        format : {'hdf5', 'qhdf5'}
            Format of storage to convert to. Prefix `q` sets the `quantize` parameter to True.
        path : str
            If provided, then path to save file to.
            Otherwise, file is saved under the same name with different extension.
        postfix : str
            Optional string to add before extension. Used only if the `path` is not provided.
        projections : str
            Which projections of data to store: `i` for the inline one, `x` for the crossline, `d` for depth.
        quantize : bool
            Whether to quantize data to `int8` dtype. If True, then `q` is appended to extension.
        quantization_parameters : dict, optional
            If provided, then used as parameters for quantization.
            Otherwise, parameters from the call to :meth:`compute_quantization_parameters` are used.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        store_meta : bool
            Whether to store meta in the same file.
        dataset_kwargs : dict, optional
            Parameters, passed directly to the dataset constructor.
            If not provided, we use the blosc compression with `lz4hc` compressor, clevel 6 and no bit shuffle.
        kwargs : dict
            Other parameters, passed directly to the file constructor.
        """
        #pylint: disable=import-outside-toplevel
        # Select format
        if format.startswith('q'):
            quantize = True
            format = format[1:]
        if format == 'hdf5':
            constructor, mode = h5py.File, 'w-'

        # Quantization
        if quantize:
            if quantization_parameters is None:
                quantization_parameters = self.compute_quantization_parameters()
            dtype, transform = np.int8, quantization_parameters['transform']
        else:
            dtype, transform = np.float32, lambda array: array

        # Default path: right next to the original file with new extension
        if path is None:
            fmt_prefix = 'q' if quantize else ''

            if postfix == '' and len(projections) < 3:
                postfix = '_' + projections

            path = os.path.join(os.path.dirname(self.path), f'{self.short_name}{postfix}.{fmt_prefix}{format}')

        # Dataset creation parameters
        if dataset_kwargs is None:
            dataset_kwargs = dict(hdf5plugin.Blosc(cname='lz4hc', clevel=6, shuffle=0))

        # Remove file, if exists
        if os.path.exists(path):
            os.remove(path)

        # Create file and datasets inside
        with constructor(path, mode=mode, **kwargs) as file:
            total = sum((letter in projections) * self.shape[idx]
                        for idx, letter in enumerate('ixd'))
            progress_bar = Notifier(pbar, total=total, ncols=110)
            name = os.path.basename(path)

            for p in projections:
                # Projection parameters
                axis = self.parse_axis(p)
                projection_name = self.PROJECTION_NAMES[axis]
                projection_transposition = self.TO_PROJECTION_TRANSPOSITION[axis]
                projection_shape = self.shape[projection_transposition]

                # Create dataset
                dataset_kwargs_ = {'chunks': (1, *projection_shape[1:]), **dataset_kwargs}
                projection = file.create_dataset(projection_name, shape=projection_shape, dtype=dtype,
                                                 **dataset_kwargs_)

                # Write data on disk
                progress_bar.set_description(f'Converting to {name}:{p}')
                for idx in range(self.shape[axis]):
                    slide = self.load_slide(idx, axis=axis)
                    slide = transform(slide)
                    projection[idx, :, :] = slide

                    progress_bar.update()
            progress_bar.close()

        # Save meta to the same file. If quantized, replace stats with the correct ones
        if store_meta:
            self.dump_meta(path=path)

            if quantize:
                for key in ['ranges', 'center', 'clip', 'quantization_error',
                            'min', 'max', 'mean', 'std', 'quantile_values']:
                    self.dump_meta_item(key=f'meta/{key}', value=quantization_parameters[key],
                                        path=path, overwrite=True)

        from .base import Geometry
        return Geometry.new(path)

    def repack_segy(self, path=None, format=8, transform=None, chunk_size=25_000, max_workers=4,
                     pbar='t', overwrite=True):
        """ Repack SEG-Y file with a different `format`: dtype of data values.
        Keeps the same binary header (except for the 3225 byte, which stores the format).
        Keeps the same header values for each trace: essentially, only the values of each trace are changed.

        The most common scenario of this function usage is to convert float32 SEG-Y into int8 one:
        the latter is a lot faster and takes ~4x less disk space at the cost of some information loss.

        Parameters
        ----------
        path : str, optional
            Path to save file to. If not provided, we use the path of the current cube with an added postfix.
        format : int
            Target SEG-Y format.
            Refer to :attr:`~.MemmapLoader.SEGY_FORMAT_TO_TRACE_DATA_DTYPE` for
            list of available formats and their data value dtype.
        transform : callable, optional
            Callable to transform data from the current file to the ones, saved in `path`.
            Must return the same dtype, as specified by `format`.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int or None
            Maximum number of parallel processes to spawn. If None, then the number of CPU cores is used.
        pbar : bool, str
            If bool, then whether to display progress bar.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        overwrite : bool
            Whether to overwrite existing `path` or raise an exception.
        """
        if format == 8 and transform is None:
            transform = self.compute_quantization_parameters()['transform']

        path = self.loader.convert(path=path, format=format, transform=transform,
                                   chunk_size=chunk_size, max_workers=max_workers, pbar=pbar, overwrite=overwrite)

        meta_path = path + '_meta'
        if overwrite and os.path.exists(meta_path):
            os.remove(meta_path)
        return path