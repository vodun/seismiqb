""" Class to load headers/traces from SEG-Y via memory mapping. """
import os
from functools import partial
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numba import njit, prange

import segyio

from batchflow import Notifier

from .segyio_loader import SegyioLoader

ForPoolExecutor = None # TODO



class MemmapLoader(SegyioLoader):
    """ Custom reader/writer for SEG-Y files.
    Relies on memory mapping mechanism for actual reads of headers and traces.

    SEG-Y description
    -----------------
    Here we give a brief intro into SEG-Y format. Each SEG-Y file consists of:
        - file-wide information, in most cases the first 3600 bytes.
            - the first 3200 bytes are reserved for textual info about the file.
            Most software uses this to keep track of processing operations, date of creation, author, etc.
            - 3200-3600 bytes contain file-wide headers, which describe the number of traces,
            used format, depth, acquisition parameters, etc.
            - 3600+ bytes can be used to store the extended textual information, which is optional and indicated by
            one of the values in 3200-3600 bytes.

        - a sequence of traces, where each trace is a combination of header and its actual data.
            - header is the first 240 bytes and it describes the meta info about that trace:
            its coordinates in different types, the method of acquisition, etc.
            - data is an array of values, usually amplitudes, which can be stored in multiple numerical types.
            As the original SEG-Y is quite old (1975), one of those numerical formats is IBM float,
            which is very different from standard IEEE floats; therefore, a special caution is required to
            correctly decode values from such files.

    For the most part, SEG-Y files are written with constant size of each trace, although the standard itself allows
    for variable-sized traces. We do not work with such files.


    Implementation details
    ----------------------
    We rely on `segyio` to infer file-wide parameters.

    For headers and traces, we use custom methods of reading binary data.
    Main differences to `segyio C++` implementation:
        - we read all of the requested headers in one file-wide sweep, speeding up by an order of magnitude
        compared to the `segyio` sequential read of every requested header.
        Also, we do that in multiple processes across chunks.

        - a memory map over traces data is used for loading values. Avoiding redundant copies and leveraging
        `numpy` superiority allows to speed up reading, especially in case of trace slicing along the samples axis.
        This is extra relevant in case of loading horizontal (depth) slices.
    """
    def __init__(self, path, endian='big', strict=False, ignore_geometry=True):
        # Re-use most of the file-wide attributes from the `segyio` loader
        super().__init__(path=path, endian=endian, strict=strict, ignore_geometry=ignore_geometry)

        # Endian symbol for creating `numpy` dtypes
        self.endian_symbol = self.ENDIANNESS_TO_SYMBOL[endian]

        # Prefix attributes with `file`/`mmap` to avoid confusion.
        # TODO: maybe, add `segy` prefix to the attributes of the base class?
        self.file_format = self.metrics['format']
        self.file_traces_offset = self.metrics['trace0']
        self.file_trace_size = self.metrics['trace_bsize']

        # Dtype for data of each trace
        mmap_trace_data_dtype = self.SEGY_FORMAT_TO_TRACE_DATA_DTYPE[self.file_format]
        if isinstance(mmap_trace_data_dtype, str):
            mmap_trace_data_dtype = self.endian_symbol + mmap_trace_data_dtype
        self.mmap_trace_data_dtype = mmap_trace_data_dtype
        self.mmap_trace_data_size = self.depth if self.file_format != 1 else (self.depth, 4)

        # Dtype of each trace
        # TODO: maybe, use `np.uint8` as dtype instead of `np.void` for headers as it has nicer repr
        self.mmap_trace_dtype = np.dtype([('headers', np.void, self.TRACE_HEADER_SIZE),
                                          ('data', self.mmap_trace_data_dtype, self.mmap_trace_data_size)])
        self.data_mmap = self._construct_data_mmap()

    def _construct_data_mmap(self):
        """ Create a memory map with the first 240 bytes (headers) of each trace skipped. """
        return np.memmap(filename=self.path, mode="r", shape=self.n_traces, dtype=self.mmap_trace_dtype,
                         offset=self.file_traces_offset)["data"]


    # Headers
    def load_headers(self, headers, chunk_size=25_000, max_workers=None, pbar=False,
                     reconstruct_tsf=True, **kwargs):
        """ Load requested trace headers from a SEG-Y file for each trace into a dataframe.
        If needed, we reconstruct the `'TRACE_SEQUENCE_FILE'` manually be re-indexing traces.

        Under the hood, we create a memory mapping over the SEG-Y file, and view it with a special dtype.
        That dtype skips all of the trace data bytes and all of the unrequested headers, leaving only passed `headers`
        as non-void dtype.

        The file is read in chunks in multiple processes.

        Parameters
        ----------
        headers : sequence
            Names of headers to load.
        chunk_size : int
            Maximum amount of traces in each chunk.
        max_workers : int or None
            Maximum number of parallel processes to spawn. If None, then the number of CPU cores is used.
        pbar : bool, str
            If bool, then whether to display progress bar over the file sweep.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        reconstruct_tsf : bool
            Whether to reconstruct `TRACE_SEQUENCE_FILE` manually.
        """
        #pylint: disable=redefined-argument-from-local
        _ = kwargs
        if reconstruct_tsf and 'TRACE_SEQUENCE_FILE' in headers:
            headers = list(headers)
            headers.remove('TRACE_SEQUENCE_FILE')

        # Split the whole file into chunks no larger than `chunk_size`
        n_chunks, last_chunk_size = divmod(self.n_traces, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            chunk_sizes += [last_chunk_size]
        chunk_starts = np.cumsum([0] + chunk_sizes[:-1])

        # Construct mmap dtype: detailed for headers
        mmap_trace_headers_dtype = self._make_mmap_headers_dtype(headers)
        mmap_trace_dtype = np.dtype([*mmap_trace_headers_dtype,
                                     ('data', self.mmap_trace_data_dtype, self.mmap_trace_data_size)])

        # Parse `n_workers` and select an appropriate pool executor
        max_workers = max_workers or os.cpu_count()
        max_workers = min(len(chunk_sizes), max_workers)
        executor_class = ForPoolExecutor if max_workers == 1 else ProcessPoolExecutor

        # Iterate over chunks
        buffer = np.empty((self.n_traces, len(headers)), dtype=np.int32)

        with Notifier(pbar, total=self.n_traces) as pbar:
            with executor_class(max_workers=max_workers) as executor:

                def callback(future, start):
                    chunk_headers = future.result()
                    chunk_size = len(chunk_headers)
                    buffer[start : start + chunk_size] = chunk_headers
                    pbar.update(chunk_size)

                for start, chunk_size_ in zip(chunk_starts, chunk_sizes):
                    future = executor.submit(read_chunk, path=self.path,
                                             shape=self.n_traces, offset=self.file_traces_offset,
                                             dtype=mmap_trace_dtype, headers=headers,
                                             start=start, chunk_size=chunk_size_)
                    future.add_done_callback(partial(callback, start=start))
        dataframe = pd.DataFrame(buffer, columns=headers)

        if reconstruct_tsf:
            dataframe['TRACE_SEQUENCE_FILE'] = self.make_tsf_header()
        return dataframe

    def load_header(self, header, chunk_size=25_000, max_workers=None, pbar=False, **kwargs):
        """ Load exactly one header. """
        return self.load_headers(header=[header], chunk_size=chunk_size, max_workers=max_workers,
                                 pbar=pbar, reconstruct_tsf=False, **kwargs)

    def _make_mmap_headers_dtype(self, headers):
        """ Create list of `numpy` dtypes to view headers data.

        Defines a dtype for exactly 240 bytes, where each of the requested headers would have its own named subdtype,
        and the rest of bytes are lumped into `np.void` of certain lengths.

        Only the headers data should be viewed under this dtype: the rest of trace data (values)
        should be processed (or skipped) separately.

        We do not apply final conversion to `np.dtype` to the resulting list of dtypes so it is easier to append to it.

        Examples
        --------
        if `headers` are `INLINE_3D` and `CROSSLINE_3D`, which are 189-192 and 193-196 bytes, the return would be:
        >>> [('unused_0', numpy.void, 188),
        >>>  ('INLINE_3D', '>i4'),
        >>>  ('CROSSLINE_3D', '>i4'),
        >>>  ('unused_1', numpy.void, 44)]
        """
        header_to_byte = segyio.tracefield.keys
        byte_to_header = {val: key for key, val in header_to_byte.items()}
        start_bytes = sorted(header_to_byte.values())
        byte_to_len = {start: end - start
                       for start, end in zip(start_bytes, start_bytes[1:] + [self.TRACE_HEADER_SIZE + 1])}
        requested_headers_bytes = {header_to_byte[header] for header in headers}

        # Iterate over all headers
        # Unrequested headers are lumped into `np.void` of certain lengths
        # Requested   headers are each its own dtype
        dtype_list = []
        unused_counter, void_counter = 0, 0
        for byte, header_len in byte_to_len.items():
            if byte in requested_headers_bytes:
                if void_counter:
                    unused_dtype = (f'unused_{unused_counter}', np.void, void_counter)
                    dtype_list.append(unused_dtype)

                    unused_counter += 1
                    void_counter = 0

                header_name = byte_to_header[byte]
                value_dtype = 'i2' if header_len == 2 else 'i4'
                value_dtype = self.endian_symbol + value_dtype
                header_dtype = (header_name, value_dtype)
                dtype_list.append(header_dtype)
            else:
                void_counter += header_len

        if void_counter:
            unused_dtype = (f'unused_{unused_counter}', np.void, void_counter)
        dtype_list.append(unused_dtype)
        return dtype_list

    # Data loading: traces
    def load_traces(self, indices, limits=None, buffer=None):
        """ Load traces by their indices.
        Under the hood, we use a pre-made memory mapping over the file, where trace data is viewed with a special dtype.
        Regardless of the numerical dtype of SEG-Y file, we output IEEE float32:
        for IBM floats, that requires an additional conversion.

        Parameters
        ----------
        indices : sequence
            Indices (TRACE_SEQUENCE_FILE) of the traces to read.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth axis.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        limits = self.process_limits(limits)

        if self.file_format != 1:
            traces = self.data_mmap[indices, limits]
        else:
            traces = self.data_mmap[indices, limits.start:limits.stop]
            if limits.step != 1:
                traces = traces[:, ::limits.step]
            traces = self._ibm_to_ieee(traces)

        if buffer is None:
            return np.require(traces, dtype=np.float32, requirements='C')
        buffer[:] = traces
        return buffer

    # Data loading: depth slices
    def load_depth_slices(self, indices, buffer=None):
        """ Load horizontal (depth) slices of the data.
        Requires a ~full sweep through SEG-Y, therefore is slow.

        Parameters
        ----------
        indices : sequence
            Indices (ordinals) of the depth slices to read.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        depth_slices = self.data_mmap[:, indices]
        if self.file_format == 1:
            depth_slices = self._ibm_to_ieee(depth_slices)
        depth_slices = depth_slices.T

        if buffer is None:
            return np.require(depth_slices, dtype=np.float32, requirements='C')
        buffer[:] = depth_slices
        return buffer

    def _ibm_to_ieee(self, array):
        """ Convert IBM floats to regular IEEE ones. """
        array_bytes = (array[:, :, 0], array[:, :, 1], array[:, :, 2], array[:, :, 3])
        if self.endian in {"little", "lsb"}:
            array_bytes = array_bytes[::-1]
        return ibm_to_ieee(*array_bytes)


def read_chunk(path, shape, offset, dtype, headers, start, chunk_size):
    """ Read headers from one chunk.
    We create memory mapping anew in each worker, as it is easier and creates no significant overhead.
    """
    # mmap is created over the entire file as
    # creating data over the requested chunk only does not speed up anything
    mmap = np.memmap(filename=path, mode="r", shape=shape, offset=offset, dtype=dtype)

    buffer = np.empty((chunk_size, len(headers)), dtype=np.int32)
    for i, header in enumerate(headers):
        buffer[:, i] = mmap[header][start : start + chunk_size]
    return buffer



@njit(nogil=True, parallel=True)
def ibm_to_ieee(hh, hl, lh, ll):
    """ Convert 4 arrays representing individual bytes of IBM 4-byte floats into a single array of floats.
    Input arrays are ordered from most to least significant bytes and have `np.uint8` dtypes.
    The result is returned as an `np.float32` array.
    """
    res = np.empty_like(hh, dtype=np.float32)
    for i in prange(res.shape[0]):  # pylint: disable=not-an-iterable
        for j in prange(res.shape[1]):  # pylint: disable=not-an-iterable
            mant = (((np.int32(hl[i, j]) << 8) | lh[i, j]) << 8) | ll[i, j]
            if hh[i, j] & 0x80:
                mant = -mant
            exp16 = (np.int8(hh[i, j]) & np.int8(0x7f)) - 70
            res[i, j] = mant * 16.0**exp16
    return res
