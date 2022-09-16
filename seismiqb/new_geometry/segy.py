""" Class to work with seismic data in SEG-Y format. """
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numba import njit

from batchflow import Notifier

from .base import Geometry
from .segyio_loader import SegyioLoader
from .memmap_loader import MemmapLoader



class GeometrySEGY(Geometry):
    """ Class to infer information about SEG-Y cubes and provide convenient methods for working with them.

    In order to initialize instance, one must supply `path`, `index_headers` and `additional_headers`:
        - `path` is a location of SEG-Y file
        - `index_headers` are used as the gather/trace unique identifier:
          for example, `INLINE_3D` and `CROSSLINE_3D` has a one-to-one correspondence with trace numbers.
          Another example is `FieldRecord` and `TraceNumber`.
        - `additional_headers` are also loaded.
    Default value of `index_headers` is ['INLINE_3D', 'CROSSLINE_3D'] with additional ['CDP_X', 'CDP_Y'],
    so that post-stack cube can be loaded by providing path only.

    For brevity, we use the 'inline/crossline' words to refer to the first/second indexing header in documentation
    and developer comments, as that is the most common scenario.

    To simplify indexing, we use ordinals of unique values of each indexing header pretty much everywhere after init.
    In the simplest case of regular structure, we can convert ordinals into unique values by using
    `value = value_min + ordinal * value_step`, where `value_min` and `value_step` are inferred from trace headers.

    For faster indexing of the traces we use indexing matrix, that maps
    `(ordinal_for_indexing_header_0, ordinal_for_indexing_header_1)` into the actual trace number to be loaded.

    At initialization or by manually calling method :meth:`collect_stats` we make a full pass through
    the cube in order to analyze distribution of amplitudes, storing global, spatial and depth-wise stats.
    They are available as attributes, e.g. `mean`, `mean_matrix` and `mean_vector`.

    Refer to the documentation of the base class :class:`Geometry` for more information about attributes and parameters.
    """
    # Headers to use as unique id of a trace
    INDEX_HEADERS_PRE = ('FieldRecord', 'TraceNumber')
    INDEX_HEADERS_POST = ('INLINE_3D', 'CROSSLINE_3D')
    INDEX_HEADERS_CDP = ('CDP_Y', 'CDP_X')

    # Headers to load from SEG-Y cube
    ADDITIONAL_HEADERS_PRE_FULL = ('FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE',
                                   'CDP', 'CDP_TRACE', 'offset')
    ADDITIONAL_HEADERS_POST_FULL = ('INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y')
    ADDITIONAL_HEADERS_POST = ('INLINE_3D', 'CROSSLINE_3D')


    def init(self, path, index_headers=INDEX_HEADERS_POST, additional_headers=ADDITIONAL_HEADERS_POST_FULL,
             loader_class=MemmapLoader, reload_headers=True, dump_headers=False, load_headers_params=None,
             collect_stats=True, recollect_stats=True, collect_stats_params=None, dump_meta=True,
             **kwargs):
        """ Init for SEG-Y geometry. The sequence of actions:
            - initialize loader instance
            - load headers by reading SEG-Y or reading from meta
            - compute additional attributes from indexing headers
            - validate structure of the coordinate system, created by the indexing headers
            - collect stats by full SEG-Y sweep or reading from meta
            - dump meta for future inits.
        """
        # TODO: switch `recollect_stats` to False
        # Store attributes
        self.index_headers = list(index_headers)
        self.additional_headers = list(additional_headers)
        self.index_length = len(index_headers)
        self.converted = False

        # Initialize loader
        self.loader = self._infer_loader_class(loader_class)(path)

        # Retrieve some of the attributes directly from the `loader`
        self.n_traces = self.loader.n_traces
        self.depth = self.loader.n_samples
        self.delay = self.loader.delay
        self.sample_rate = self.loader.sample_rate

        self.dtype = self.loader.dtype
        self.quantized = (self.dtype == np.int8)

        self.segy_path = self.loader.path
        self.segy_text = self.loader.text

        # Load all of the requested headers, either from SEG-Y directly or previously stored dump
        headers_to_load = list(set(index_headers) | set(additional_headers))

        if self.has_meta_item(key='headers') and not reload_headers:
            headers = self.load_meta_item(key='headers')
        else:
            load_headers_params = load_headers_params or {}
            headers = self.load_headers(headers_to_load, **load_headers_params)
            if dump_headers:
                self.dump_meta_item(key='headers', value=headers)
        self.headers = headers

        # Infer attributes based on indexing headers: values and coordinates
        self.add_index_attributes()

        # Collect amplitude stats, either by passing through SEG-Y or from previously stored dump
        if self.meta_exists and not recollect_stats:
            self.load_meta(names=self.PRESERVED + self.PRESERVED_LAZY)
            self.has_stats = True
        elif collect_stats:
            collect_stats_params = collect_stats_params or {}
            self.collect_stats(**collect_stats_params)
            self.has_stats = True

        # Dump inferred attributes to a separate file for later loads
        self.dump_meta()

    def _infer_loader_class(self, loader_class):
        """ Select appropriate loader class. """
        if isinstance(loader_class, type):
            return loader_class
        if 'seg' in loader_class:
            return SegyioLoader
        return MemmapLoader

    def load_headers(self, headers_to_load, reconstruct_tsf=True, chunk_size=25_000, max_workers=4, pbar=False):
        """ Load all of the requested headers into dataframe. """
        return self.loader.load_headers(headers_to_load, reconstruct_tsf=reconstruct_tsf,
                                        max_workers=max_workers, pbar=pbar)

    def add_index_attributes(self):
        """ Add attributes, based on the values of indexing headers. """
        # For each indexing headers compute set of its values, its sorted version,
        # and the mapping from each unique value to its ordinal in sorted list
        self.index_unsorted_uniques = [np.unique(self.headers[index_header])
                                       for index_header in self.index_headers]
        self.index_sorted_uniques = [np.sort(item) for item in self.index_unsorted_uniques]
        self.index_value_to_ordinal = [{value: i for i, value in enumerate(item)}
                                       for item in self.index_sorted_uniques]

        # Infer coordinates for indexing headers
        self.shifts = [np.min(item) for item in self.index_sorted_uniques]
        self.lengths = [len(item) for item in self.index_sorted_uniques]
        self.ranges = [(np.min(item), np.max(item)) for item in self.index_sorted_uniques]
        self.shape = np.array([*self.lengths, self.depth])

        # Check if indexing headers provide regular structure
        self.increments = []
        regular_structure = True
        for i, index_header in enumerate(self.index_headers):
            increments = np.diff(self.index_sorted_uniques[i])
            unique_increments = set(increments)

            if len(unique_increments) > 1:
                print(f'`{index_header}` has irregular spacing! {unique_increments}')
                regular_structure = False
            else:
                self.increments.append(increments[0])
        self.regular_structure = regular_structure

        # Create indexing matrix
        if self.index_length == 2:
            index_values = self.headers[self.index_headers].values
            index_ordinals = self.lines_to_ordinals(index_values)
            idx_0, idx_1 = index_ordinals[:, 0], index_ordinals[:, 1]

            dtype = np.int32 if self.n_traces < np.iinfo(np.int32).max else np.int64
            self.index_matrix = np.full(self.lengths, -1, dtype=dtype)
            self.index_matrix[idx_0, idx_1] = self.headers['TRACE_SEQUENCE_FILE'] - 1

            self.absent_traces_matrix = (self.index_matrix == -1).astype(np.bool_)


    # Collect stats
    def collect_stats(self, chunk_size=20, max_workers=16,
                      n_quantile_traces=100_000, quantile_precision=3, seed=42, pbar='t'):
        """ One sweep through the entire SEG-Y data to collects stats, which are available as instance attributes:
            - global: one number for the entire cube, e.g. `mean`
            - spatial: a matrix of values for each trace, e.g. `mean_matrix`
            - depth-wise: one value for each depth slice, e.g. `mean_vector`.
        Other than `mean`, we also collect `min`, `max` and `std`.
        Moreover, we compute a certain amount of quantiles: they are computed from a random subset of the traces.
        TODO: add `limits`?

        The traces are iterated over in chunks: chunking is performed along the first indexing header, e.g. `INLINE_3D`.
        Computation of stats is performed in multiple threads to speed up the process.

        Implementation detail: we store buffers for stats, e.g. `mean_matrix` in the instance itself.
        Each thread has the access to buffers and modifies them in-place.
        Moreover, even the underlying numba functions are using the same buffers in-place:
        this way we avoid unnecessary copies and data conversions.

        Parameters
        ----------
        chunk_size : int
            Number of full inlines to include in one chunk.
        max_workers : int
            Maximum number of threads for parallelization.
        n_quantile_traces : int
            Size of the subset to compute quantiles.
        quantile_precision : int
            Compute an approximate quantile for each value with that number of decimal places.
        seed : int
            Seed for quantile traces subset selection.
        pbar : bool, str
            If bool, then whether to display progress bar over the file sweep.
            If str, then type of progress bar to display: `'t'` for textual, `'n'` for widget.
        """
        # pylint: disable=too-many-statements, redefined-argument-from-local
        # Prepare chunks
        n = self.lengths[0]
        n_chunks, last_chunk_size = divmod(n, chunk_size)
        chunk_sizes = [chunk_size] * n_chunks
        if last_chunk_size:
            chunk_sizes += [last_chunk_size]
            n_chunks += 1

        chunk_starts = np.cumsum([0] + chunk_sizes[:-1])
        chunk_ends = np.cumsum(chunk_sizes)
        chunk_weights = np.array(chunk_sizes, dtype=np.float64) / n

        # Define buffers: chunked vectors
        self.min_vector_chunked = np.full((n_chunks, self.depth), np.inf, dtype=np.float32)
        self.max_vector_chunked = np.full((n_chunks, self.depth), -np.inf, dtype=np.float32)
        self.mean_vector_chunked = np.zeros((n_chunks, self.depth), dtype=np.float64)
        self.var_vector_chunked = np.zeros((n_chunks, self.depth), dtype=np.float64)

        # Define buffers: matrices
        self.min_matrix = np.full(self.lengths, np.inf, dtype=np.float32)
        self.max_matrix = np.full(self.lengths, -np.inf, dtype=np.float32)
        self.mean_matrix = np.zeros(self.lengths, dtype=np.float64)
        self.var_matrix = np.zeros(self.lengths, dtype=np.float64)

        # Read data in chunks, compute stats for each of them, store into buffer
        description = f'Collecting stats for `{self.name}`'
        with Notifier(pbar, total=n, desc=description, ncols=110) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                def callback(future):
                    chunk_size = future.result()
                    pbar.update(chunk_size)

                for chunk_i, (start, end) in enumerate(zip(chunk_starts, chunk_ends)):
                    future = executor.submit(self.collect_stats_chunk,
                                             start=start, end=end, chunk_i=chunk_i)
                    future.add_done_callback(callback)

        # Finalize vectors
        self.min_vector = np.average(self.min_vector_chunked, axis=0, weights=chunk_weights)
        self.max_vector = np.average(self.max_vector_chunked, axis=0, weights=chunk_weights)
        mean_vector = np.average(self.mean_vector_chunked, axis=0, weights=chunk_weights)
        var_vector = np.average(self.var_vector_chunked + (self.mean_vector_chunked - mean_vector)**2,
                                axis=0, weights=chunk_weights)

        self.mean_vector = mean_vector.astype(np.float32)
        self.std_vector = np.sqrt(var_vector).astype(np.float32)

        # Finalize matrices
        self.mean_matrix = self.mean_matrix.astype(np.float32)
        self.std_matrix = np.sqrt(self.var_matrix).astype(np.float32)
        self.dead_traces_matrix = (self.min_matrix == self.max_matrix).astype(np.bool_)

        # Clean-up redundant buffers
        del (self.min_vector_chunked, self.max_vector_chunked,
             self.mean_vector_chunked, self.var_vector_chunked,
             self.var_matrix)

        # Add scalar values
        self.min = self.min_matrix[~self.dead_traces_matrix].min()
        self.max = self.max_matrix[~self.dead_traces_matrix].max()
        self.mean = self.mean_matrix[~self.dead_traces_matrix].mean()

        n_dead_traces = np.sum(self.dead_traces_matrix)
        n_alive_traces = np.prod(self.lengths) - n_dead_traces
        self.std = np.sqrt((self.std_matrix[~self.dead_traces_matrix] ** 2).sum() / n_alive_traces)

        # Load subset of data to compute quantiles
        alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].ravel()
        indices = np.random.default_rng(seed=seed).choice(alive_traces_indices, size=n_quantile_traces)
        data = self.load_by_indices(indices)

        quantile_support = np.round(np.linspace(0, 1, num=10**quantile_precision+1),
                                    decimals=quantile_precision)
        quantile_values = np.quantile(data, q=quantile_support)
        quantile_values[0], quantile_values[-1] = self.min, self.max

        # Store stats of the subset to compare against fair ones
        self.subset_min = data.min()
        self.subset_max = data.max()
        self.subset_mean = data.mean()
        self.subset_std = data.std()
        self.quantile_precision = quantile_precision
        self.quantile_support, self.quantile_values = quantile_support, quantile_values

    def collect_stats_chunk(self, start, end, chunk_i):
        """ Read requested chunk, compute stats for it. """
        # Retrieve chunk data
        indices = self.index_matrix[start:end].ravel()

        data = self.load_by_indices(indices)
        data = data.reshape(end - start, self.lengths[1], self.depth)

        # Actually compute all of the stats. Modifies buffers in-place
        _collect_stats_chunk(data,
                             min_vector=self.min_vector_chunked[chunk_i],
                             max_vector=self.max_vector_chunked[chunk_i],
                             mean_vector=self.mean_vector_chunked[chunk_i],
                             var_vector=self.var_vector_chunked[chunk_i],
                             min_matrix=self.min_matrix[start:end],
                             max_matrix=self.max_matrix[start:end],
                             mean_matrix=self.mean_matrix[start:end],
                             var_matrix=self.var_matrix[start:end])
        return end - start


    # Data loading: arbitrary trace indices
    def load_by_indices(self, indices, limits=None, buffer=None):
        """ Read requested traces from SEG-Y file.
        Value `-1` is interpreted as missing trace, and corresponding traces are filled with zeros.

        Parameters
        ----------
        indices : sequence
            Indices (TRACE_SEQUENCE_FILE) of the traces to read.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        if buffer is None:
            limits = self.process_limits(limits)
            buffer = np.empty((len(indices), self.depth), dtype=self.dtype)[:, limits]
        else:
            buffer = buffer.reshape((len(indices), -1))

        if -1 in indices:
            mask = indices > 0
            buffer[~mask] = self.FILL_VALUE
            self.loader.load_traces(indices=indices[mask], limits=limits, buffer=buffer[mask])
        else:
            self.loader.load_traces(indices=indices, limits=limits, buffer=buffer)
        return buffer

    def load_depth_slice(self, index, buffer=None):
        """ !!. """
        if buffer is None:
            buffer = np.empty((1, self.n_traces), dtype=self.dtype)
        else:
            buffer = buffer.reshape((1, self.n_traces))

        buffer = self.loader.load_depth_slices([index], buffer=buffer)[0]
        if buffer.size == np.prod(self.lengths):
            buffer = buffer.reshape(self.lengths)
        else:
            buffer_ = np.zeros_like(self.dead_traces_matrix, dtype=np.float32)
            buffer_[~self.zero_traces] = buffer
            buffer = buffer_
        return buffer

    # Data loading: 2D
    def load_slide(self, index, axis=0, limits=None, buffer=None):
        """ Load one slide of data along specified axis.

        Parameters
        ----------
        index : int, str
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        limits : sequence of ints, slice, optional
            Slice of the data along the depth (last) axis.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        if axis in {0, 1}:
            index = self.get_slide_index(index=index, axis=axis)
            indices = np.take(self.index_matrix, indices=index, axis=axis)
            slide = self.load_by_indices(indices=indices, limits=limits, buffer=buffer)
        else:
            slide = self.load_depth_slice(index, buffer=buffer)
        return slide

    # Data loading: 3D
    def load_crop(self, locations, buffer=None):
        """ Load crop (3D subvolume) from the cube.

        Parameters
        ----------
        locations : sequence
            A triplet of slices to specify the location of a subvolume.
        buffer : np.ndarray, optional
            Buffer to read the data into. If possible, avoids copies.
        """
        indices = self.index_matrix[locations[0], locations[1]].ravel()
        buffer = self.load_by_indices(indices=indices, limits=locations[-1], buffer=buffer)

        shape = [((slc.stop or stop) - (slc.start or 0)) for slc, stop in zip(locations, self.shape)]
        buffer = buffer.reshape(shape)
        return buffer


@njit(nogil=True)
def _collect_stats_chunk(data,
                         min_vector, max_vector, mean_vector, var_vector,
                         min_matrix, max_matrix, mean_matrix, var_matrix):
    """ Compute stats of a 3D array: min, max, mean, variance.

    We use provided buffers to avoid unnecessary copies.
    We use buffers for mean and var to track the running sum of values / squared values.
    """
    shape = data.shape

    for i in range(shape[0]):
        for x in range(shape[1]):
            for d in range(shape[2]):
                # Read traces values
                trace_value = data[i, x, d]
                trace_value64 = np.float64(trace_value)

                # Update vectors
                min_vector[d] = min(min_vector[d], trace_value)
                max_vector[d] = max(max_vector[d], trace_value)
                mean_vector[d] += trace_value64
                var_vector[d] += trace_value64 ** 2

                # Update matrices
                min_matrix[i, x] = min(min_matrix[i, x], trace_value)
                max_matrix[i, x] = max(max_matrix[i, x], trace_value)
                mean_matrix[i, x] += trace_value64
                var_matrix[i, x] += trace_value64 ** 2

    # Finalize vectors
    area = shape[0] * shape[1]
    mean_vector /= area
    var_vector /= area
    var_vector -= mean_vector ** 2

    # Finalize matrices
    mean_matrix /= shape[2]
    var_matrix /= shape[2]
    var_matrix -= mean_matrix ** 2

    return (min_vector, max_vector, mean_vector, var_vector,
            min_matrix, max_matrix, mean_matrix, var_matrix)
