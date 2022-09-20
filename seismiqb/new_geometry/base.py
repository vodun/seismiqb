""" Base class for working with seismic data. """
import os
import sys
from textwrap import dedent

import numpy as np
from scipy.interpolate import interp1d

from .benchmark_mixin import BenchmarkMixin
from .conversion_mixin import ConversionMixin
from .export_mixin import ExportMixin
from .meta_mixin import MetaMixin

from ..utils import CacheMixin, TransformsMixin, select_printer, transformable
from ..plotters import plot



class Geometry(BenchmarkMixin, CacheMixin, ConversionMixin, ExportMixin, MetaMixin, TransformsMixin):
    """ Class to infer information about seismic cube in various formats and provide format agnostic interface to them.

    During the SEG-Y processing, a number of statistics are computed. They are saved next to the cube under the
    `.segy_meta` extension, so that subsequent loads (in, possibly, other formats) don't have to recompute them.
    Most of them are loaded at initialization, but the most memory-intensive ones are loaded on demand.
    For more details about meta, refer to :class:`MetaMixin` documentation.

    Based on the extension of the path, a different subclass is used to implement key methods for data indexing.
    Currently supported extensions are SEG-Y and TODO:
    The last two are created by converting the original SEG-Y cube.
    During the conversion, an extra step of `int8` quantization can be performed to reduce the disk usage.

    Independent of the exact format, `SeismicGeometry` provides the following:
        - attributes to describe shape and structure of the cube like `shape` and `lengths`,
        as well as exact values of file-wide headers, for example, `depth`, `delay` and `sample_rate`.

        - method :meth:`collect_stats` to infer information about the amplitudes distribution:
        under the hood, we make a full pass through the cube data to collect global, spatial and depth-wise stats.

        - :meth:`load_slide` (2D entity) or :meth:`load_crop` (3D entity) methods to load data from the cube:
            - :meth:`load_slide` takes an ordinal index of the slide and its axis;
            - :meth:`load_crop` works off of complete location specification (triplet of slices).

        - textual representation of cube geometry: method `print` shows the summary of an instance with
        information about its location and values; `print_textual` allows to see textual header from a SEG-Y.

        - visual representation of cube geometry:
            - :meth:`show` to display top view on cube with computed statistics;
            - :meth:`show_slide` to display front view on various slices of data.

    Parameters
    ----------
    path : str
        Path to seismic cube. Supported formats are `segy`, TODO.
    meta_path : str, optional
        Path to pre-computed statistics. If not provided, use the same as `path` with `.meta` extension.

    SEG-Y parameters
    ----------------
    TODO

    HDF5 parameters
    ---------------
    TODO
    """
    # Value to use in dead traces
    FILL_VALUE = 0.0

    # Attributes to store in a separate file with meta
    PRESERVED = [ # loaded at instance initialization
        # Crucial geometry properties
        'depth', 'delay', 'sample_rate', 'shape',
        'shifts', 'lengths', 'ranges', 'increments', 'regular_structure',
        'index_matrix', 'absent_traces_matrix', 'dead_traces_matrix',
        'n_alive_traces', 'n_dead_traces',

        # Additional info from SEG-Y
        'segy_path', 'segy_text',
        'rotation_matrix', 'area',

        # Scalar stats for cube values: computed for the entire SEG-Y / its subset
        'min', 'max', 'mean', 'std', 'n_value_uniques',
        'subset_min', 'subset_max', 'subset_mean', 'subset_std',
        'quantile_precision', 'quantile_support', 'quantile_values',
    ]

    PRESERVED_LAZY = [ # loaded at the time of the first access
        'min_vector', 'max_vector', 'mean_vector', 'std_vector',
        'min_matrix', 'max_matrix', 'mean_matrix', 'std_matrix',
    ]


    def __init__(self, path, meta_path=None, use_line_cache=False, **kwargs):
        # Path to the file
        self.path = path

        # Names
        self.name = os.path.basename(self.path)
        self.short_name, self.format = os.path.splitext(self.name)

        # Meta
        self.meta_path = meta_path
        self.meta_list_loaded = set()
        self.meta_list_failed_to_dump = set()

        # Cache
        self.use_line_cache = use_line_cache

        # Lazy properties
        self._quantile_interpolator = None

        # Init from subclasses
        self._init_kwargs = kwargs
        self.init(path, **kwargs)


    # Redefined protocols
    def __getattr__(self, key):
        """ Load item from stored meta. """
        if key in self.PRESERVED_LAZY and self.meta_exists and self.has_meta_item(key) and key not in self.__dict__:
            return self.load_meta_item(key)
        return object.__getattribute__(self, key)

    def __getnewargs__(self):
        return (self.path, )

    def __getstate__(self):
        self.reset_cache()
        state = self.__dict__.copy()
        for name in ['file', 'axis_to_projection']:
            if name in state:
                state.pop(name)
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

        self.init(self.path, **self._init_kwargs)


    # Data loading
    def __getitem__(self, key):
        """ Slice the cube using the usual `NumPy`-like semantics. """
        key, axis_to_squeeze = self.process_key(key)

        crop = self.load_crop(key)
        if axis_to_squeeze:
            crop = np.squeeze(crop, axis=tuple(axis_to_squeeze))
        return crop

    def process_key(self, key):
        """ Convert tuple of slices/ints into locations. """
        # Convert to list
        if isinstance(key, (int, slice)):
            key = [key]
        elif isinstance(key, tuple):
            key = list(key)

        # Pad not specified dimensions
        if len(key) != len(self.shape):
            key += [slice(None)] * (len(self.shape) - len(key))

        # Parse each subkey. Remember location of integers for later squeeze
        key_, axis_to_squeeze = [], []
        for i, (subkey, limit) in enumerate(zip(key, self.shape)):
            if isinstance(subkey, slice):
                slc = slice(max(subkey.start or 0, 0), min(subkey.stop or limit, limit), subkey.step)
            elif isinstance(subkey, int):
                subkey = subkey if subkey >= 0 else limit - subkey
                slc = slice(subkey, subkey + 1)
                axis_to_squeeze.append(i)

            key_.append(slc)

        return key_, axis_to_squeeze


    # Coordinate system conversions
    def lines_to_ordinals(self, array):
        """ Convert values from inline-crossline coordinate system to their ordinals.
        In the simplest case of regular grid `ordinal = (value - value_min) // value_step`.
        In the case of irregular spacings between values, we have to manually map values to ordinals.
        """
        # Indexing headers
        if self.regular_structure:
            for i in range(self.index_length):
                array[:, i] -= self.shifts[i]
                if self.increments[i] != 1:
                    array[:, i] //= self.increments[i]
        else:
            raise NotImplementedError

        # Depth to units
        if array.shape[1] == self.index_length + 1:
            array[:, self.index_length + 1] -= self.delay
            array[:, self.index_length + 1] /= self.sample_rate
        return array

    def ordinals_to_lines(self, array):
        """ Convert ordinals to values in inline-crossline coordinate system.
        In the simplest case of regular grid `value = value_min + ordinal * value_step`.
        In the case of irregular spacings between values, we have to manually map ordinals to values.
        """
        # Indexing headers
        if self.regular_structure:
            for i in range(self.index_length):
                if self.increments[i] != 1:
                    array[:, i] *= self.increments[i]
                array[:, i] += self.shifts[i]
        else:
            raise NotImplementedError

        # Units to depth
        if array.shape[1] == self.index_length + 1:
            array[:, self.index_length + 1] *= self.sample_rate
            array[:, self.index_length + 1] += self.delay
        return array

    def lines_to_cdp(self, points):
        """ Convert lines to CDP. """
        return (self.rotation_matrix[:, :2] @ points.T + self.rotation_matrix[:, 2].reshape(2, -1)).T

    def cdp_to_lines(self, points):
        """ Convert CDP to lines. """
        inverse_matrix = np.linalg.inv(self.rotation_matrix[:, :2])
        lines = (inverse_matrix @ points.T - inverse_matrix @ self.rotation_matrix[:, 2].reshape(2, -1)).T
        return np.rint(lines)


    # Stats and normalization
    @property
    def quantile_interpolator(self):
        """ Quantile interpolator for arbitrary values. """
        if self._quantile_interpolator is None:
            self._quantile_interpolator = interp1d(self.quantile_support, self.quantile_values)
        return self._quantile_interpolator

    def get_quantile(self, q):
        """ Get q-th quantile of the cube data. Works with any `q` in [0, 1] range. """
        #pylint: disable=not-callable
        return self.quantile_interpolator(q).astype(np.float32)

    @property
    def normalization_stats(self):
        """ Values for performing normalization of data from the cube. """
        q_01, q_05, q_95, q_99 = self.get_quantile(q=[0.01, 0.05, 0.95, 0.99])
        normalization_stats = {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'q_01': q_01,
            'q_05': q_05,
            'q_95': q_95,
            'q_99': q_99,
        }
        return normalization_stats


    # Spatial matrices
    @property
    def snr(self):
        """ Signal-to-noise ratio. """
        return np.log(self.mean_matrix**2 / self.std_matrix**2)

    @transformable
    def get_dead_traces_matrix(self):
        """ Dead traces matrix.
        Due to decorator, allows for additional transforms at loading time.

        Parameters
        ----------
        dilate : bool
            Whether to apply dilation to the matrix.
        dilation_iterations : int
            Number of dilation iterations to apply.
        """
        return self.dead_traces_matrix.copy()

    @transformable
    def get_alive_traces_matrix(self):
        """ Alive traces matrix.
        Due to decorator, allows for additional transforms at loading time.

        Parameters
        ----------
        dilate : bool
            Whether to apply dilation to the matrix.
        dilation_iterations : int
            Number of dilation iterations to apply.
        """
        return 1 - self.dead_traces_matrix

    def get_grid(self, frequency=100, iline=True, xline=True, margin=20):
        """ Compute the grid over alive traces. """
        #pylint: disable=unexpected-keyword-arg
        # Parse parameters
        frequency = frequency if isinstance(frequency, (tuple, list)) else (frequency, frequency)

        # Prepare dilated `dead_traces_matrix`
        dead_traces_matrix = self.get_dead_traces_matrix(dilate=True, dilation_iterations=margin)

        if margin:
            dead_traces_matrix[:+margin, :] = 1
            dead_traces_matrix[-margin:, :] = 1
            dead_traces_matrix[:, :+margin] = 1
            dead_traces_matrix[:, -margin:] = 1

        # Select points to keep
        idx_i, idx_x = np.nonzero(~dead_traces_matrix)
        grid = np.zeros_like(dead_traces_matrix)
        if iline:
            mask = (idx_i % frequency[0] == 0)
            grid[idx_i[mask], idx_x[mask]] = 1
        if xline:
            mask = (idx_x % frequency[1] == 0)
            grid[idx_i[mask], idx_x[mask]] = 1
        return grid


    # Properties
    @property
    def axis_names(self):
        """ Names of the axes: indexing headers and `DEPTH` as the last one. """
        return self.index_headers + ['DEPTH']

    @property
    def bbox(self):
        """ Bounding box with geometry limits. """
        return np.array([[0, s] for s in self.shape])

    @property
    def textual(self):
        """ Wrapped textual header of SEG-Y file. """
        text = self.segy_text[0].decode('ascii')
        lines = [text[start:start + 80] for start in range(0, len(text), 80)]
        return '\n'.join(lines)

    @property
    def file_size(self):
        """ Storage size in GB. """
        return round(os.path.getsize(self.path) / (1024**3), 3)

    @property
    def nbytes(self):
        """ Size of the instance in bytes. """
        attributes = set(['headers'])
        attributes.update({attribute for attribute in self.__dict__
                           if 'matrix' in attribute or '_quality' in attribute})

        return self.cache_size + sum(sys.getsizeof(getattr(self, attribute))
                                     for attribute in attributes if hasattr(self, attribute))

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024**3)


    # Attribute retrieval. Used by `Field` instances
    def load_attribute(self, src, **kwargs):
        """ Load instance attribute from a string, e.g. `snr` or `std_matrix`.
        Used from a field to re-direct calls.
        """
        return self.get_property(src=src, **kwargs)

    @transformable
    def get_property(self, src, **_):
        """ Load a desired instance attribute. Decorated to allow additional postprocessing steps. """
        return getattr(self, src)


    # Textual representation
    def __repr__(self):
        msg = f'geometry for cube `{self.short_name}`'
        if not hasattr(self, 'shape'):
            return f'<Unprocessed {msg}>'
        return f'<Processed {msg}: {tuple(self.cube_shape)} at {hex(id(self))}>'

    def __str__(self):
        if not hasattr(self, 'shape'):
            return f'<Unprocessed geometry for cube {self.displayed_path}>'

        msg = f"""
        Processed geometry for cube    {self.path}
        Index headers:                 {self.index_headers}
        Traces:                        {self.n_traces:_}
        Shape:                         {tuple(self.shape)}
        Time delay:                    {self.delay} ms
        Sample rate:                   {self.sample_rate} ms
        Area:                          {self.area:4.1f} km²

        File size:                     {self.file_size:4.3f} GB
        Instance (memory) size:        {self.ngbytes:4.3f} GB
        """

        if self.converted and os.path.exists(self.segy_path):
            segy_size = os.path.getsize(self.segy_path) / (1024 ** 3)
            msg += f'\nSEG-Y original size:           {segy_size:4.3f} GB'

        if hasattr(self, 'dead_traces_matrix'):
            msg += f"""
        Number of dead  traces:        {self.n_dead_traces:_}
        Number of alive traces:        {self.n_alive_traces:_}
        Fullness:                      {self.n_alive_traces / self.n_traces:2.2f}
        """

        if self.has_stats:
            msg += f"""
        Value statistics:
        mean | std:                    {self.mean:>10.2f} | {self.std:<10.2f}
        min | max:                     {self.min:>10.2f} | {self.max:<10.2f}
        q01 | q99:                     {self.get_quantile(0.01):>10.2f} | {self.get_quantile(0.99):<10.2f}
        Number of unique values:       {self.n_value_uniques:>10}
        """

        if self.quantized:
            msg += f"""
        Quantization ranges:           {self.ranges[0]:>10.2f} | {self.ranges[1]:<10.2f}
        Quantization error:            {self.quantization_error:>10.3f}
        """
        return dedent(msg).strip()

    def print(self, printer=print):
        """ Show textual representation. """
        select_printer(printer)(self)

    def print_textual(self, printer=print):
        """ Show textual header from original SEG-Y. """
        select_printer(printer)(self.textual)

    def print_location(self, printer=print):
        """ Show ranges for each of the headers. """
        msg = ''
        for i, name in enumerate(self.index_headers):
            name += ':'
            msg += f'\n{name:<30} [{self.uniques[i][0]}, {self.uniques[i][-1]}]'
        select_printer(printer)(msg)

    def log(self):
        """ Log info about geometry to a file next to the cube. """
        self.print(printer=os.path.dirname(self.path) + '/CUBE_INFO.log')


    # Visual representation
    def show(self, matrix='snr', plotter=plot, **kwargs):
        """ Show geometry related top-view map. """
        matrix_name = matrix if isinstance(matrix, str) else kwargs.get('matrix_name', 'custom matrix')
        kwargs = {
            'cmap': 'magma',
            'title': f'`{matrix_name}` map of cube `{self.short_name}`',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'colorbar': True,
            **kwargs
            }
        matrix = getattr(self, matrix) if isinstance(matrix, str) else matrix
        return plotter(matrix, **kwargs)

    def show_histogram(self, n_quantile_traces=100_000, seed=42, bins=50, plotter=plot, **kwargs):
        """ Show distribution of amplitudes in a random subset of the cube. """
        # Load subset of data
        alive_traces_indices = self.index_matrix[~self.dead_traces_matrix].ravel()
        indices = np.random.default_rng(seed=seed).choice(alive_traces_indices, size=n_quantile_traces)
        data = self.load_by_indices(indices)

        kwargs = {
            'title': (f'Amplitude distribution for {self.short_name}' +
                      f'\n Mean/std: {np.mean(data):3.3f}/{np.std(data):3.3f}'),
            'label': 'Amplitudes histogram',
            'xlabel': 'amplitude',
            'ylabel': 'density',
            **kwargs
        }
        return plotter(data, bins=bins, mode='histogram', **kwargs)

    def show_slide(self, index, axis=0, zoom=None, plotter=plot, **kwargs):
        """ Show seismic slide in desired index.
        Under the hood relies on :meth:`load_slide`, so works with geometries in any formats.

        Parameters
        ----------
        index : int, str
            Index of the slide to show.
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        zoom : tuple, None or 'auto'
            Tuple of slices to apply directly to 2d images. If None, slicing is not applied.
            If 'auto', zero traces on bounds will be dropped.
        plotter : instance of `plot`
            Plotter instance to use.
            Combined with `positions` parameter allows using subplots of already existing plotter.
        """
        axis = self.parse_axis(axis)
        slide = self.load_slide(index=index, axis=axis)
        xmin, xmax, ymin, ymax = 0, slide.shape[0], slide.shape[1], 0

        if zoom == 'auto':
            zoom = self.compute_auto_zoom(index, axis)
        if zoom:
            slide = slide[zoom]
            xmin = zoom[0].start or xmin
            xmax = zoom[0].stop or xmax
            ymin = zoom[1].stop or ymin
            ymax = zoom[1].start or ymax

        # Plot params
        if len(self.index_headers) > 1:
            title = f'{self.axis_names[axis]} {index} out of {self.cube_shape[axis]}'

            if axis in [0, 1]:
                xlabel = self.index_headers[1 - axis]
                ylabel = 'DEPTH'
            else:
                xlabel = self.index_headers[0]
                ylabel = self.index_headers[1]
        else:
            title = '2D seismic slide'
            xlabel = self.index_headers[0]
            ylabel = 'DEPTH'

        kwargs = {
            'title': title,
            'suptitle':  f'Field `{self.short_name}`',
            'xlabel': xlabel,
            'ylabel': ylabel,
            'cmap': 'Greys_r',
            'colorbar': True,
            'extent': (xmin, xmax, ymin, ymax),
            'labeltop': False,
            'labelright': False,
            **kwargs
        }
        return plotter(slide, **kwargs)


    # Utilities for 2D slides
    def get_slide_index(self, index, axis=0):
        """ Get the slide index along specified axis.
        Integer `12` means 12-th (ordinal) inline.
        String `#244` means inline 244.

        Parameters
        ----------
        index : int, str
            If int, then interpreted as the ordinal along the specified axis.
            If `'random'`, then we generate random index along the axis.
            If string of the `'#XXX'` format, then we interpret it as the exact indexing header value.
        axis : int
            Axis of the slide.
        """
        if isinstance(index, (int, np.integer)):
            if index >= self.shape[axis]:
                raise KeyError(f'Index={index} is out of geometry bounds={self.shape[axis]}!')
            return index
        if index == 'random':
            return np.random.randint(0, self.lengths[axis])
        if isinstance(index, str) and index.startswith('#'):
            index = int(index[1:])
            return self.index_value_to_ordinal[axis][index]
        raise ValueError(f'Unknown type of index={index}')

    def get_slide_bounds(self, index, axis=0):
        """ Compute bounds of the slide: indices of the first/last alive traces of it.

        Parameters
        ----------
        index : int
            Ordinal index of the slide.
        axis : int
            Axis of the slide.
        """
        dead_traces = np.take(self.dead_traces_matrix, indices=index, axis=axis)
        left_bound = np.argmin(dead_traces)
        right_bound = len(dead_traces) - np.argmin(dead_traces[::-1]) # the first dead trace
        return left_bound, right_bound

    def compute_auto_zoom(self, index, axis=0):
        """ Compute zoom for a given slide. """
        return slice(*self.get_slide_bounds(index=index, axis=axis))


    # General utility methods
    STRING_TO_AXIS = {
        'i': 0, 'il': 0, 'iline': 0, 'inline': 0,
        'x': 1, 'xl': 1, 'xline': 1, 'xnline': 1,
        'd': 2, 'depth': 2,
    }

    def parse_axis(self, axis):
        """ Convert string representation of an axis into integer, if needed. """
        if isinstance(axis, str):
            if axis in self.index_headers:
                axis = self.index_headers.index(axis)
            elif axis in self.STRING_TO_AXIS:
                axis = self.STRING_TO_AXIS[axis]
        return axis

    def make_slide_locations(self, index, axis=0):
        """ Create locations (sequence of slices for each axis) for desired slide along given axis. """
        locations = [slice(0, item) for item in self.shape]

        axis = self.parse_axis(axis)
        locations[axis] = slice(index, index + 1)
        return locations

    def process_limits(self, limits):
        """ Convert given `limits` to a `slice`. """
        if limits is None:
            return slice(0, self.depth, 1)
        if isinstance(limits, (tuple, list)):
            limits = slice(*limits)
        return limits

    @staticmethod
    def locations_to_shape(locations):
        """ Compute shape of a location. """
        return tuple(slc.stop - slc.start for slc in locations)
