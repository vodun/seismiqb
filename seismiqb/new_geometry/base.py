""" Base class for working with seismic data. """
import os

import numpy as np
from scipy.interpolate import interp1d

from .benchmark_mixin import BenchmarkMixin
from .conversion_mixin import ConversionMixin
from .export_mixin import ExportMixin
from .meta_mixin import MetaMixin



class Geometry(BenchmarkMixin, ConversionMixin, ExportMixin, MetaMixin):
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
        'min', 'max', 'mean', 'std',
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
        """ !!. """
        return tuple(slc.stop - slc.start for slc in locations)
