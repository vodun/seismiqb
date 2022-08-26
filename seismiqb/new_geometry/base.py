""" Base class for working with seismic data. """
import os

import numpy as np

from .meta_mixin import MetaMixin



class Geometry(MetaMixin):
    """ Class to infer information about seismic cube in various formats and provide format agnostic interface to them.

    During the SEG-Y processing, a number of statistics are computed. They are saved next to the cube under the
    `.segy_meta` extension, so that subsequent loads (in, possibly, other formats) don't have to recompute them.
    Most of them are loaded at initialization.
    The most memory-intensive ones are loaded on demand.

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
            - load slides takes an ordinal index of the slide and its axis;
            - load crops works off of complete location specification (triplet of slices).

        - textual representation of cube geometry: method `print` shows the summary of an instance with
        information about its location and values; `print_textual` allows to see textual header from a SEG-Y.

        - visual representation of cube geometry:
            - :meth:`show` to display top view on cube with computed statistics;
            - :meth:`show_slide` to display front view on various slices of data.

    Parameters
    ----------
    path : str
        Path to seismic cube. Supported formats are `segy`, `hdf5`, `qhdf5`, `blosc` `qblosc`.
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

    # Attributes to store in a separate file
    PRESERVED = [ # loaded at instance initialization
        # Crucial geometry properties
        'depth', 'delay', 'sample_rate', 'shape',
        'shifts', 'lengths', 'ranges', 'increments',
        'index_matrix', 'absent_traces_matrix', 'dead_traces_matrix',

        # Additional info from SEG-Y
        'segy_path', 'segy_text', 'rotation_matrix', 'area',

        # Scalar stats for amplitude values: computed for all SEG-Y / its subset
        'min', 'max', 'mean', 'std',
        'subset_min', 'subset_max', 'subset_mean', 'subset_std',
        'quantile_support', 'quantile_values',
    ]

    PRESERVED_LAZY = [ # loaded at the time of the first access
        'min_vector', 'max_vector', 'mean_vector', 'std_vector',
        'min_matrix', 'max_matrix', 'mean_matrix', 'std_matrix',
    ]


    def __init__(self, path, meta_path=None, **kwargs):
        #
        self.path = path
        self.anonymize = ...

        # Names
        self.name = os.path.basename(self.path)
        self.short_name = os.path.splitext(self.name)[0]
        self.format = os.path.splitext(self.path)[1][1:]

        # Meta
        self._meta_path = meta_path
        self.meta_list_loaded = set()
        self.meta_list_failed_to_dump = set()

        # Init from subclasses
        self.init(path, **kwargs)


    def __getattr__(self, key):
        """ Load item from stored meta. """
        if key in self.PRESERVED_LAZY and self.meta_path is not None and key not in self.__dict__:
            return self.load_meta_item(key)
        return object.__getattribute__(self, key)

    # Coordinate system conversions
    def lines_to_ordinals(self, array):
        """ Convert values from inline-crossline coordinate system to their ordinals.
        In the simplest case of regular grid `ordinal = (value - value_min) // value_step`.
        In the case of irregular spacings between values, we have to manually map values to ordinals.
        """
        # Indexing headers
        for i in range(self.index_length):
            if self.regular_structure:
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
        for i in range(self.index_length):
            if self.regular_structure:
                if self.increments[i] != 1:
                    array[:, i] *= self.increments[i]
                array[:, i] += self.shifts[i]

        # Units to depth
        if array.shape[1] == self.index_length + 1:
            array[:, self.index_length + 1] *= self.sample_rate
            array[:, self.index_length + 1] += self.delay
        return array

    # Stats and normalization
    def get_quantile(self, q):
        """ Get q-th quantile of the cube data. """
        return self.quantile_values[q]

    @property
    def normalization_stats(self):
        """ Values for performing normalization of data from the cube. """
        normalization_stats = {
            'mean': self.mean,
            'std': self.std,
            'min': self.min,
            'max': self.max,
            'q_01': self.get_quantile(q=0.01),
            'q_05': self.get_quantile(q=0.05),
            'q_95': self.get_quantile(q=0.95),
            'q_99': self.get_quantile(q=0.99),
        }
        normalization_stats = {key : float(value) for key, value in normalization_stats.items()}
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
        """ Compute bounds of the slide: the number of dead traces on the left/right side of it.

        Parameters
        ----------
        index : int
            Ordinal index of the slide.
        axis : int
            Axis of the slide.
        """
        alive_traces = 1 - np.take(self.dead_traces, indices=index, axis=axis)
        left_bound = np.argmax(alive_traces)
        right_bound = len(alive_traces) - np.argmax(alive_traces[::-1]) - 1
        return left_bound, right_bound
