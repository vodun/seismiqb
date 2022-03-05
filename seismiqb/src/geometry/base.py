""" Base class for seismic cube geometrical and geological info. """
import os
import re
import sys
import time

from textwrap import dedent

import numpy as np
import psutil
import h5py

from .export import ExportMixin
from ..utils import CacheMixin

from ..utils import file_print, get_environ_flag, lru_cache, transformable
from ..plotters import plot_image



class SpatialDescriptor:
    """ Allows to set names for parts of information about index.
    ilines_len = SpatialDescriptor('INLINE_3D', 'lens', 'ilines_len')
    allows to get instance.lens[idx], where `idx` is position of `INLINE_3D` inside instance.index.

    Roughly equivalent to::
    @property
    def ilines_len(self):
        idx = self.index_headers.index('INLINE_3D')
        return self.lens[idx]
    """
    def __set_name__(self, owner, name):
        self.name = name

    def __init__(self, header=None, attribute=None, name=None):
        self.header = header
        self.attribute = attribute

        if name is not None:
            self.name = name

    def __get__(self, obj, obj_class=None):
        # If attribute is already stored in object, just return it
        if self.name in obj.__dict__:
            return obj.__dict__[self.name]

        # Find index of header, use it to access attr
        try:
            idx = obj.index_headers.index(self.header)
            return getattr(obj, self.attribute)[idx]
        except ValueError as exc:
            raise ValueError(f'Current index does not contain {self.header}.') from exc


def add_descriptors(cls):
    """ Add multiple descriptors to the decorated class.
    Name of each descriptor is `alias + postfix`.

    Roughly equivalent to::
    ilines = SpatialDescriptor('INLINE_3D', 'uniques', 'ilines')
    xlines = SpatialDescriptor('CROSSLINE_3D', 'uniques', 'xlines')

    ilines_len = SpatialDescriptor('INLINE_3D', 'lens', 'ilines_len')
    xlines_len = SpatialDescriptor('CROSSLINE_3D', 'lens', 'xlines_len')
    etc
    """
    attrs = ['uniques', 'offsets', 'lens']  # which attrs hold information
    postfixes = ['', '_offset', '_len']     # postfix of current attr

    headers = ['INLINE_3D', 'CROSSLINE_3D'] # headers to use
    aliases = ['ilines', 'xlines']          # alias for header

    for attr, postfix in zip(attrs, postfixes):
        for alias, header in zip(aliases, headers):
            name = alias + postfix
            descriptor = SpatialDescriptor(header=header, attribute=attr, name=name)
            setattr(cls, name, descriptor)
    return cls



@add_descriptors
class SeismicGeometry(CacheMixin, ExportMixin):
    """ Class to infer information about seismic cube in various formats, and provide API for data loading.

    During the SEG-Y processing, a number of statistics are computed. They are saved next to the cube under the
    `.meta` extension, so that subsequent loads (in, possibly, other formats) don't have to recompute them.
    Most of them (`SeismicGeometry.PRESERVED`) are loaded at initialization; yet, the most memory-intensive ones
    (`SeismicGeometry.PRESERVED_LAZY`) are loaded on demand.

    Based on the extension of the path, a different subclass is used to implement key methods for data indexing.
    Currently supported extensions:
        - `segy`
        - `hdf5` and its quantized version `qhdf5`
        - `blosc` and its quantized version `qblosc`
    The last two are created by converting the original SEG-Y cube.
    During the conversion, an extra step of `int8` quantization can be performed to reduce the disk usage.

    Independent of the exact format, `SeismicGeometry` provides the following:
        - Attributes to describe shape and structure of the cube like `cube_shape` and `lens`,
        as well as exact values of file-wide headers, for example, `delay` and `sample_rate`.

        - Ability to infer information about the cube amplitudes:
          `trace_container` attribute contains examples of amplitudes inside the cube and allows to compute statistics.

        - If needed, spatial stats can also be inferred: attributes `min_matrix`, `mean_matrix`, etc
          allow creating a complete spatial map (that is a view from above) of the desired statistic for the whole cube.
          `hist_matrix` contains a histogram of values for each trace in the cube, and can be used as
          a proxy for amplitudes in each trace for evaluating aggregated statistics.

        - :meth:`load_slide` (2D entity) or :meth:`load_crop` (3D entity) methods to load data from the cube.
          Load slides takes a number of slide and axis to cut along; makes use of `lru_cache` to work
          faster for subsequent loads. Cache is bound for each instance.
          Load crops works off of complete location specification (3D slice).

        - `quality_map` attribute is a spatial matrix that estimates cube hardness;
          `quality_grid` attribute contains a grid of locations to train model on, based on `quality_map`.

        - textual representation of cube geometry: method `print` shows the summary of an instance with
        information about its location and values; `print_textual` allows to see textual header from a SEG-Y.

        - visual representation of cube geometry: methods `show` and  `show_quality_map` display top view on
        cube with computed statistics; `show_slide` can be used for front view on various axis of data.

    Parameters
    ----------
    path : str
        Path to seismic cube. Supported formats are `segy`, `hdf5`, `qhdf5`, `blosc` `qblosc`.
    path_meta : str, optional
        Path to pre-computed statistics. If not provided, use the same as `path` with `.meta` extension.
    process : bool
        Whether to process the data: open the file and infer initial stats.
    collect_stats : bool
        If cube is in `segy` format, collect more stats about values.
    spatial : bool
        If cube is in `segy` format and `collect_stats` is True, collect stats as if the cube is POST-STACK.
    """
    #TODO: add separate class for cube-like labels
    SEGY_ALIASES = ['sgy', 'segy', 'seg']
    HDF5_ALIASES = ['hdf5', 'qhdf5']
    BLOSC_ALIASES = ['blosc', 'qblosc']
    NPZ_ALIASES = ['npz']
    ARRAY_ALIASES = ['dummyarray']

    # Attributes to store in a separate `.meta` file
    PRESERVED = [ # loaded at instance initialization
        # Crucial geometry properties
        'depth', 'delay', 'sample_rate', 'cube_shape',
        'byte_no', 'offsets', 'ranges', 'lens', # `uniques` can't be saved due to different lenghts of arrays
        'bins', 'zero_traces', '_quality_map',

        # Additional info from SEG-Y
        'segy_path', 'segy_text', 'rotation_matrix', 'area',

        # Convenient aliases for post-stack cubes
        'ilines', 'xlines', 'ilines_offset', 'xlines_offset', 'ilines_len', 'xlines_len',

        # Value-stats
        'v_uniques', 'v_min', 'v_max', 'v_mean', 'v_std',
        'v_q001', 'v_q01', 'v_q05', 'v_q95', 'v_q99', 'v_q999',

        # Parameters of quantization and quantized stats
        'qnt_ranges', 'qnt_bins', 'qnt_clip', 'qnt_center', 'qnt_error',
        'qnt_min', 'qnt_max', 'qnt_mean', 'qnt_std',
        'qnt_q001', 'qnt_q01', 'qnt_q05', 'qnt_q95', 'qnt_q99', 'qnt_q999',
    ]

    PRESERVED_LAZY = [ # loaded at the time of the first access
        'trace_container', 'hist_matrix',
        'min_matrix', 'max_matrix', 'mean_matrix', 'std_matrix',
    ]

    # Headers to load from SEG-Y cube
    HEADERS_PRE_FULL = ['FieldRecord', 'TraceNumber', 'TRACE_SEQUENCE_FILE', 'CDP', 'CDP_TRACE', 'offset', ]
    HEADERS_POST_FULL = ['INLINE_3D', 'CROSSLINE_3D', 'CDP_X', 'CDP_Y']
    HEADERS_POST = ['INLINE_3D', 'CROSSLINE_3D']

    # Headers to use as id of a trace
    INDEX_PRE = ['FieldRecord', 'TraceNumber']
    INDEX_POST = ['INLINE_3D', 'CROSSLINE_3D']
    INDEX_CDP = ['CDP_Y', 'CDP_X']

    def __new__(cls, path, *args, **kwargs):
        """ Select the type of geometry based on file extension.
        Breaks the autoreload magic (but only for this class).
        """
        #pylint: disable=import-outside-toplevel
        _ = args, kwargs
        fmt = os.path.splitext(path)[1][1:]

        if fmt in cls.SEGY_ALIASES:
            from .segy import SeismicGeometrySEGY
            new_cls = SeismicGeometrySEGY
        elif fmt in cls.HDF5_ALIASES:
            from .hdf5 import SeismicGeometryHDF5
            new_cls = SeismicGeometryHDF5
        elif fmt in cls.BLOSC_ALIASES:
            from .blosc import SeismicGeometryBLOSC
            new_cls = SeismicGeometryBLOSC
        elif fmt in cls.NPZ_ALIASES:
            from .npz import SeismicGeometryNPZ
            new_cls = SeismicGeometryNPZ
        elif fmt in cls.ARRAY_ALIASES:
            from .array import SeismicGeometryArray
            new_cls = SeismicGeometryArray
        else:
            raise TypeError(f'Unknown format of the cube: {fmt}')

        instance = super().__new__(new_cls)
        return instance

    def __getnewargs__(self):
        return (self.path, )

    def __init__(self, path, *args, process=True, path_meta=None, **kwargs):
        _ = args
        self.path = path
        self.anonymize = get_environ_flag('SEISMIQB_ANONYMIZE')

        # Names of different lengths and format: helpful for outside usage
        self.name = os.path.basename(self.path)
        self.field_name = self.extract_field_name()
        self.short_name = os.path.splitext(self.name)[0]
        self.long_name = ':'.join(self.path.split('/')[-2:])
        self.format = os.path.splitext(self.path)[1][1:]

        # Property holders
        self._quality_map = None
        self._quality_grid = None

        self.path_meta = path_meta or os.path.splitext(self.path)[0] + '.meta'
        self.loaded = []
        self.has_stats = False

        self._process = process
        self._init_kwargs = kwargs
        if process:
            self.process(**kwargs)


    def __getstate__(self):
        self.reset_cache()
        state = self.__dict__.copy()
        for name in ['file', 'axis_to_cube', 'cube_i', 'cube_x', 'cube_h']:
            if name in state:
                state.pop(name)
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            setattr(self, key, value)

        if self._process:
            self.process(**self._init_kwargs)



    def extract_field_name(self):
        """ Try to parse field from geometry name. """

        # search for a sequence of uppercase letters between '_' and '.' symbols
        field_search = re.search(r'_([A-Z]+?)\.', self.name)
        if field_search is None:
            if self.anonymize:
                msg = f"""
                Cannot anonymize name {self.name}, because field cannot be parsed from it.
                Expected name in `<attribute>_<NUM>_<FIELD>.<extension>` format.
                """
                raise ValueError(msg)
            return ""
        return self.name[slice(*field_search.span(1))]

    # Utility functions
    def parse_axis(self, axis):
        """ Convert string representation of an axis into integer, if needed. """
        if isinstance(axis, str):
            if axis in self.index_headers:
                axis = self.index_headers.index(axis)
            elif axis in ['i', 'il', 'iline']:
                axis = 0
            elif axis in ['x', 'xl', 'xline']:
                axis = 1
            elif axis in ['h', 'height', 'depth']:
                axis = 2
        return axis

    def make_slide_locations(self, loc, axis=0):
        """ Create locations (sequence of slices for each axis) for desired slide along given axis. """
        locations = [slice(0, item) for item in self.cube_shape]

        axis = self.parse_axis(axis)
        locations[axis] = slice(loc, loc + 1)
        return locations


    # Meta information: storing / retrieving attributes
    def store_meta(self, path=None):
        """ Store collected stats on disk. Uses either provided `path` or `path_meta` attribute. """
        path_meta = path or self.path_meta

        # Remove file, if exists: h5py can't do that
        if os.path.exists(path_meta):
            os.remove(path_meta)

        # Create file and datasets inside
        with h5py.File(path_meta, "a") as file_meta:
            # Save all the necessary attributes to the `info` group
            for attr in self.PRESERVED + self.PRESERVED_LAZY:
                try:
                    if hasattr(self, attr) and getattr(self, attr) is not None:
                        file_meta['/info/' + attr] = getattr(self, attr)
                except ValueError:
                    # Raised when you try to store post-stack descriptors for pre-stack cube
                    pass

    def load_meta(self):
        """ Retrieve stored stats from disk. Uses `path_meta` attribute. """
        for item in self.PRESERVED:
            value = self.load_meta_item(item)
            if value is not None:
                setattr(self, item, value)

        self.has_stats = True

    def load_meta_item(self, item):
        """ Load individual item. """
        with h5py.File(self.path_meta, "r") as file_meta:
            try:
                value = file_meta['/info/' + item][()]
                self.loaded.append(item)
                return value
            except KeyError:
                return None

    def __getattr__(self, key):
        """ Load item from stored meta. """
        if key in self.PRESERVED_LAZY and self.path_meta is not None and key not in self.__dict__:
            return self.load_meta_item(key)
        return object.__getattribute__(self, key)


    # Data loading
    def process_key(self, key):
        """ Convert multiple slices into locations. """
        key_ = list(key)
        if len(key_) != len(self.cube_shape):
            key_ += [slice(None)] * (len(self.cube_shape) - len(key_))

        key, shape, squeeze = [], [], []
        for i, item in enumerate(key_):
            max_size = self.cube_shape[i]

            if isinstance(item, slice):
                slc = slice(item.start or 0, item.stop or max_size)
            elif isinstance(item, int):
                item = item if item >= 0 else max_size - item
                slc = slice(item, item + 1)
                squeeze.append(i)
            key.append(slc)
            shape.append(slc.stop - slc.start)

        return key, shape, squeeze

    def __getitem__(self, key):
        """ Assuming that cube is POST-STACK, get sub-volume using the usual `NumPy`-like semantics.
        Can be re-implemented in child classes.
        """
        key, _, squeeze = self.process_key(key)

        crop = self.load_crop(key)
        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop

    def normalize(self, array, mode=None):
        """ Normalize array of values cut from the cube.
        Constants are computed from the entire volume.
        Constants for normalization are automatically chosen depending on the quantization of the cube.

        Parameters
        ----------
        array : ndarray
            Crop of amplitudes.
        mode : str
            If `std`, then data is divided by standard deviation.
            If `meanstd`, then data is centered and divided by standard deviation.
            If `minmax`, then data is scaled to [0, 1] via minmax scaling.
            If `q` or `normalize`, then data is divided by the
            maximum of absolute values of the 0.01 and 0.99 quantiles.
            If `q_clip`, then data is clipped to 0.01 and 0.99 quantiles and then divided
            by the maximum of absolute values of the two.
        """
        if mode is None or mode == 'auto':
            mode = 'std' if self.quantized else 'q'

        if mode == 'std':
            return array / (self.qnt_std if self.quantized else self.v_std)
        if mode == 'meanstd':
            array -= self.qnt_mean if self.quantized else self.v_mean
            return array / (self.qnt_std if self.quantized else self.v_std)

        if mode == 'q':
            return array / max(abs(self.v_q01), abs(self.v_q99))
        if mode == 'q_clip':
            array = np.clip(array, self.v_q01, self.v_q99)
            return array / max(abs(self.v_q01), abs(self.v_q99))

        if mode == 'minmax':
            min_ = self.qnt_min if self.quantized else self.v_min
            max_ = self.qnt_max if self.quantized else self.v_max
            return (array - min_) / (max_ - min_)
        raise ValueError('Wrong mode', mode)

    # Coordinates transforms
    def lines_to_cubic(self, array):
        """ Convert ilines-xlines to cubic coordinates system. """
        array[:, 0] -= self.ilines_offset
        array[:, 1] -= self.xlines_offset
        array[:, 2] -= self.delay
        array[:, 2] /= self.sample_rate
        return array

    def cubic_to_lines(self, array):
        """ Convert cubic coordinates to ilines-xlines system. """
        array = array.astype(np.float32)
        array[:, 0] += self.ilines_offset
        array[:, 1] += self.xlines_offset
        array[:, 2] *= self.sample_rate
        array[:, 2] += self.delay
        return array

    def depth_to_time(self, depthes):
        """ Convert depth to time. """
        return depthes * self.sample_rate + self.delay

    # Spatial matrices
    @property
    def snr(self):
        """ Signal-to-noise ratio. """
        return np.log(self.mean_matrix**2 / self.std_matrix**2)

    @lru_cache(100)
    def get_quantile_matrix(self, q):
        """ Restore the quantile matrix for desired `q` from `hist_matrix`.

        Parameters
        ----------
        q : number
            Quantile to compute. Must be in (0, 1) range.
        """
        #pylint: disable=line-too-long
        threshold = self.depth * q
        cumsums = np.cumsum(self.hist_matrix, axis=-1)

        positions = np.argmax(cumsums >= threshold, axis=-1)
        idx_1, idx_2 = np.nonzero(positions)
        indices = positions[idx_1, idx_2]

        broadcasted_bins = np.broadcast_to(self.bins, (*positions.shape, len(self.bins)))

        q_matrix = np.zeros_like(positions, dtype=np.float)
        q_matrix[idx_1, idx_2] += broadcasted_bins[idx_1, idx_2, indices]
        q_matrix[idx_1, idx_2] += (broadcasted_bins[idx_1, idx_2, indices+1] - broadcasted_bins[idx_1, idx_2, indices]) * \
                                   (threshold - cumsums[idx_1, idx_2, indices-1]) / self.hist_matrix[idx_1, idx_2, indices]
        q_matrix[q_matrix == 0.0] = np.nan
        return q_matrix

    @property
    def quality_map(self):
        """ Spatial matrix to show harder places in the cube. """
        if self._quality_map is None:
            self.make_quality_map([0.1], ['support_js', 'support_hellinger'])
        return self._quality_map

    def make_quality_map(self, quantiles, metric_names, **kwargs):
        """ Create `quality_map` matrix that shows harder places in the cube.

        Parameters
        ----------
        quantiles : sequence of floats
            Quantiles for computing hardness thresholds. Must be in (0, 1) ranges.
        metric_names : sequence or str
            Metrics to compute to assess hardness of cube.
        kwargs : dict
            Other parameters of metric(s) evaluation.
        """
        from ..metrics import GeometryMetrics #pylint: disable=import-outside-toplevel
        quality_map = GeometryMetrics(self).evaluate('quality_map', quantiles=quantiles,
                                                     metric_names=metric_names, **kwargs)
        self._quality_map = quality_map
        return quality_map

    @property
    def quality_grid(self):
        """ Spatial grid based on `quality_map`. """
        if self._quality_grid is None:
            self.make_quality_grid()
        return self._quality_grid

    def make_quality_grid(self, frequencies=(100, 200), iline=True, xline=True, margin=0,
                          extension='cell', filter_outliers=0, **kwargs):
        """ Create `quality_grid` based on `quality_map`.

        Parameters
        ----------
        frequencies : sequence of numbers
            Grid frequencies for individual levels of hardness in `quality_map`.
        iline, xline : bool
            Whether to make lines in grid to account for `ilines`/`xlines`.
        margin : int
            Margin of boundaries to not include in the grid.
        extension : 'full', 'cell', False or int
            Number of traces to grid lines extension.
            If 'full', then extends quality grid base points to field borders.
            If 'cell', then extends quality grid base points to sparse grid cells borders.
            If False, then make no extension.
            If int, then extends quality grid base points to +-extension//2 neighboring points.
        filter_outliers : int
            A degree of quality map thinning.
            `filter_outliers` more than zero cuts areas that contain too small connectivity regions.
            Notice that the method cut the squared area with these regions. It is made for more thinning.
        kwargs : dict
            Other parameters of grid making.
        """
        from ..metrics import GeometryMetrics #pylint: disable=import-outside-toplevel

        full_lines = kwargs.pop('full_lines', False) # for old api consistency
        extension = 'full' if full_lines else extension

        quality_grid = GeometryMetrics(self).make_grid(self.quality_map, frequencies, iline=iline, xline=xline,
                                                       margin=margin, extension=extension,
                                                       filter_outliers=filter_outliers, **kwargs)
        self._quality_grid = quality_grid
        return quality_grid


    # Properties
    @property
    def axis_names(self):
        """ Names of the axis: multiple headers and `DEPTH` as the last one. """
        return self.index_headers + ['DEPTH']

    @property
    def textual(self):
        """ Wrapped textual header of SEG-Y file. """
        txt = ''.join([chr(item) for item in self.segy_text[0]])
        txt = '\n#'.join(txt.split('C'))
        return txt.strip()

    @property
    def displayed_name(self):
        """ Return name with masked field name, if anonymization needed. """
        return self.short_name.replace(f"_{self.field_name}", "") if self.anonymize else self.short_name

    @property
    def displayed_path(self):
        """ Return path with masked field name, if anonymization needed. """
        return self.path.replace(self.field_name, '*') if self.anonymize else self.path

    @property
    def nonzero_traces(self):
        """ Amount of meaningful traces in a cube. """
        return np.prod(self.zero_traces.shape) - self.zero_traces.sum()

    @property
    def total_traces(self):
        """ Total amount of traces in a cube. """
        if hasattr(self, 'zero_traces'):
            return np.prod(self.zero_traces.shape)
        if hasattr(self, 'dataframe'):
            return len(self.dataframe)
        return self.cube_shape[0] * self.cube_shape[1]

    def __len__(self):
        """ Number of meaningful traces. """
        if hasattr(self, 'zero_traces'):
            return self.nonzero_traces
        return self.total_traces

    @property
    def shape(self):
        """ Cube 3D shape. Same API, as NumPy. """
        return tuple(self.cube_shape)

    @property
    def bbox(self):
        """ Bounding box that define field limits. """
        return np.array([[0, max] for max in self.shape])

    @property
    def spatial_shape(self):
        """ Shape of indexing axis. """
        return self.shape[:2]

    @property
    def file_size(self):
        """ Storage size in GB."""
        return round(os.path.getsize(self.path) / (1024**3), 3)

    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        names = set()
        if self.structured is False:
            names.add('dataframe')
            if self.has_stats:
                names.add('trace_container')
                names.add('zero_traces')
        else:
            for name in ['trace_container', 'zero_traces']:
                names.add(name)
        names.update({name for name in self.__dict__
                      if 'matrix' in name or '_quality' in name})

        return sum(sys.getsizeof(getattr(self, name)) for name in names if hasattr(self, name)) + self.cache_size

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024**3)


    # Attribute retrieval
    @staticmethod
    def matrix_fill_to_num(matrix, value):
        """ Change the matrix values at points where field is absent to a supplied one. """
        matrix[np.isnan(matrix)] = value
        return matrix

    @staticmethod
    def matrix_normalize(matrix, mode):
        """ Normalize matrix values.

        Parameters
        ----------
        mode : bool, str, optional
            If `min-max` or True, then use min-max scaling.
            If `mean-std`, then use mean-std scaling.
            If False, don't scale matrix.
        """
        values = matrix[~np.isnan(matrix)]

        if mode in ['min-max', True]:
            min_, max_ = np.nanmin(values), np.nanmax(values)
            matrix = (matrix - min_) / (max_ - min_)
        elif mode == 'mean-std':
            mean, std = np.nanmean(values), np.nanstd(values)
            matrix = (matrix - mean) / std
        else:
            raise ValueError(f'Unknown normalization mode `{mode}`.')
        return matrix

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
        msg = f'geometry for cube `{self.displayed_name}`'
        if not hasattr(self, 'cube_shape'):
            return f'<Unprocessed {msg}>'
        return f'<Processed {msg}: {tuple(self.cube_shape)} at {hex(id(self))}>'

    def __str__(self):
        if not hasattr(self, 'cube_shape'):
            return f'<Unprocessed geometry for cube {self.displayed_path}>'

        msg = f"""
        Processed geometry for cube    {self.displayed_path}
        Current index:                 {self.index_headers}
        Cube shape:                    {tuple(self.cube_shape)}
        Time delay:                    {self.delay}
        Sample rate:                   {self.sample_rate}
        Area:                          {self.area:4.1f} km²
        """

        if os.path.exists(self.segy_path):
            segy_size = os.path.getsize(self.segy_path) / (1024 ** 3)
            msg += f"""
        SEG-Y original size:           {segy_size:4.3f} GB
        """

        msg += f"""Current cube size:             {self.file_size:4.3f} GB
        Size of the instance:          {self.ngbytes:4.3f} GB

        Number of traces:              {self.total_traces}
        """

        if hasattr(self, 'zero_traces'):
            msg += f"""Number of non-zero traces:     {self.nonzero_traces}
        Fullness:                      {self.nonzero_traces / self.total_traces:2.2f}
        """

        if self.has_stats:
            msg += f"""
        Original cube values:
        Number of uniques:             {self.v_uniques:>10}
        mean | std:                    {self.v_mean:>10.2f} | {self.v_std:<10.2f}
        min | max:                     {self.v_min:>10.2f} | {self.v_max:<10.2f}
        q01 | q99:                     {self.v_q01:>10.2f} | {self.v_q99:<10.2f}
        """

        if self.quantized or hasattr(self, 'qnt_error'):
            msg += f"""
        Quantized cube info:
        Error of quantization:         {self.qnt_error:>10.3f}
        Ranges:                        {self.qnt_ranges[0]:>10.2f} | {self.qnt_ranges[1]:<10.2f}
        """
        return dedent(msg)

    def print(self, printer=print):
        """ Show textual representation. """
        printer(self)

    def print_textual(self, printer=print):
        """ Show textual header from original SEG-Y. """
        printer(self.textual)

    def print_location(self, printer=print):
        """ Show ranges for each of the headers. """
        msg = ''
        for i, name in enumerate(self.index_headers):
            name += ':'
            msg += f'\n{name:<30} [{self.uniques[i][0]}, {self.uniques[i][-1]}]'
        printer(msg)

    def log(self, printer=None):
        """ Log info about cube into desired stream. By default, creates a file next to the cube. """
        if not callable(printer):
            path_log = os.path.dirname(self.path) + '/CUBE_INFO.log'
            printer = lambda msg: file_print(msg, path_log)
        printer(str(self))


    # Visual representation
    def show(self, matrix='snr', **kwargs):
        """ Show geometry related top-view map. """
        matrix_name = matrix if isinstance(matrix, str) else kwargs.get('matrix_name', 'custom matrix')
        kwargs = {
            'cmap': 'viridis_r',
            'title': f'`{matrix_name}` map of cube `{self.displayed_name}`',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'colorbar': True,
            **kwargs
            }
        matrix = getattr(self, matrix) if isinstance(matrix, str) else matrix
        return plot_image(matrix, **kwargs)

    def show_histogram(self, normalize=None, bins=50, **kwargs):
        """ Show distribution of amplitudes in `trace_container`. Optionally applies chosen normalization. """
        data = np.copy(self.trace_container)
        if normalize:
            data = self.normalize(data, mode=normalize)

        kwargs = {
            'title': (f'Amplitude distribution for {self.short_name}' +
                      f'\n Mean/std: {np.mean(data):3.3}/{np.std(data):3.3}'),
            'label': 'Amplitudes histogram',
            'xlabel': 'amplitude',
            'ylabel': 'density',
            **kwargs
        }
        return plot_image(data, backend='matplotlib', bins=bins, mode='hist', **kwargs)

    def show_slide(self, loc=None, start=None, end=None, step=1, axis=0, zoom_slice=None, stable=True, **kwargs):
        """ Show seismic slide in desired place.
        Under the hood relies on :meth:`load_slide`, so works with geometries in any formats.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int or str
            Axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        start, end, step : int
            Parameters of slice loading for 1D index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        axis = self.parse_axis(axis)
        slide = self.load_slide(loc=loc, start=start, end=end, step=step, axis=axis, stable=stable)
        xmin, xmax, ymin, ymax = 0, slide.shape[0], slide.shape[1], 0

        if zoom_slice:
            slide = slide[zoom_slice]
            xmin = zoom_slice[0].start or xmin
            xmax = zoom_slice[0].stop or xmax
            ymin = zoom_slice[1].stop or ymin
            ymax = zoom_slice[1].start or ymax

        # Plot params
        if len(self.index_headers) > 1:
            title = f'{self.axis_names[axis]} {loc} out of {self.cube_shape[axis]}'

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
            'xlabel': xlabel,
            'ylabel': ylabel,
            'cmap': 'gray',
            'extent': (xmin, xmax, ymin, ymax),
            'labeltop': False,
            'labelright': False,
            **kwargs
        }
        return plot_image(slide, **kwargs)

    def show_quality_map(self, **kwargs):
        """ Show quality map. """
        self.show(matrix=self.quality_map, cmap='Reds', title=f'Quality map of `{self.displayed_name}`')

    def show_quality_grid(self, **kwargs):
        """ Show quality grid. """
        self.show(matrix=self.quality_grid, cmap='Reds', interpolation='bilinear',
                  title=f'Quality grid of `{self.displayed_name}`')


    # Coordinate conversion
    def lines_to_cdp(self, points):
        """ Convert lines to CDP. """
        return (self.rotation_matrix[:, :2] @ points.T + self.rotation_matrix[:, 2].reshape(2, -1)).T

    def cdp_to_lines(self, points):
        """ Convert CDP to lines. """
        inverse_matrix = np.linalg.inv(self.rotation_matrix[:, :2])
        lines = (inverse_matrix @ points.T - inverse_matrix @ self.rotation_matrix[:, 2].reshape(2, -1)).T
        return np.rint(lines)

    def benchmark(self, n_slide=300, projections='ixh', n_crop=300, crop_shapes_min=5, crop_shapes_max=200,
                  use_cache=False, seed=42):
        """ Calculate average data loading timings (in ms) in user, system and wall mode for slides and crops.

        Time measurement idea (sys, user and wall) is from python magic function `time` realization.

        Parameters
        ----------
        n_slide : int, optional
            Amount of slides to load in an experiment.
        projections : str or sequence of int or str, optional
            Preferable axes to load slide along: `i` or 0 for iline one, `x` or 1 for the crossline, `h` or 2 for depth.
        n_crop : int, optional
            Amount of crops to load in an experiment.
        crop_shapes_min : int or tuple of int, optional
            A minimum size of crop.
            If int, then it applied for the first index, the second, and depth.
            If tuple, then each number correspond to each crop index.
        crop_shapes_max : int or tuple of int, optional
            A maximum size of crop.
            If int, then it applied for the first index, the second, and depth.
            If tuple, then each number correspond to each crop index.
        use_cache : bool, optional
            Whether to use lru_cache for :meth:`load_slide` and :meth:`load_crop`.
        seed : int, optional
            Seed the random numbers generator.
        """
        rng = np.random.default_rng(seed)
        timings = {}

        # Parse projections:
        projections = [self.parse_axis(proj) for proj in projections]

        # Calculate the average loading slide time by loading random slides `n_slide` times
        self.reset_cache()

        wall_st = time.perf_counter()
        start = psutil.cpu_times() # 0 key - user time in seconds,  2 key - system time in seconds

        for _ in range(n_slide):
            axis = rng.choice(a=projections)
            loc = rng.integers(low=0, high=self.cube_shape[axis])
            _ = self.load_slide(loc=loc, axis=axis, use_cache=use_cache)

        end = psutil.cpu_times()
        wall_end = time.perf_counter()

        timings['slide'] = {
            'user': 1000 * (end[0] - start[0]) / n_slide,
            'system': 1000 * (end[2] - start[2]) / n_slide,
            'wall': 1000 * (wall_end - wall_st) / n_slide
        }

        # Preparation for timing loading crop calculation
        self.reset_cache()

        if isinstance(crop_shapes_min, int):
            crop_shapes_min = (crop_shapes_min, crop_shapes_min, crop_shapes_min)
        if isinstance(crop_shapes_max, int):
            crop_shapes_max = (crop_shapes_max, crop_shapes_max, crop_shapes_max)

        # The `load_crop` method by default doesn't use cache. It uses cache in the `slide` mode.
        crop_kwargs = {'mode': 'slide'} if use_cache else {}

        # Calculate the average loading crop time by loading random crops `n_crop` times
        wall_st = time.perf_counter()
        start = psutil.cpu_times() # 0 key - user time in seconds,  2 key - system time in seconds

        for _ in range(n_crop):
            point = rng.integers(low=(0, 0, 0), high=self.cube_shape) // 2
            shape = rng.integers(low=crop_shapes_min, high=crop_shapes_max)
            locations = [slice(start_, np.clip(start_+shape_, 0, max_shape))
                        for start_, shape_, max_shape in zip(point, shape, self.cube_shape)]
            _ = self.load_crop(locations, **crop_kwargs)

        end = psutil.cpu_times()
        wall_end = time.perf_counter()

        timings['crop'] = {
            'user': 1000 * (end[0] - start[0]) / n_slide,
            'system': 1000 * (end[2] - start[2]) / n_slide,
            'wall': 1000 * (wall_end - wall_st) / n_slide
        }

        self.reset_cache()
        return timings
