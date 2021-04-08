""" SeismicGeometry-class containing geometrical info about seismic-cube."""
import os
import re
import sys
import shutil
import itertools

from textwrap import dedent
from tqdm.auto import tqdm

import numpy as np
import h5py
import segyio

from ..utils import file_print, compute_attribute, make_axis_grid, \
                    fill_defaults, get_environ_flag
from ..utility_classes import lru_cache
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
class SeismicGeometry:
    """ This class selects which type of geometry to initialize: the SEG-Y or the HDF5 one,
    depending on the passed path.

    Independent of exact format, `SeismicGeometry` provides following:
        - Attributes to describe shape and structure of the cube like `cube_shape` and `lens`,
        as well as exact values of file-wide headers, for example, `time_delay` and `sample_rate`.

        - Ability to infer information about the cube amplitudes:
          `trace_container` attribute contains examples of amplitudes inside the cube and allows to compute statistics.

        - If needed, spatial stats can also be inferred: attributes `min_matrix`, `mean_matrix`, etc
          allow to create a complete spatial map (that is view from above) of the desired statistic for the whole cube.
          `hist_matrix` contains a histogram of values for each trace in the cube, and can be used as
          a proxy for amplitudes in each trace for evaluating aggregated statistics.

        - `load_slide` (2D entity) or `load_crop` (3D entity) methods to load data from the cube.
          Load slides takes a number of slide and axis to cut along; makes use of `lru_cache` to work
          faster for subsequent loads. Cache is bound for each instance.
          Load crops works off of complete location specification (3D slice).

        - `quality_map` attribute is a spatial matrix that estimates cube hardness;
          `quality_grid` attribute contains a grid of locations to train model on, based on `quality_map`.

        - `show_slide` method allows to do exactly what the name says, and has the same API as `load_slide`.
          `repr` allows to get a quick summary of the cube statistics.

    Refer to the documentation of respective classes to learn more about their structure, attributes and methods.
    """
    #TODO: add separate class for cube-like labels
    SEGY_ALIASES = ['sgy', 'segy', 'seg']
    HDF5_ALIASES = ['hdf5', 'h5py']
    BLOSC_ALIASES = ['blosc']
    NPZ_ALIASES = ['npz']

    # Attributes to store during SEG-Y -> HDF5 conversion
    PRESERVED = [
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

    PRESERVED_LAZY = [
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
        """ Select the type of geometry based on file extension. """
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
        else:
            raise TypeError('Unknown format of the cube.')

        instance = super().__new__(new_cls)
        return instance

    def __init__(self, path, *args, process=True, path_meta=None, **kwargs):
        _ = args
        self.path = path
        self.anonymize = get_environ_flag('SEISMIQB_ANONYMIZE')

        name = os.path.basename(self.path)
        # Find span of uppercase letter sequence between '_' and '.' symbols in filename
        field_search = re.search(r'_([A-Z]+?)\.', name)
        self.field = name[slice(*field_search.span(1))] if field_search is not None else ""
        self.name = name.replace("_" * bool(self.field) + self.field, "") if self.anonymize else name
        if not self.field and self.anonymize:
            raise ValueError("Geometry name was not anonymized, since field name cannot be parsed from it.")

        # Names of different lengths and format: helpful for outside usage
        self.short_name = self.name.split('.')[0]
        self.long_name = ':'.join(self.path.split('/')[-2:])
        self.format = os.path.splitext(self.path)[1][1:]

        # Property holders
        self._quality_map = None
        self._quality_grid = None

        self.path_meta = path_meta or os.path.splitext(self.path)[0] + '.meta'
        self.loaded = []
        self.has_stats = False
        if process:
            self.process(**kwargs)


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
        """ Create locations (sequence of locations for each axis) for desired slide along desired axis. """
        axis = self.parse_axis(axis)

        locations = [slice(0, item) for item in self.cube_shape]
        locations[axis] = slice(loc, loc + 1)
        return locations


    # Meta information: storing / retrieving attributes
    def store_meta(self, path=None):
        """ Store collected stats on disk. """
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
        """ Retrieve stored stats from disk. """
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
        """ Load item from stored meta, if needed. """
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
        """ Get sub-cube be slices. Can be re-implemented in child classes. """
        key, _, squeeze = self.process_key(key)

        crop = self.load_crop(key)
        if squeeze:
            crop = np.squeeze(crop, axis=tuple(squeeze))
        return crop

    def normalize(self, array, mode=None):
        """ Normalize array of amplitudes cut from the cube.
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
            self.make_quality_grid((100, 200))
        return self._quality_grid

    def make_quality_grid(self, frequencies, iline=True, xline=True, full_lines=True, margin=0, **kwargs):
        """ Create `quality_grid` based on `quality_map`.

        Parameters
        ----------
        frequencies : sequence of numbers
            Grid frequencies for individual levels of hardness in `quality_map`.
        iline, xline : bool
            Whether to make lines in grid to account for `ilines`/`xlines`.
        full_lines : bool
            Whether to make lines on the whole spatial range.
        margin : int
            Margin of boundaries to not include in the grid.
        kwargs : dict
            Other parameters of grid making.
        """
        from ..metrics import GeometryMetrics #pylint: disable=import-outside-toplevel
        quality_grid = GeometryMetrics(self).make_grid(self.quality_map, frequencies,
                                                       iline=iline, xline=xline, full_lines=full_lines,
                                                       margin=margin, **kwargs)
        self._quality_grid = quality_grid
        return quality_grid


    # Cache: introspection and reset
    def reset_cache(self):
        """ Clear cached slides. """
        if self.structured is False:
            method = self.load_slide
        else:
            method = self._cached_load
        method.reset(instance=self)

    @property
    def cache_length(self):
        """ Total amount of cached slides. """
        if self.structured is False:
            method = self.load_slide
        else:
            method = self._cached_load

        return len(method.cache()[self])

    @property
    def cache_size(self):
        """ Total size of cached slides. """
        if self.structured is False:
            method = self.load_slide
        else:
            method = self._cached_load

        return sum(item.nbytes / (1024 ** 3) for item in method.cache()[self].values())


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
        return txt.replace('26 \n#', '26 C').strip()

    @property
    def displayed_path(self):
        """ Return path with masked field name, if anonymization needed. """
        return self.path.replace(self.field, "*" * bool(self.field)) if self.anonymize else self.path

    @property
    def nonzero_traces(self):
        """ Amount of meaningful traces in a cube. """
        return np.prod(self.zero_traces.shape) - self.zero_traces.sum()

    @property
    def total_traces(self):
        """ Total amount of traces in a cube. """
        if hasattr(self, 'zero_traces'):
            return np.prod(self.zero_traces.shape)
        elif hasattr(self, 'dataframe'):
            return len(self.dataframe)
        return self.cube_shape[0] * self.cube_shape[1]

    def __len__(self):
        """ Number of meaningful traces. """
        if hasattr(self, 'zero_traces'):
            return self.nonzero_traces
        return self.total_traces

    @property
    def nbytes(self):
        """ Size of instance in bytes. """
        attrs = [
            'dataframe', 'trace_container', 'zero_traces',
            *[attr for attr in self.__dict__
              if 'matrix' in attr or '_quality' in attr],
        ]
        return sum(sys.getsizeof(getattr(self, attr)) for attr in attrs if hasattr(self, attr)) + self.cache_size

    @property
    def ngbytes(self):
        """ Size of instance in gigabytes. """
        return self.nbytes / (1024**3)


    # Textual representation
    def __repr__(self):
        return f'<Inferred geometry for {self.name}: {tuple(self.cube_shape)}>'

    def __str__(self):
        msg = f"""
        Geometry for cube              {self.displayed_path}
        Current index:                 {self.index_headers}
        Cube shape:                    {self.cube_shape}
        Time delay:                    {self.delay}
        Sample rate:                   {self.sample_rate}

        Cube size:                     {os.path.getsize(self.path) / (1024**3):4.3f} GB
        Size of the instance:          {self.ngbytes:4.3f} GB

        Number of traces:              {self.total_traces}
        """

        if hasattr(self, 'zero_traces'):
            msg += f"""Number of non-zero traces:     {self.nonzero_traces}
            """

        if self.has_stats:
            msg += f"""
        Original cube values:
        Number of uniques:             {self.v_uniques:>10}
        mean | std:                    {self.v_mean:>10.2f} | {self.v_std:<10.2f}
        min | max:                     {self.v_min:>10.2f} | {self.v_max:<10.2f}
        q01 | q99:                     {self.v_q01:>10.2f} | {self.v_q99:<10.2f}
        """

        if self.quantized:
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
        """ Show geometry-related map. """
        kwargs = {
            'cmap': 'viridis_r',
            'title': f'{matrix if isinstance(matrix, str) else ""} map of `{self.name}`',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            **kwargs
            }
        matrix = getattr(self, matrix) if isinstance(matrix, str) else matrix
        plot_image(matrix, mode='single', **kwargs)

    def show_histogram(self, scaler=None, bins=50, **kwargs):
        """ Show distribution of amplitudes in `trace_container`. Optionally applies chosen `scaler`. """
        data = np.copy(self.trace_container)
        if scaler:
            data = self.scaler(data, mode=scaler)

        kwargs = {
            'title': (f'Amplitude distribution for {self.short_name}' +
                      f'\n Mean/std: {np.mean(data):3.3}/{np.std(data):3.3}'),
            'label': 'Amplitudes histogram',
            'xlabel': 'amplitude',
            'ylabel': 'density',
            **kwargs
        }
        plot_image(data, backend='matplotlib', bins=bins, mode='histogram', **kwargs)

    def show_slide(self, loc=None, start=None, end=None, step=1, axis=0, zoom_slice=None,
                   n_ticks=5, delta_ticks=100, stable=True, **kwargs):
        """ Show seismic slide in desired place. Works with both SEG-Y and HDF5 files.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        start, end, step : int
            Parameters of slice loading for 1D index.
        stable : bool
            Whether or not to use the same sorting order as in the segyfile.
        """
        axis = self.parse_axis(axis)
        slide = self.load_slide(loc=loc, start=start, end=end, step=step, axis=axis, stable=stable)
        xticks = list(range(slide.shape[0]))
        yticks = list(range(slide.shape[1]))

        if zoom_slice:
            slide = slide[zoom_slice]
            xticks = xticks[zoom_slice[0]]
            yticks = yticks[zoom_slice[1]]

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

        xticks = xticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [xticks[-1]]
        xticks = sorted(list(set(xticks)))
        yticks = yticks[::max(1, round(len(xticks) // (n_ticks - 1) / delta_ticks)) * delta_ticks] + [yticks[-1]]
        yticks = sorted(list(set(yticks)), reverse=True)

        if len(xticks) > 2 and (xticks[-1] - xticks[-2]) < delta_ticks:
            xticks.pop(-2)
        if len(yticks) > 2 and (yticks[0] - yticks[1]) < delta_ticks:
            yticks.pop(1)

        kwargs = {
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'cmap': 'gray',
            'xticks': xticks,
            'yticks': yticks,
            'labeltop': False,
            'labelright': False,
            **kwargs
        }
        plot_image(slide, **kwargs)

    def show_quality_map(self, **kwargs):
        """ Show quality map. """
        self.show(matrix=self.quality_map, cmap='Reds', title=f'Quality map of `{self.name}`')

    def show_quality_grid(self, **kwargs):
        """ Show quality grid. """
        self.show(matrix=self.quality_grid, cmap='Reds', interpolation='bilinear',
                  title=f'Quality map of `{self.name}`')


    # Coordinate conversion
    def lines_to_cdp(self, points):
        """ Convert lines to CDP. """
        return (self.rotation_matrix[:, :2] @ points.T + self.rotation_matrix[:, 2].reshape(2, -1)).T

    def cdp_to_lines(self, points):
        """ Convert CDP to lines. """
        inverse_matrix = np.linalg.inv(self.rotation_matrix[:, :2])
        lines = (inverse_matrix @ points.T - inverse_matrix @ self.rotation_matrix[:, 2].reshape(2, -1)).T
        return np.rint(lines)


    # Attributes
    def compute_attribute(self, locations=None, window=10, attribute='semblance', device='cpu'):
        """ Compute attribute on cube.

        Parameters
        ----------
        locations : tuple of slices
            slices for each axis of cube to compute attribute. If locations is None,
            attribute will be computed for the whole cube.
        points : np.ndarray
            points where compute the attribute. In other points attribute will be equal to numpy.nan.
        window : int or tuple of ints
            window for the filter.
        stride : int or tuple of ints
            stride to compute attribute
        attribute : str
            name of the attribute

        Returns
        -------
        np.ndarray
            array of the shape corresponding to locations
        """
        if locations is None:
            locations = [slice(0, self.cube_shape[i]) for i in range(3)]
        data = self.file_hdf5['cube'][locations]

        return compute_attribute(data, window, device, attribute)

    def create_attribute_hdf5(self, attr, dst, chunk_shape=None, chunk_stride=None, window=10,
                              agg=None, projections='ixh', pbar=False, device='cpu'):
        """ Create hdf5 file from np.ndarray or with geological attribute.

        Parameters
        ----------
        path_hdf5 : str

        src : np.ndarray, iterable or str
            If `str`, must be a name of the attribute to compute.
            If 'iterable, items must be tuples (coord of chunk, chunk).
        chunk_shape : int, tuple or None
            Shape of chunks.
        chunk_stride : int
            Stride for chunks.
        pbar : bool
            Progress bar.
        """
        shape = self.cube_shape

        chunk_shape = fill_defaults(chunk_shape, shape)
        chunk_stride = fill_defaults(chunk_stride, chunk_shape)

        grid = [make_axis_grid((0, shape[i]), chunk_stride[i], shape[i], chunk_shape[i] ) for i in range(3)]

        def _iterator():
            for coord in itertools.product(*grid):
                locations = [slice(coord[i], coord[i] + chunk_shape[i]) for i in range(3)]
                yield coord, self.compute_attribute(locations, window, attribute=attr, device=device)
        chunks = _iterator()
        total = np.prod([len(item) for item in grid])
        chunks = tqdm(chunks, total=total) if pbar else chunks
        return self.create_file_from_iterable(chunks, self.cube_shape, chunk_shape,
                                                     chunk_stride, dst=dst, agg=agg, projection='ixh')

    # Misc
    @classmethod
    def create_file_from_iterable(cls, src, shape, window, stride, dst=None,
                                  agg=None, projection='ixh', threshold=None):
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
            Path to the resulting .hdf5. If None, function will return array with predictions
        agg : 'mean', 'min' or 'max' or None, optional
            The way to aggregate values in overlapped regions. None means that new chunk will rewrite
            previous value in cube.
        projection : str, optional
            Projections to create in hdf5 file, by default 'ixh'
        threshold : float or None, optional
            If not None, threshold to transform values into [0, 1], by default None
        """
        shape = np.array(shape)
        window = np.array(window)
        stride = np.array(stride)

        if dst is None:
            dst = np.zeros(shape)
        else:
            file_hdf5 = h5py.File(dst, 'a')
            dst = file_hdf5.create_dataset('cube', shape)
            cube_hdf5_x = file_hdf5.create_dataset('cube_x', shape[[1, 2, 0]])
            cube_hdf5_h = file_hdf5.create_dataset('cube_h', shape[[2, 0, 1]])

        lower_bounds = [make_axis_grid((0, shape[i]), stride[i], shape[i], window[i]) for i in range(3)]
        lower_bounds = np.stack(np.meshgrid(*lower_bounds), axis=-1).reshape(-1, 3)
        upper_bounds = lower_bounds + window
        grid = np.stack([lower_bounds, upper_bounds], axis=-1)

        for position, chunk in src:
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
                chunk /= agg_map
                chunk = _chunk + chunk
            dst[slices] = chunk
        if isinstance(dst, np.ndarray):
            if threshold is not None:
                dst = (dst > threshold).astype(int)
        else:
            for i in range(0, dst.shape[0], window[0]):
                slide = dst[i:i+window[0]]
                if threshold is not None:
                    slide = (slide > threshold).astype(int)
                    dst[i:i+window[0]] = slide
                cube_hdf5_x[:, :, i:i+window[0]] = slide.transpose((1, 2, 0))
                cube_hdf5_h[:, i:i+window[0]] = slide.transpose((2, 0, 1))
        return dst


    def make_sgy(self, path_hdf5=None, path_spec=None, postfix='',
                 remove_hdf5=False, zip_result=True, path_segy=None, pbar=False):
        """ Convert POST-STACK HDF5 cube to SEG-Y format with current geometry spec.

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
            cube_hdf5 = src['cube']

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
