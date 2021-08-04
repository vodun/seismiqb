""" Mixin with computed along horizon geological attributes. """
# pylint: disable=too-many-statements
import numpy as np
from matplotlib import pyplot as plt

from scipy.signal import hilbert, ricker
from scipy.ndimage import convolve
from scipy.ndimage.morphology import binary_fill_holes, binary_erosion
from skimage.measure import label
from sklearn.decomposition import PCA

from ..functional import smooth_out
from ..utils import to_list, retrieve_function_arguments, lru_cache
from ..plotters import plot_image



class AttributesMixin:
    """ Geological attributes along horizon:
    - geometrical statistics like number of holes, perimeter, coverage
    - straight-forward maps along horizons, computed directly from its matrix such as `binary`, `borders`, `grad` etc
    - methods to cut data from the cube along horizon
    - maps, derived from amplitudes along horizon.

    Method for getting desired attributes is `load_attribute`. It works with nested keys, i.e. one can get attributes
    of horizon subsets. Address method documentation for further details.


    - Method `show` visualizes horizon and its attributes in both separate and overlap manners. It allows visual overlap
    of various attributes with one or more labels masks.
    """
    METHOD_TO_ATTRIBUTE = {
        'get_cube_values': ['cube_values', 'amplitudes'],
        'get_full_matrix': ['full_matrix', 'heights', 'depths'],
        'evaluate_metric': ['metrics'],
        'get_instantaneous_phases': ['instant_phases'],
        'get_instantaneous_amplitudes': ['instant_amplitudes'],
        'get_full_binary_matrix': ['full_binary_matrix', 'masks'],
        'fourier_decompose': ['fourier', 'fourier_decompose'],
        'wavelet_decompose': ['wavelet', 'wavelet_decompose']
    }
    ATTRIBUTE_TO_METHOD = {attr: func for func, attrs in METHOD_TO_ATTRIBUTE.items() for attr in attrs}

    # Geometrical and geological properties
    @property
    def amplitudes(self):
        """ Alias for cube values. Depending on what loaded to cube geometries
        might actually not be amplitudes, so use it with caution.
        """
        return self.cube_values

    @property
    def binary_matrix(self):
        """ Matrix with ones at places where horizon is present and zeros everywhere else. """
        return (self.matrix > 0).astype(bool)

    @property
    def borders_matrix(self):
        """ Borders of horizons (borders of holes inside are not included). """
        filled_matrix = self.filled_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(filled_matrix, structure, border_value=0)
        return filled_matrix ^ eroded # binary difference operation

    @property
    def boundaries_matrix(self):
        """ Borders of horizons (borders of holes inside included). """
        binary_matrix = self.binary_matrix
        structure = np.ones((3, 3))
        eroded = binary_erosion(binary_matrix, structure, border_value=0)
        return binary_matrix ^ eroded # binary difference operation

    @property
    def coverage(self):
        """ Ratio between number of present values and number of good traces in cube. """
        return len(self) / (np.prod(self.cube_shape[:2]) - np.sum(self.geometry.zero_traces))

    @property
    def cube_values(self):
        """ Values from the cube along the horizon. """
        cube_values = self.get_cube_values(window=1)
        cube_values[self.full_matrix == self.FILL_VALUE] = np.nan
        return cube_values

    @property
    def filled_matrix(self):
        """ Binary matrix with filled holes. """
        structure = np.ones((3, 3))
        filled_matrix = binary_fill_holes(self.binary_matrix, structure)
        return filled_matrix

    @property
    def full_matrix(self):
        """ Matrix in cubic coordinate system. """
        return self.get_full_matrix()

    @property
    def presence_matrix(self):
        """ Binary matrix in cubic coordinate system. """
        return self.put_on_full(self.binary_matrix, fill_value=False, dtype=bool)

    @property
    def grad_i(self):
        """ Change of heights along iline direction. """
        return self.grad_along_axis(0)

    @property
    def grad_x(self):
        """ Change of heights along xline direction. """
        return self.grad_along_axis(1)

    @property
    def hash(self):
        """ Hash on current data of the horizon. """
        return hash(self.matrix.data.tobytes())

    @property
    def horizon_metrics(self):
        """ Calculate :class:`~HorizonMetrics` on demand. """
        # pylint: disable=import-outside-toplevel
        from ..metrics import HorizonMetrics
        return HorizonMetrics(self)

    @property
    def instantaneous_phase(self):
        """ Phase along the horizon. """
        return self.horizon_metrics.evaluate('instantaneous_phase')

    @property
    def number_of_holes(self):
        """ Number of holes inside horizon borders. """
        holes_array = self.filled_matrix != self.binary_matrix
        _, num = label(holes_array, connectivity=2, return_num=True, background=0)
        return num

    @property
    def perimeter(self):
        """ Number of points in the borders. """
        return np.sum((self.borders_matrix == 1).astype(np.int32))

    @property
    def solidity(self):
        """ Ratio of area covered by horizon to total area inside borders. """
        return len(self) / np.sum(self.filled_matrix)


    # Carcass properties: should be used only if the horizon is a carcass
    @property
    def is_carcass(self):
        """ Check if the horizon is a sparse carcass. """
        return len(self) / self.filled_matrix.sum() < 0.5

    @property
    def carcass_ilines(self):
        """ Labeled inlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 0], return_counts=True)
        return uniques[counts > 256]

    @property
    def carcass_xlines(self):
        """ Labeled xlines in a carcass. """
        uniques, counts = np.unique(self.points[:, 1], return_counts=True)
        return uniques[counts > 256]

    @property
    def carcass_grid(self):
        """ Full matrix with present lines. """
        return self.put_on_full(self.binary_matrix, fill_value=0.0)


    # Retrieve data from seismic along horizon
    def transform_where_present(self, array, normalize=None, fill_value=None, shift=None, rescale=None, res_ndim=None):
        """ Normalize array where horizon is present, fill with constant where the horizon is absent.

        Parameters
        ----------
        array : np.array
            Matrix of (cube_ilines, cube_xlines, ...) shape.
        normalize : 'min-max', 'mean-std', 'shift-rescale' or None/False
            Normalization mode for data where `presence_matrix` is True.
            If None, no normalization applied. Defaults to None.
        fill_value : number
            Value to fill `array` in where :attr:`.presence_matrix` is False. Must be compatible with `array.dtype`.
            If None, no filling applied. Defaults to None.
        shift, rescale : number, optional
            For 'shift-rescale` normalization mode.
        res_ndim : int or None
            Number of dimensions returned result should have.
        """
        if not normalize:
            pass
        elif normalize == 'min-max':
            values = array[self.presence_matrix]
            min_, max_ = values.min(), values.max()
            array = (array - min_) / (max_ - min_)
        elif normalize == 'mean-std':
            values = array[self.presence_matrix]
            mean, std = values.mean(), values.std()
            array = (array - mean) / std
        elif normalize == 'shift-rescale':
            array = (array + shift) * rescale
        else:
            raise ValueError('Unknown normalize mode `{}`'.format(normalize))

        if fill_value is not None:
            array[~self.presence_matrix] = fill_value

        if res_ndim == array.ndim + 1:
            array = array[..., np.newaxis]
        elif res_ndim is not None and res_ndim != array.ndim:
            msg = f"Result ndim is {array.ndim}, while requested ndim is {res_ndim}. "\
                  f"Adding more than one new axis is not currently implemented."
            raise ValueError(msg)

        return array


    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def get_cube_values(self, window=23, offset=0, chunk_size=256, **kwargs):
        """ Get values from the cube along the horizon.

        Parameters
        ----------
        window : int
            Width of data slice along the horizon.
        offset : int
            Offset of data slice with respect to horizon heights matrix.
        chunk_size : int
            Size of data along height axis processed at a time.
        kwargs :
            Passed directly to :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        low = window // 2
        high = max(window - low, 0)
        chunk_size = min(chunk_size, self.h_max - self.h_min + window)

        background = np.zeros((self.geometry.ilines_len, self.geometry.xlines_len, window), dtype=np.float32)

        for h_start in range(max(low, self.h_min), self.h_max + 1, chunk_size):
            h_end = min(h_start + chunk_size, self.h_max + 1)

            # Get chunk from the cube (depth-wise)
            location = (slice(None), slice(None),
                        slice(h_start - low, min(h_end + high, self.geometry.depth)))
            data_chunk = self.geometry.load_crop(location, use_cache=False)

            # Check which points of the horizon are in the current chunk (and present)
            idx_i, idx_x = np.asarray((self.matrix != self.FILL_VALUE) &
                                      (self.matrix >= h_start) &
                                      (self.matrix < h_end)).nonzero()
            heights = self.matrix[idx_i, idx_x]

            # Convert spatial coordinates to cubic, convert height to current chunk local system
            idx_i += self.i_min
            idx_x += self.x_min
            heights -= (h_start - offset)

            # Subsequently add values from the cube to background, then shift horizon 1 unit lower
            for j in range(window):
                background[idx_i, idx_x, np.full_like(heights, j)] = data_chunk[idx_i, idx_x, heights]
                heights += 1
                mask = heights < data_chunk.shape[2]
                idx_i = idx_i[mask]
                idx_x = idx_x[mask]
                heights = heights[mask]

        background[self.geometry.zero_traces == 1] = np.nan
        return self.transform_where_present(background, **transform_kwargs)


    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def get_full_matrix(self, **kwargs):
        """ Transform `matrix` attribute to match cubic coordinates.

        Parameters
        ----------
        kwargs :
            Passed directly to :meth:`.put_on_full` and :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        matrix = self.put_on_full(self.matrix, **kwargs)
        return self.transform_where_present(matrix, **transform_kwargs)


    def get_array_values(self, array, shifts=None, grid_info=None, width=5, axes=(2, 1, 0)):
        """ Get values from an external array along the horizon.

        Parameters
        ----------
        array : np.ndarray
            A data-array to make a cut from.
        shifts : tuple or None
            an offset defining the location of given array with respect to the horizon.
            If None, `grid_info` with key `range` must be supplied.
        grid_info : dict
            Whenever passed, must contain key `range`.
            Used for infering shifts of the array with respect to horizon.
        width : int
            required width of the resulting cut.
        axes : tuple
            if not None, axes-transposition with the required axes-order is used.
        """
        if shifts is None and grid_info is None:
            raise ValueError('Either shifts or dataset with filled grid_info must be supplied!')

        if shifts is None:
            shifts = [grid_info['range'][i][0] for i in range(3)]

        shifts = np.array(shifts)
        horizon_shift = np.array((self.bbox[0, 0], self.bbox[1, 0]))

        if axes is not None:
            array = np.transpose(array, axes=axes)

        # compute start and end-points of the ilines-xlines overlap between
        # array and matrix in horizon and array-coordinates
        horizon_shift, shifts = np.array(horizon_shift), np.array(shifts)
        horizon_max = horizon_shift[:2] + np.array(self.matrix.shape)
        array_max = np.array(array.shape[:2]) + shifts[:2]
        overlap_shape = np.minimum(horizon_max[:2], array_max[:2]) - np.maximum(horizon_shift[:2], shifts[:2])
        overlap_start = np.maximum(0, horizon_shift[:2] - shifts[:2])
        heights_start = np.maximum(shifts[:2] - horizon_shift[:2], 0)

        # recompute horizon-matrix in array-coordinates
        slc_array = [slice(l, h) for l, h in zip(overlap_start, overlap_start + overlap_shape)]
        slc_horizon = [slice(l, h) for l, h in zip(heights_start, heights_start + overlap_shape)]
        overlap_matrix = np.full(array.shape[:2], fill_value=self.FILL_VALUE, dtype=np.float32)
        overlap_matrix[slc_array] = self.matrix[slc_horizon]
        overlap_matrix -= shifts[-1]

        # make the cut-array and fill it with array-data located on needed heights
        result = np.full(array.shape[:2] + (width, ), np.nan, dtype=np.float32)
        iterator = [overlap_matrix + shift for shift in range(-width // 2 + 1, width // 2 + 1)]

        for i, surface_level in enumerate(np.array(iterator)):
            mask = (surface_level >= 0) & (surface_level < array.shape[-1]) & (surface_level !=
                                                                               self.FILL_VALUE - shifts[-1])
            mask_where = np.where(mask)
            result[mask_where[0], mask_where[1], i] = array[mask_where[0], mask_where[1],
                                                            surface_level[mask_where].astype(np.int)]

        return result


    # Modify things
    def grad_along_axis(self, axis=0):
        """ Change of heights along specified direction. """
        grad = np.diff(self.matrix, axis=axis, prepend=0)
        grad[np.abs(grad) > 10000] = self.FILL_VALUE
        grad[self.matrix == self.FILL_VALUE] = self.FILL_VALUE
        return grad

    def make_float_matrix(self, kernel=None, kernel_size=7, sigma=2., margin=5, iters=1):
        """ Smooth the depth matrix to produce floating point numbers. """
        float_matrix = smooth_out(self.full_matrix, kernel=kernel,
                                  kernel_size=kernel_size, sigma=sigma, margin=margin,
                                  fill_value=self.FILL_VALUE, preserve=True, iters=iters)
        return float_matrix

    def put_on_full(self, matrix=None, fill_value=None, dtype=np.float32):
        """ Create a matrix in cubic coordinate system. """
        matrix = matrix if matrix is not None else self.matrix
        fill_value = fill_value if fill_value is not None else self.FILL_VALUE

        background = np.full(self.cube_shape[:-1], fill_value, dtype=dtype)
        background[self.i_min:self.i_max+1, self.x_min:self.x_max+1] = matrix
        return background


    # Generic attributes loading
    def load_attribute(self, src, location=None, **kwargs):
        """ Load horizon or its subset attribute values at requested location.

        Parameters
        ----------
        src : str
            Key of the desired attribute.
            If attribute is from horizon subset, key must be like "subset/attribute".

            Valid attributes are:
            - 'cube_values' or 'amplitudes': cube values;
            - 'depths' or 'full_matrix': horizon depth map in cubic coordinates;
            - 'metrics': random support metrics matrix.
            - 'instant_phases': instantaneous phase;
            - 'instant_amplitudes': instantaneous amplitude;
            - 'fourier' or 'fourier_decomposition': fourier transform with optional PCA;
            - 'wavelet' or 'wavelet decomposition': wavelet transform with optional PCA;
            - 'masks' or 'full_binary_matrix': mask of horizon;
        location : sequence of 3 slices
            First two slices are used as `iline` and `xline` ranges to cut crop from.
            Last 'depth' slice is not used, since points are sampled exactly on horizon.
            If None, `src` is returned uncropped.
        kwargs :
            Passed directly to either:
            - one of attribute-evaluating methods from :attr:`.ATTRIBUTE_TO_METHOD` depending on `src`;
            - or attribute-transforming method :meth:`.transform_where_present`.

        Examples
        --------
        Load 'depths' attribute for whole horizon:
        >>> horizon.load_attribute('depths')

        Load 'cube_values' attribute for requested slice of fixed width:
        >>> horizon.load_attribute('cube_values', (x_slice, i_slice, 1), window=10)

        Load 'metrics' attribute with specific evaluation parameter and following normalization.
        >>> horizon.load_attribute('metrics', metrics='hilbert', normalize='min-max')

        Load "wavelet" attribute from "channels" subset of `horizon`:
        >>> horizon.load_attribute(src="channels/wavelet")
        """
        if '/' in src:
            subset_name, src = src.split('/')
        else:
            subset_name = None

        data = self._load_attribute(src=src, location=location, **kwargs)

        if subset_name:
            subset = self.get_subset(subset_name)
            # pylint: disable=protected-access
            mask = subset._load_attribute(src='masks', location=location, fill_value=0).astype(bool)
            data[~mask] = kwargs.get('fill_value', self.FILL_VALUE)

        return data

    def _load_attribute(self, src, location=None, use_cache=True, **kwargs):
        """ Load horizon attribute at requested location. """
        try:
            method_name = self.ATTRIBUTE_TO_METHOD[src]
        except KeyError as e:
            raise ValueError(f"Unknown `src` {src}. Expected one of {list(self.ATTRIBUTE_TO_METHOD.keys())}.") from e

        method = getattr(self, method_name)
        data = method(use_cache=use_cache, **kwargs)
        # TODO: Someday, we would need to re-write attribute loading methods
        # so they use locations not to crop the loaded result, but to load attribute only at location.
        if location is not None:
            i_slice, x_slice, _ = location
            data = data[i_slice, x_slice]

        return data


    # Specific attributes loading
    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def get_full_binary_matrix(self, **kwargs):
        """ Transform `binary_matrix` attribute to match cubic coordinates.

        Parameters
        ----------
        kwargs :
            Passed directly to :meth:`.put_on_full` and :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        full_binary_matrix = self.put_on_full(self.binary_matrix, **kwargs)
        return self.transform_where_present(full_binary_matrix, **transform_kwargs)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def get_instantaneous_amplitudes(self, window=23, depths=None, **kwargs):
        """ Calculate instantaneous amplitude along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        depths : slice, sequence of int or None
            Which depth channels of resulted array to return.
            If slice or sequence of int, used for slicing calculated attribute along last axis.
            If None, infer middle channel index from 'window' and slice at it calculated attribute along last axis.
        kwargs :
            Passed directly to :meth:`.get_cube_values` and :meth:`.transform_where_present``.

        Notes
        -----
        Keep in mind, that Hilbert transform produces artifacts at signal start and end. Therefore if you want to get
        an attribute with `N` channels along depth axis, you should provide `window` broader then `N`. E.g. in call
        `label.get_instantaneous_amplitudes(depths=range(10, 21), window=41)` the attribute will be first calculated
        by array of `(xlines, ilines, 41)` shape and then the slice `[..., ..., 10:21]` of them will be returned.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        depths = [window // 2] if depths is None else depths
        amplitudes = self.get_cube_values(window, use_cache=False, **kwargs) #pylint: disable=unexpected-keyword-arg
        result = np.abs(hilbert(amplitudes))[:, :, depths]
        return self.transform_where_present(result, **transform_kwargs)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def get_instantaneous_phases(self, window=23, depths=None, **kwargs):
        """ Calculate instantaneous phase along the horizon.

        Parameters
        ----------
        window : int
            Width of cube values cutout along horizon to use for attribute calculation.
        depths : slice, sequence of int or None
            Which depth channels of resulted array to return.
            If slice or sequence of int, used for slicing calculated attribute along last axis.
            If None, infer middle channel index from 'window' and slice at it calculated attribute along last axis.
        kwargs :
            Passed directly to :meth:`.get_cube_values` and :meth:`.transform_where_present`.

        Notes
        -----
        Keep in mind, that Hilbert transform produces artifacts at signal start and end. Therefore if you want to get
        an attribute with `N` channels along depth axis, you should provide `window` broader then `N`. E.g. in call
        `label.get_instantaneous_phases(depths=range(10, 21), window=41)` the attribute will be first calculated
        by array of `(xlines, ilines, 41)` shape and then the slice `[..., ..., 10:21]` of them will be returned.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        depths = [window // 2] if depths is None else depths
        amplitudes = self.get_cube_values(window, use_cache=False, **kwargs) #pylint: disable=unexpected-keyword-arg
        result = np.angle(hilbert(amplitudes))[:, :, depths]
        # result[self.full_matrix == self.FILL_VALUE] = np.nan
        return self.transform_where_present(result, **transform_kwargs)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def evaluate_metric(self, metric='support_corrs', supports=50, agg='nanmean', **kwargs):
        """ Cached metrics calcucaltion with disabled plotting option.

        Parameters
        ----------
        metric, supports, agg :
            Passed directly to :meth:`.HorizonMetrics.evaluate`.
        kwargs :
            Passed directly to :meth:`.HorizonMetrics.evaluate` and :meth:`.transform_where_present`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)
        metrics = self.horizon_metrics.evaluate(metric=metric, supports=supports, agg=agg,
                                                plot=False, savepath=None, **kwargs)
        metrics = np.nan_to_num(metrics)
        return self.transform_where_present(metrics, **transform_kwargs)

    @staticmethod
    def reduce_dimensionality(data, n_components=3, **kwargs):
        """ Reduce number of channels along the depth axis. """
        flattened = data.reshape(-1, data.shape[-1])
        flattened[np.isnan(flattened).any(axis=-1)] = 0
        pca = PCA(n_components, **kwargs)
        reduced = pca.fit_transform(flattened)
        return reduced.reshape(*data.shape[:2], -1)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def fourier_decompose(self, window=50, n_components=None, **kwargs):
        """ Cached fourier transform calculation follower by dimensionaluty reduction via PCA.

        Parameters
        ----------
        window : int
            Width of amplitudes slice to calculate fourier transform on.
        n_components : int or None
            Number of components to keep after PCA.
            If None, do not perform dimensionality reduction.
        kwargs :
            For `sklearn.decomposition.PCA`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)

        amplitudes = self.load_attribute('amplitudes', window=window)
        result = np.abs(np.fft.rfft(amplitudes))

        if n_components is not None:
            result = self.reduce_dimensionality(result, n_components, **kwargs)

        return self.transform_where_present(result, **transform_kwargs)

    @lru_cache(maxsize=1, apply_by_default=False, copy_on_return=True)
    def wavelet_decompose(self, widths=range(1, 14, 3), window=50, n_components=None, **kwargs):
        """ Cached wavelet transform calculation follower by dimensionaluty reduction via PCA.

        Parameters
        ----------
        widths : list of numbers
            Widths of wavelets to calculate decomposition for.
        window : int
            Width of amplitudes slice to calculate wavelet transform on.
        n_components : int
            Number of components to keep after PCA.
            If None, do not perform dimensionality reduction.
        kwargs :
            For `sklearn.decomposition.PCA`.
        """
        transform_kwargs = retrieve_function_arguments(self.transform_where_present, kwargs)

        amplitudes = self.load_attribute('amplitudes', window=window)
        result_shape = *amplitudes.shape[:2], len(widths)
        result = np.empty(result_shape, dtype=np.float32)
        for idx, width in enumerate(widths):
            wavelet = ricker(window, width).reshape(1, 1, -1)
            result[:, :, idx] = convolve(amplitudes, wavelet, mode='constant')[:, :, window // 2]

        if n_components is not None:
            result = self.reduce_dimensionality(result, n_components, **kwargs)

        return self.transform_where_present(result, **transform_kwargs)


    # Attributes visualization
    def show(self, attributes='depths', mode='imshow', return_figure=False, enlarge=True, width=9, **kwargs):
        """ Display facies attributes with predefined defaults.

        Loads requested data, constructs default parameters wrt to that data and delegates plot to `plot_image`.

        Parameters
        ----------
        attributes : str or dict or np.ndarray, or a list of objects of those types
            Defines attributes to display.
            If str, a name of attribute to load. Address `Facies.METHOD_TO_ATTRIBUTE` values for details.
            If dict, arguments for `Facies.load_attribute` and optional callable param under 'postprocess` key.
            If np.ndarray, must be 2d and match facies full matrix shape.
            If list, defines several data object to display. For details about nestedness address `plot_image` docs.
        mode : 'imshow' or 'hist'
            Mode to display images in.
        return_figure : bool
            Whether return resulted figure or not.
        kwargs : for `plot_image`

        Examples
        --------
        Display depth attribute:
        >>> facies.show()
        Display several attributes one over another:
        >>> facies.show(['amplitudes', 'channels/masks'])
        Display several attributes separately:
        >>> facies.show(['amplitudes', 'instant_amplitudes'], separate=True)
        Display several attributes in mixed manner:
        >>> facies.show(['amplitudes', ['amplitudes', 'channels/masks']])
        Display attribute with additional postprocessing:
        >>> facies.show({'src': 'amplitudes', 'fill_value': 0, 'normalize': 'min-max'})

        Notes
        -----
        Asterisks in title-like and 'savepath' parameters are replaced by label displayed name.
        """
        def apply_by_scenario(action, params):
            """ Generic method that applies given action to params depending on their type. """
            if not isinstance(params, list):
                res = action(params)
            elif all(not isinstance(item, list) for item in params):
                res = [action(subplot_params) for subplot_params in params]
            else:
                res = []
                for subplot_params in params:
                    subplot_res = [action(layer_params) for layer_params in to_list(subplot_params)]
                    res.append(subplot_res)
            return res

        # Load attributes and put obtained data in a list with same nestedness as `load`
        def load_data(attributes):
            """ Manage data loading depending on load params type. """
            if isinstance(attributes, np.ndarray):
                return attributes
            if isinstance(attributes, str):
                load = {'src': attributes}
            if isinstance(attributes, dict):
                load = attributes
            postprocess = load.pop('postprocess', lambda x: x)
            load_defaults = {'fill_value': np.nan}
            if load['src'].split('/')[-1] in ['amplitudes', 'cube_values']:
                load_defaults['window'] = 1
            if load['src'].split('/')[-1] in ['fourier', 'wavelet']:
                load_defaults['n_components'] = 1
            if load['src'].split('/')[-1] in ['masks', 'full_binary_matrix']:
                load_defaults['fill_value'] = 0
            load = {**load_defaults, **load}
            data = self.load_attribute(**load)
            return postprocess(data)

        def enlarge_data(data):
            if self.is_carcass and enlarge:
                data = self.enlarge_carcass_image(data, width)
            return data

        data = apply_by_scenario(load_data, attributes)
        data = apply_by_scenario(enlarge_data, data)

        # Make titles
        def extract_data_name(attributes):
            if isinstance(attributes, np.ndarray):
                name = 'custom'
            elif isinstance(attributes, dict):
                name = attributes['src']
            elif isinstance(attributes, str):
                name = attributes
            return name

        names = apply_by_scenario(extract_data_name, attributes)
        n_subplots = len(data) if isinstance(data, list) else 1

        def make_titles(names):
            if any(isinstance(item, list) for item in attributes):
                return [', '.join(subplot_names) for subplot_names in names]
            return names

        defaults = {
            'title_label': make_titles(names),
            'suptitle_label': f"`{self.short_name}` of cube `{self.geometry.displayed_name}`",
            'colorbar': mode == 'imshow',
            'tight_layout': True,
            'return_figure': True,
        }

        # Infer defaults for `mode`: generate cmaps according to requested data, set axis labels as index headers
        default_colors = ['firebrick', 'darkorchid', 'sandybrown']
        gen_color = (color for color in default_colors)
        name_to_color = {}
        def make_cmap(name):
            attr = name.split('/')[-1]
            if attr == 'depths':
                return 'Depths'
            if attr == 'metrics':
                return 'Metric'
            if attr == 'masks':
                if name not in name_to_color:
                    name_to_color[name] = next(gen_color)
                return name_to_color[name]
            return 'ocean'

        def make_alpha(name):
            return 0.7 if name.split('/')[-1] == 'masks' else 1.0

        if mode == 'imshow':
            x, y = self.matrix.shape
            defaults = {
                **defaults,
                'figsize': (x / min(x, y) * n_subplots * 7, y / min(x, y) * 7),
                'xlim': self.bbox[0],
                'ylim': self.bbox[1][::-1],
                'cmap': apply_by_scenario(make_cmap, names),
                'alpha': apply_by_scenario(make_alpha, names),
                'xlabel': self.geometry.index_headers[0],
                'ylabel': self.geometry.index_headers[1],
            }
        elif mode == 'hist':
            defaults = {**defaults, 'figsize': (n_subplots * 10, 5)}
        else:
            raise ValueError(f"Valid modes are 'imshow' or 'hist', but '{mode}' was given.")

        # Merge default and given params
        params = {**defaults, **kwargs}
        # Substitute asterisks with label name
        for text in ['suptitle_label', 'suptitle', 'title_label', 'title', 't', 'savepath']:
            if text in params:
                params[text] = apply_by_scenario(lambda s: s.replace('*', defaults['suptitle_label']), params[text])
        # Plot image with given params and return resulting figure
        figure = plot_image(data=data, mode=mode, **params)
        plt.show()

        return figure if return_figure else None
