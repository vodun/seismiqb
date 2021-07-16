""" Seismic facies container. """
import os

import numpy as np
import pandas as pd

from scipy.signal import hilbert, ricker
from scipy.ndimage import convolve
from sklearn.decomposition import PCA


from ..plotters import filter_parameters, plot_image
from ..horizon import Horizon
from ..utils import to_list, retrieve_function_arguments
from ..utility_classes import lru_cache



class Facies(Horizon):
    """ Extends basic `Horizon` functionality, allowing interaction with label subsets.

    Class methods heavily rely on the concept of nested subset storage. The underlaying idea is that label stores all
    its subsets. With this approach label subsets and their attributes can be accessed via the parent label.

    - Main methods for interaction with label subsets are `add_subset` and `get_subset`. First allows adding given label
    instance under provided name into parent subsets storage. Second returns the subset label under requested name.

    - Method for getting desired attributes is `load_attribute`. It works with nested keys, i.e. one can get attributes
    of horizon susbsets. Address method documentation for further details.

    - Method `show` visualizes horizon and its attributes in both separate and overlap manners.
    """

    # Correspondence between attribute alias and the class function that calculates it
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


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampler = None
        self.subsets = {}


    def add_subset(self, name, item):
        """ Add item to subsets storage.

        Parameters
        ----------
        name : str
            Key to store given horizon under.
        item : Facies
            Instance to store.
        """
        self.subsets[name] = item


    def get_subset(self, name):
        """ Get item from subsets storage.

        Parameters
        ----------
        name : str
            Key desired item is stored under.
        """
        try:
            return self.subsets[name]
        except KeyError as e:
            msg = f"Requested subset {name} is missing in subsets storage. Availiable subsets are {list(self.subsets)}."
            raise KeyError(msg) from e


    def _load_attribute(self, src, location=None, use_cache=True, **kwargs):
        """ Make crops from `src` of horizon at `location`.

        Parameters
        ----------
        src : str
            A keyword defining horizon attribute to make crops from:
            - 'cube_values' or 'amplitudes': cube values cut along the horizon;
            - 'depths' or 'full_matrix': horizon depth map in cubic coordinates;
            - 'metrics': random support metrics matrix.
            - 'instant_phases': instantaneous phase along the horizon;
            - 'instant_amplitudes': instantaneous amplitude along the horizon;
            - 'masks' or 'full_binary_matrix': mask of horizon;
        location : sequence of 3 slices
            First two slices are used as `iline` and `xline` ranges to cut crop from.
            Last 'depth' slice is used to infer `window` parameter when `src` is 'cube_values'.
            If None, `src` is returned uncropped.
        kwargs :
            Passed directly to either:
            - one of attribute-evaluating methods from :attr:`.ATTRIBUTE_TO_METHOD` depending on `src`;
            - or attribute-transforming method :meth:`.transform_where_present`.
        Examples
        --------

        >>> horizon.load_attribute('cube_values', (x_slice, i_slice, h_slice), window=10)

        >>> horizon.load_attribute('depths')

        >>> horizon.load_attribute('metrics', metrics='hilbert', normalize='min-max')

        Notes
        -----

        Although the function can be used in a straightforward way as described above, its main purpose is to act
        as an interface for accessing :class:`.Horizon` attributes from :class:`~SeismicCropBatch` to allow calls like:

        >>> Pipeline().load_attribute('cube_values', dst='amplitudes')
        """
        try:
            method_name = self.ATTRIBUTE_TO_METHOD[src]
        except KeyError as e:
            raise ValueError(f"Unknown `src` {src}. Expected one of {list(self.ATTRIBUTE_TO_METHOD.keys())}.") from e

        method = getattr(self, method_name)
        data = method(use_cache=use_cache, **kwargs)
        if location is not None:
            i_slice, x_slice, _ = location
            data = data[i_slice, x_slice]

        return data


    def load_attribute(self, src, location=None, **kwargs):
        """ Get attribute for horizon or its subset.

        Parameters
        ----------
        src : str
            Key of the desired attribute. If attribute is from horizon subset, key must be like "subset/attribute".
        kwargs : for `Horizon.load_attribute`

        Examples
        --------
        To load "depths" attribute from "channels" subset of `horizon` one should do the following:
        >>> horizon.load_attribute(src="channels/depths")
        """
        *subsets, src = src.split('/')
        if len(subsets) > 1:
            raise NotImplementedError()

        data = self._load_attribute(src=src, location=location, **kwargs)

        if subsets:
            subset = self.get_subset(subsets[0])
            mask = subset.load_attribute(src='masks', location=location, fill_value=0).astype(bool)
            data[~mask] = kwargs.get('fill_value', self.FILL_VALUE)

        return data


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

    def show(self, load='depths', mode='imshow', draw=None, return_figure=False, **kwargs):
        """ Display facies attributes with predefined defaults.

        Loads requested data, constructs default parameters wrt to that data and delegates plot to `plot_image`.

        Parameters
        ----------
        load : str or dict or np.ndarray, or a list of objects of those types
            Defines data to display.
            If str, a name of attribute to load. Address `Facies.METHOD_TO_ATTRIBUTE` values for details.
            If dict, arguments for `Facies.load_attribute` and optional callable param under 'postprocess` key.
            If np.ndarray, must be 2d and match facies full matrix shape.
            If list, defines several data object to display. For details about nestedness address `plot_image` docs.
        mode : 'imshow' or 'hist'
            Mode to display images in.
        return_figure : bool
            Whether return resulted figure or not.
        draw : str, None or list of objects of those types
            Aliases for actions applied to resulting figure axes.
            E.g., if `draw='crop_grid'`, than `Facies.draw_crop_grid` is applied to first axis.
            If `draw=[None, 'crop_grid']`, than `Facies.draw_crop_grid` is applied to second axis.
        kwargs : for `plot_image`

        Examples
        --------
        Display depth attribute:
        >>> facies.show()
        Display several attributes one over another:
        >>> facies.show(load=['amplitudes', 'channels/masks'])
        Display several attributes separately:
        >>> facies.show(load=['amplitudes', 'instant_amplitudes'], separate=True)
        Display several attributes in mixed manner:
        >>> facies.show(load=['amplitudes', ['amplitudes', 'channels/masks']])
        Display attribute with additional postprocessing:
        >>> facies.show(load={'src': 'amplitudes', 'fill_value': 0, 'normalize': 'min-max'})

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
        def load_data(load):
            """ Manage data loading depending on load params type. """
            if isinstance(load, np.ndarray):
                return load
            if isinstance(load, str):
                load = {'src': load}
            postprocess = load.pop('postprocess', lambda x: x)
            load_defaults = {'fill_value': np.nan}
            if load['src'].split('/')[-1] in ['amplitudes', 'cube_values']:
                load_defaults['width'] = 1
            if load['src'].split('/')[-1] in ['fourier', 'wavelet']:
                load_defaults['n_components'] = 1
            load = {**load_defaults, **load}
            data = self.load_attribute(**load)
            return postprocess(data)

        data = apply_by_scenario(load_data, load)

        # Make titles
        def extract_data_name(load):
            if isinstance(load, np.ndarray):
                name = 'custom'
            elif isinstance(load, dict):
                name = load['src']
            elif isinstance(load, str):
                name = load
            return name

        names = apply_by_scenario(extract_data_name, load)
        n_subplots = len(data) if isinstance(data, list) else 1

        def make_titles(names):
            if any(isinstance(item, list) for item in load):
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
        def make_cmap(name):
            attr = name.split('/')[-1]
            if attr == 'depths':
                return 'Depths'
            if attr == 'metrics':
                return 'Metric'
            if attr == 'masks':
                return 'firebrick'
            return 'ocean'

        def make_alpha(name):
            return 0.7 if name.split('/')[-1] == 'masks' else 1.0

        if mode == 'imshow':
            x, y = self.matrix.shape
            defaults = {
                **defaults,
                'figsize': (x / min(x, y) * n_subplots * 10, y / min(x, y) * 10),
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

        # Display additional info over axes via `Facies` methods starting with 'draw_' prefix
        draw = to_list(draw) if draw is not None else []
        for ax_draw, ax in zip(draw, figure.axes):
            if ax_draw is not None:
                getattr(self, f"draw_{ax_draw}")(ax, **kwargs)

        return figure if return_figure else None


    def __sub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Operands types do not match. Got {type(self)} and {type(other)}.")
        presence = other.presence_matrix
        discrepancies = self.full_matrix[presence] != other.full_matrix[presence]
        if discrepancies.any():
            raise ValueError("Horizons have different depths where present.")
        result = self.full_matrix.copy()
        result[presence] = self.FILL_VALUE
        name = f"~{other.name}"
        return type(self)(result, self.geometry, name)


    def invert_subset(self, subset):
        """ Subtract subset matrix from facies matrix. """
        return self - self.get_subset(subset)


    def dump(self, path, name=None):
        """ Save facies. """
        path = path.replace('*', self.geometry.short_name)
        os.makedirs(path, exist_ok=True)
        file_path = f"{path}/{name or self.name}"
        super().dump(file_path)


    def reset_cache(self):
        """ Clear cached data. """
        super().reset_cache()
        for subset_label in self.subsets.values():
            subset_label.reset_cache()


    def draw_crop_grid(self, ax, **kwargs):
        """ Draw crops grid on given axis using grid info. """
        try:
            info = self.grid_info
        except AttributeError as e:
            raise AttributeError("To draw a grid, one must create it via `FaciesCubeset.make_grid`.") from e
        xrange, yrange = info['range'][:2]

        default_kwargs = {
            'crop_grid_colors': 'darkslategray',
            'crop_grid_linestyles': 'dashed',
            'crop_corner_colors': 'crimson',
            'crop_corner_linewidth': 3
        }

        kwargs = {**default_kwargs, **kwargs}

        for ax_num, draw_lines in enumerate([ax.vlines, ax.hlines]):
            stride = info['strides'][ax_num]
            crop_shape = info['crop_shape'][ax_num]
            lines_max = xrange[1] if ax_num == 0 else yrange[1]
            lines_range = yrange if ax_num == 0 else xrange
            lines = np.r_[np.arange(0, lines_max, stride), [lines_max - crop_shape]]

            filtered_keys = ['colors', 'linestyles', 'linewidth']
            grid_kwargs = filter_parameters(kwargs, filtered_keys, prefix='crop_grid_')
            draw_lines(lines, *lines_range, **grid_kwargs)

            crop_kwargs = filter_parameters(kwargs, filtered_keys, prefix='crop_corner_')
            draw_lines(lines[0] + crop_shape, lines_range[0], crop_shape, **crop_kwargs) # draw first crop
            draw_lines(lines[-1], lines_range[1] - crop_shape, lines_range[1], **crop_kwargs) # draw last crop

    def evaluate(self, src_true, src_pred, metrics_fn, metrics_name='metrics', **kwargs):
        """ Apply given function to 'masks' attribute of requested labels subsets.

        Parameters
        ----------
        src_true : str
            Name of `labels` subset to load true mask from.
        src_pred : str
            Name of `labels` subset to load prediction mask from.
        metrics_fn : callable
            Metrics function to calculate.
        metrics_name : str
            Name of the column with metrics values in resulted dataframe.
        kwargs :
            For `metrics_fn`.
        """
        pd.options.display.float_format = '{:,.3f}'.format

        labeled_traces = self.get_full_binary_matrix(fill_value=0).astype(bool)
        true = self.load_attribute(f"{src_true}/masks", fill_value=0)[labeled_traces]
        pred = self.load_attribute(f"{src_pred}/masks", fill_value=0)[labeled_traces]
        value = metrics_fn(true, pred, **kwargs)

        index = pd.MultiIndex.from_arrays([[self.geometry.displayed_name], [self.short_name]],
                                          names=['geometry_name', 'horizon_name'])
        result = pd.DataFrame(index=index, data=[value], columns=[metrics_name])
        return result
