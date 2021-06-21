""" Seismic facies container. """
import os
from collections import defaultdict

import numpy as np

from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns


from ..plotters import filter_parameters, plot_image
from ..horizon import Horizon, _filtering_function
from ..utils import to_list, retrieve_function_arguments, exec_callable
from ..utility_classes import lru_cache, AttachStr, HorizonSampler
from ...batchflow import Config



class Facies(Horizon):
    """ Extends basic `Horizon` functionality, allowing interaction with label subsets.

    Class methods heavily rely on the concept of nested subset storage. The underlaying idea is that label stores all
    its subsets. With this approach label subsets and their attributes can be accessed via the parent label.

    - Main methods for interaction with label subsets are `add_subset` and `get_subset`. First allows adding given label
    instance under provided name into parent subsets storage. Second returns the subset label under requested name.

    - Method for getting desired attributes is `load_attribute`. It works with nested keys, i.e. one can get attributes
    of horizon susbsets. Address method documentation for further details.

    - Method `show_label` serves to visualizing horizon and its attribute both in separate and overlap styles,
    as well as these histograms in similar manner. Address method documentation for further details.
    """

    # Correspondence between attribute alias and the class function that calculates it
    METHOD_TO_ATTRIBUTE = {
        'get_cube_values': ['cube_values', 'amplitudes'],
        'get_full_matrix': ['full_matrix', 'heights', 'depths'],
        'evaluate_metric': ['metrics'],
        'get_instantaneous_phases': ['instant_phases'],
        'get_instantaneous_amplitudes': ['instant_amplitudes'],
        'get_full_binary_matrix': ['full_binary_matrix', 'masks'],
        '_get_grid_matrix': ['grid_matrix', 'grid']
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
        # if not isinstance(item, Facies):
        #     msg = f"Only instances of `Facies` can be added as subsets, but {type(item)} was given."
        #     raise TypeError(msg)
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


    def _load_attribute(self, src, location=None, **kwargs):
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
        location : sequence of 3 slices or None
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
        x_slice, i_slice, h_slice = location if location is not None else (slice(None), slice(None), slice(None))

        default_kwargs = {'use_cache': True}
        # Update `default_kwargs` with extra arguments depending on `src`
        if src in ['cube_values', 'amplitudes']:
            if h_slice != slice(None):
                # `window` arg for `get_cube_values` can be infered from `h_slice`
                default_kwargs = {'window': h_slice.stop - h_slice.start, **default_kwargs}
        kwargs = {**default_kwargs, **kwargs}

        func_name = self.ATTRIBUTE_TO_METHOD.get(src)
        if func_name is None:
            raise ValueError("Unknown `src` {}. Expected {}.".format(src,
                                                                               self.ATTRIBUTE_TO_METHOD.keys()))
        data = getattr(self, func_name)(**kwargs)
        return data[x_slice, i_slice]


    def load_attribute(self, src, **kwargs):
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

        if src == 'amplitudes':
            kwargs['window'] = kwargs.get('window', 1)
        elif src == 'grid':
            kwargs['iterations'] = 5

        data = self._load_attribute(src=src, **kwargs)
        if subsets:
            subset = self.get_subset(subsets[0])
            location = kwargs.get('location', None)
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
        # result[self.full_matrix == self.FILL_VALUE] = np.nan
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
            E.g., if `draw='grid'`, than `Facies.draw_grid` is applied to first axis.
            If `draw=[None, 'grid']`, than `Facies.draw_grid` is applied to second axis.
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
            if attr == 'masks':
                return 'firebrick'
            return 'ocean'

        if mode == 'imshow':
            x, y = self.matrix.shape
            min_ax = min(x, y)
            defaults['figsize'] = (x / min_ax * n_subplots * 10, y / min_ax * 10)

            defaults['cmap'] = apply_by_scenario(make_cmap, names)
            defaults['xlabel'] = self.geometry.index_headers[0]
            defaults['ylabel'] = self.geometry.index_headers[1]
        elif mode == 'hist':
            defaults['figsize'] = (n_subplots * 10, 5)
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
        draw = to_list(draw, default=[])
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


    def create_sampler(self, bins=None, anomaly_grid=None, weights=None, threshold=0, **kwargs):
        """ Create sampler based on horizon location.

        Parameters
        ----------
        bins : sequence
            Size of ticks alongs each respective axis.
        quality_grid : ndarray or None
            If not None, then must be a matrix with zeroes in locations to keep, ones in locations to remove.
            Applied to `points` before sampler creation.
        weights : ndarray or bool
            Weights matrix with shape (ilines_len, xlines_len) for weights of sampling.
            If True support correlation metric will be used.
        """
        _ = kwargs
        default_bins = self.cube_shape[:2] // np.array([20, 20])
        bins = bins if bins is not None else default_bins
        anomaly_grid = self.anomaly_grid if anomaly_grid is True else anomaly_grid

        points = self.points[:, :2]
        if isinstance(anomaly_grid, np.ndarray):
            points = _filtering_function(np.copy(points), 1 - anomaly_grid)


        if weights:
            if not isinstance(weights, np.ndarray):
                corrs_matrix = self.evaluate_metric()
                weights = corrs_matrix[points[:, 0], points[:, 1]]
            points = points[~np.isnan(weights)]
            weights = weights[~np.isnan(weights)]
            points = points[weights > threshold]
            weights = weights[weights > threshold]

        sampler = HorizonSampler(np.histogramdd(points/self.cube_shape[:2], bins=bins, weights=weights), **kwargs)
        sampler = sampler.apply(AttachStr(string=self.short_name, mode='append'))
        self.sampler = sampler


    def draw_grid(self, ax, **kwargs):
        """ Draw grid on given axis using grid info. """
        try:
            info = self.grid_info
        except AttributeError as e:
            raise AttributeError("To draw a grid, one must create it via `FaciesCubeset.make_grid`.") from e
        xrange, yrange = info['range'][:2]

        default_kwargs = {
            'grid_colors': 'darkslategray',
            'grid_linestyles': 'dashed',
            'crop_colors': 'crimson',
            'crop_linewidth': 3
        }

        kwargs = {**default_kwargs, **kwargs}

        for ax_num, draw_lines in enumerate([ax.vlines, ax.hlines]):
            stride = info['strides'][ax_num]
            crop_shape = info['crop_shape'][ax_num]
            lines_max = xrange[1] if ax_num == 0 else yrange[1]
            lines_range = yrange if ax_num == 0 else xrange
            lines = np.r_[np.arange(0, lines_max, stride), [lines_max - crop_shape]]

            filtered_keys = ['colors', 'linestyles', 'linewidth']
            grid_kwargs = filter_parameters(kwargs, filtered_keys, prefix='grid_')
            draw_lines(lines, *lines_range, **grid_kwargs)

            crop_kwargs = filter_parameters(kwargs, filtered_keys, prefix='crop_')
            draw_lines(lines[0] + crop_shape, lines_range[0], crop_shape, **crop_kwargs) # draw first crop
            draw_lines(lines[-1], lines_range[1] - crop_shape, lines_range[1], **crop_kwargs) # draw last crop
