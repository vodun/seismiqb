""" Seismic facies container. """
import os
from collections import defaultdict

import numpy as np

from scipy.signal import hilbert
import matplotlib.pyplot as plt
import seaborn as sns


from ..plotters import filter_kwargs, plot_image
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


    def _load_attribute(self, src_attribute, location=None, **kwargs):
        """ Make crops from `src_attribute` of horizon at `location`.

        Parameters
        ----------
        src_attribute : str
            A keyword defining horizon attribute to make crops from:
            - 'cube_values' or 'amplitudes': cube values cut along the horizon;
            - 'depths' or 'full_matrix': horizon depth map in cubic coordinates;
            - 'metrics': random support metrics matrix.
            - 'instant_phases': instantaneous phase along the horizon;
            - 'instant_amplitudes': instantaneous amplitude along the horizon;
            - 'masks' or 'full_binary_matrix': mask of horizon;
        location : sequence of 3 slices or None
            First two slices are used as `iline` and `xline` ranges to cut crop from.
            Last 'depth' slice is used to infer `window` parameter when `src_attribute` is 'cube_values'.
            If None, `src_attribute` is returned uncropped.
        kwargs :
            Passed directly to either:
            - one of attribute-evaluating methods from :attr:`.ATTRIBUTE_TO_METHOD` depending on `src_attribute`;
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
        # Update `default_kwargs` with extra arguments depending on `src_attribute`
        if src_attribute in ['cube_values', 'amplitudes']:
            if h_slice != slice(None):
                # `window` arg for `get_cube_values` can be infered from `h_slice`
                default_kwargs = {'window': h_slice.stop - h_slice.start, **default_kwargs}
        kwargs = {**default_kwargs, **kwargs}

        func_name = self.ATTRIBUTE_TO_METHOD.get(src_attribute)
        if func_name is None:
            raise ValueError("Unknown `src_attribute` {}. Expected {}.".format(src_attribute,
                                                                               self.ATTRIBUTE_TO_METHOD.keys()))
        data = getattr(self, func_name)(**kwargs)
        return data[x_slice, i_slice]


    def load_attribute(self, src_attribute, **kwargs):
        """ Get attribute for horizon or its subset.

        Parameters
        ----------
        src_attribute : str
            Key of the desired attribute. If attribute is from horizon subset, key must be like "subset/attribute".
        kwargs : for `Horizon.load_attribute`

        Examples
        --------
        To load "depths" attribute from "channels" subset of `horizon` one should do the following:
        >>> horizon.load_attribute(src_attribute="channels/depths")
        """
        *subsets, src_attribute = src_attribute.split('/')

        if src_attribute == 'amplitudes':
            kwargs['window'] = kwargs.get('window', 1)
        elif src_attribute == 'grid':
            kwargs['iterations'] = 5

        data = self._load_attribute(src_attribute=src_attribute, **kwargs)
        if subsets:
            subset = self.get_subset(subsets[0])
            src_mask = '/'.join(subsets + ['masks'])
            location = kwargs.get('location', None)
            mask = subset.load_attribute(src_attribute=src_mask, location=location, fill_value=0).astype(bool)
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


    def show(self, attributes=None, linkage=None, show=True, save=False, return_figure=False, **figure_params):
        """ Show attributes or their histograms of horizon and its subsets.

        Parameters
        ----------
        attributes : str or list of str or list of lists of str
            Attributes names to visualize. If of str type, single attribute will be shown. If list of str, several
            attributes will be shown on separate subplots. If list of lists of str, attributes from each inner list
            will be shown overlapped in corresponding subplots. See `Usage` section 1 for examples.
            Note, that this argument can't be used with `linkage`.
        linkage : list of lists of dicts
            Contains subplot parameters:
            >>> linkage = [subplot_0, subplot_1, …, subplot_n]
            Each of which in turn is a list that contains layers parameters:
            >>> subplot = [layer_0, layer_1, …, layer_m]
            Each of which in turn is a dict, that has mandatory 'load' and optional 'show' keys:
            >>> layer = {'load': load_parameters, 'show': show_parameters}
            Load parameters are either a dict of arguments meant for `load_attribute` or `np.ndarray` to show.
            Show parameters are from the following:
            TODO
            See `Usage` section 2 for examples.
        show : bool
            Whether display plotted images or not.
        save : str or False
            Whether save plotter images or not. If str, must be a path for image saving.
            If path contains '*' symbol, it will be substited by `short_name` attribute of horizon.
        return_figure : bool
            Whether return figure or not.
        figure_params : TODO

        Usage
        -----
        1. How to compose `attributes` argument:

        Show 'amplitudes' attribute of horizon:
        >>> attributes = 'amplitudes'

        Show 'amplitudes' and 'instant_amplitudes' attributes of horizon on separate subplots:
        >>> attributes = ['amplitudes', 'instant_amplitudes']

        Show 'amplitudes' and 'instant_amplitudes' attributes of horizon and its 'channels' subset on separate subplots:
        >>> attributes = ['amplitudes', 'channels/amplitudes', 'instant_amplitudes', 'channels/instant_amplitudes']

        Show 'amplitudes' attribute of horizon and overlay it with 'channels' subset `masks` attribute:
        >>> attributes = [['amplitudes', 'channels/masks']]

        Show 'amplitudes' attribute of horizon and overlay it with 'channels' subset `masks` attribute
        on a separate subplot:
        >>> attributes=[['amplitudes'], ['amplitudes', 'channels/masks']]

        2. How to compose `linkage` argument:

        Show 'amplitudes' attribute of horizon with 'min/max' normalization and 'tab20c' colormap:
        >>> linkage = [
            [
                {
                    'load': dict(src_attribute='amplitudes', normalize='min/max'),
                    'show': dict(cmap='tab20c')
                }
            ]
        ]

        Show 'amplitudes' and 'instant_amplitudes' attributes of horizon with 'min/max' normalization
        and 'tab20c' colormap on separate subplots:
        >>> linkage = [
            [
                {
                    'load': dict(src_attribute='amplitudes', normalize='min/max'),
                    'show': dict(cmap='tab20c')
                }
            ],
            [
                {
                    'load': dict(src_attribute='instant_amplitudes', normalize='min/max'),
                    'show': dict(cmap='tab20c')
                }
            ]
        ]

        Show 'amplitudes' and 'instant_amplitudes' attributes of horizon with 'min/max' normalization
        and 'tab20c' colormap on separate subplots and overlay them with `masks` attribute of 'channels` subset:
        >>> linkage = [
            [
                {
                    'load': dict(src_attribute='amplitudes', normalize='min/max'),
                    'show': dict(cmap='tab20c')
                },
                {
                    'load': dict(src_attribute='channels/masks')
                }
            ],
            [
                {
                    'load': dict(src_attribute='instant_amplitudes', normalize='min/max'),
                    'show': dict(cmap='tab20c')
                },
                {
                    'load': dict(src_attribute='channels/masks')
                }
            ]
        ]
        """
        # pylint: disable=too-many-statements
        def make_figure(label, figure_params, linkage):

            default_figure_params = {
                # general parameters
                'scale': 10,
                'mode': 'overlap',
                # for `plt.subplots`
                'figure/tight_layout': True,
                # for `plt.suptitle`
                'suptitle/y': 1.1,
                'suptitle/size': 25,
                # for every subplot
                'subplot/mode': 'overlap',
            }

            figure_params = Config({**default_figure_params, **figure_params})
            mode = figure_params['mode']
            if mode == 'overlap':
                x, y = label.matrix.shape
                min_ax = min(x, y)
                figsize = [(x / min_ax) * len(linkage), y / min_ax]
            elif mode == 'hist':
                figsize = [len(linkage), 0.5]
            else:
                raise ValueError(f"Expected `subplot/mode` from `['overlap', 'hist']`, but {mode} was given.")
            figure_params['figure/figsize'] = np.array(figsize) * figure_params['scale']
            figure_params['figure/ncols'] = len(linkage)

            fig, axes = plt.subplots(**figure_params['figure'])
            axes = to_list(axes)

            default_suptitle = f"attributes for `{label.short_name}` horizon on `{label.geometry.displayed_name}` cube"
            if mode == 'hist':
                default_suptitle = f"histogram of {default_suptitle}"

            figure_params['suptitle/t'] = figure_params.get('suptitle/t', default_suptitle)
            fig.suptitle(**figure_params['suptitle'])

            figure_params['subplot/mode'] = mode
            return fig, axes, figure_params['subplot']

        def update_data_params(params, layer, label):
            default_load_params = {'fill_value': np.nan}
            load = layer.get('load')
            if isinstance(load, np.ndarray):
                data = load
                data_name = 'user data'
            elif isinstance(load, dict):
                load = {**default_load_params, **load}
                data_name = load['src_attribute']
                if data_name.startswith('apply:'):
                    layer['apply'] = getattr(label, data_name.split('apply:')[1])
                    data = np.array(np.nan)
                    data_name = None
                else:
                    data = label.load_attribute(**load).squeeze()
                    params['xlim'], params['ylim'] = label.bbox[:2]
            else:
                msg = f"Data to load can be either `np.array` or `dict` of params for `{type(label)}.load_attribute`."
                raise ValueError(msg)

            postprocess = layer.get('postprocess', None)
            if postprocess is not None:
                exec_callable(postprocess)

            if params['mode'] == 'hist':
                data = data.flatten()

            if data_name:
                params['image'].append(data)
            params['data_name'] = data_name

        def update_show_params(params, layer, layer_num):
            data_name = params.pop('data_name')
            if data_name is None:
                return None

            def generate_default_color(layer_num, mode):
                colors_order = [3, 2, 1, 0, 4, 5, 6, 8, 9, 7]
                default_colors = np.array(sns.color_palette('muted', as_cmap=True))[colors_order]
                color_num = layer_num - 1 if mode == 'overlap' else layer_num
                return default_colors[color_num % len(default_colors)]

            default_cmaps = {
                'depths': 'Depths',
                'metrics': 'Metrics'
            }

            mode = params['mode']
            layer_label = ' '.join(data_name.split('/'))
            layer_color = generate_default_color(layer_num, mode)

            defaults = {
                'base': {
                    'title_label': layer_label, 'legend_label': layer_label,
                    'color': layer_color, 'legend_color': layer_color,
                },
                'overlap': {
                    'cmap': default_cmaps.get(data_name.split('/')[-1], 'ocean'),
                    'colorbar': True,
                    'alpha': 0.8,
                    'legend_size': 20,
                    'xlabel': self.geometry.index_headers[0],
                    'ylabel': self.geometry.index_headers[1],
                    },
                'hist': {
                    'bins': 50,
                    'colorbar': False,
                    'alpha': 0.9,
                    'legend_size': 10
                    }
            }

            show = {
                **defaults['base'],
                **defaults[mode],
                **layer.get('show', {}),
            }

            base_primary_params = ['title_label', 'title_y']
            primary_params = []
            base_secondary_params = ['alpha', 'color', 'legend_label', 'legend_size', 'legend_color']
            secondary_params = []
            if mode == 'overlap':
                if layer_num == 0:
                    primary_params = base_primary_params
                    primary_params += ['cmap', 'colorbar', 'aspect', 'fraction', 'xlabel', 'ylabel']
                else:
                    secondary_params = base_secondary_params
            elif mode == 'hist':
                if layer_num == 0:
                    primary_params = base_primary_params + ['bins']
                    secondary_params = base_secondary_params
                else:
                    secondary_params = base_secondary_params
            _ = [params.update({param: show[param]}) for param in primary_params if param in show]
            _ = [params[param].append(show[param]) for param in secondary_params if param in show]

            return None

        def apply_extra_actions(params, layer):
            apply = layer.get('apply', None)
            if apply is None:
                return None
            if isinstance(apply, str):
                return getattr(self, apply)(**params)
            return exec_callable(apply, **params)


        if (attributes is not None) and (linkage is not None):
            raise ValueError("Can't use both `attributes` and `linkage`.")

        if linkage is None:
            attributes = attributes or 'depths'
            if isinstance(attributes, str):
                subplots_attributes = [[attributes]]
            elif isinstance(attributes, list):
                subplots_attributes = [to_list(item) for item in attributes]
            else:
                raise ValueError("`attributes` can be only str or list")

            linkage = []
            for layer_attributes in subplots_attributes:
                subplot = [{'load': dict(src_attribute=attribute)} for attribute in layer_attributes]
                linkage.append(subplot)

        fig, axes, subplot_params = make_figure(label=self, figure_params=figure_params, linkage=linkage)
        for axis, subplot_layers in zip(axes, linkage):
            params = defaultdict(list)
            params['ax'] = axis
            params['show'] = show
            params['savepath'] = save.replace('*', self.short_name) if save else None
            _ = [params.update({k: v}) for k, v in subplot_params.items()]
            for layer_num, layer in enumerate(subplot_layers):
                update_data_params(params=params, layer=layer, label=self)
                update_show_params(params=params, layer=layer, layer_num=layer_num)
            plot_image(**params)
            for layer in subplot_layers:
                apply_extra_actions(params=params, layer=layer)
        return fig if return_figure else None


    def __sub__(self, other):
        if not isinstance(other, type(self)):
            raise TypeError(f"Subtrahend expected to be of {type(self)} type, but appeared to be {type(other)}.")
        minuend, subtrahend = self.full_matrix, other.full_matrix
        presence = other.presence_matrix
        discrepancies = minuend[presence] != subtrahend[presence]
        if discrepancies.any():
            raise ValueError("Horizons have different depths where present.")
        result = minuend.copy()
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
        info = self.grid_info # added to Facies in `FaciesCubeset.make_grid`
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
            grid_kwargs = filter_kwargs(kwargs, filtered_keys, prefix='grid_')
            draw_lines(lines, *lines_range, **grid_kwargs)

            crop_kwargs = filter_kwargs(kwargs, filtered_keys, prefix='crop_')
            draw_lines(lines[0] + crop_shape, lines_range[0], crop_shape, **crop_kwargs) # draw first crop
            draw_lines(lines[-1], lines_range[1] - crop_shape, lines_range[1], **crop_kwargs) # draw last crop
