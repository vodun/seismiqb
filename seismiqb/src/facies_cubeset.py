""" Container for storing seismic data and labels with facies-specific interaction model. """
import os
import json
from glob import glob
from warnings import warn
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit

from .cubeset import SeismicCubeset
from .horizon import Horizon
from .utility_classes import IndexedDict
from .utils import get_environ_flag



class UpdatableDict(dict):
    """ Dictionary extended with ability to be updated on interactive events. """
    def interactive_update(self, update):
        """ Method to pass to ipywidgets `observe` call. """
        # pylint: disable=protected-access
        owner = update['owner']
        subset = owner._subset
        cube = owner._cube
        label = owner._label
        interact = 'append' if update['new'] else 'remove'
        getattr(self[subset][cube], interact)(label)


class SeismicCubesetInfo():
    """ Class to manipulate geometry-labels correspondence stored in json. """
    DEFAULT_INFO = {
        'cubes_dir': '/data/seismic_data/seismic_interpretation',
        'labels_dir': 'INPUTS/FACIES',
        'labels': ['FANS_HORIZONS', 'FANS'],
        'labels_extensions': ['.char'],
        'subsets': {},
        'apply': {}
    }

    def __init__(self, path=None):

        info = SeismicCubesetInfo.DEFAULT_INFO
        if path is not None:
            if path.endswith('.json'):
                with open(path, 'r') as json_info:
                    info = {**info, **json.load(json_info)}
            else:
                info["cubes_dir"] = path
                raise NotImplementedError()

        self.cubes_dir = info.pop("cubes_dir")
        self.cubes = info.pop("cubes", self.list_all_cubes())

        self.labels_dir = info.pop("labels_dir")
        self.labels = info.pop("labels")

        self.labels_extensions = tuple(info.pop("labels_extensions"))
        self.main_labels_dir = info.pop("main_labels", self.infer_main_labels_dir())

        self.subsets = self.make_subsets_storage(info.pop("subsets"))
        self.apply = info.pop("apply")

        if info:
            warn(f"Unknown keys ignored in {os.path.basename(path)}:\n{list(info.keys())}")

    def list_all_cubes(self):
        all_directories = next(os.walk(self.cubes_dir))[1]
        all_cubes = [directory for directory in all_directories if directory.startswith('CUBE_')]
        return all_cubes

    def infer_main_labels_dir(self):
        if len(self.labels) == 1:
            return self.labels[0]
        horizon_labels = [label for label in self.labels if 'HORIZON' in label]
        if not horizon_labels:
            msg = f"""
            Cannot automatically choose main labels directory from {self.labels}.
            Please specify it explicitly.
            """
            raise ValueError(msg)
        return horizon_labels[0]

    def list_main_labels(self, cube):
        main_labels_path = '/'.join([self.cubes_dir, f"CUBE_{cube}", self.labels_dir, self.main_labels_dir])
        main_labels = [file for file in os.listdir(main_labels_path) if file.endswith(self.labels_extensions)]
        return main_labels

    def make_subsets_storage(self, subsets):
        storage = {}
        for k, v in subsets.items():
            storage[k] = defaultdict(list, v)
        return UpdatableDict(storage)

    def interactive_split(self, subsets=('train', 'infer')):
        # pylint: disable=import-outside-toplevel, protected-access
        from ipywidgets import Checkbox, VBox, HBox, Button, Layout

        anonymize = get_environ_flag('SEISMIQB_ANONYMIZE')

        def on_change(v):
            self.subsets.interactive_update(v)

        box_layout = Layout(display='flex', flex_flow='column', align_items='center', width='23%')

        vboxes = []
        for subset in subsets:
            subset_controls = [Button(description=subset, button_style='info')]
            for cube in self.cubes:

                displayed_cube_name = cube[:cube.rfind('_')] if anonymize else cube
                cube_button = Button(description=displayed_cube_name)
                cube_button._subset = subset
                cube_button.observe(on_change, names='value')
                subset_controls.append(cube_button)

                for label in self.list_main_labels(cube):
                    displayed_label_name = label[:label.rfind('.')]
                    default_label_value = label in self.subsets[subset].get(cube, [])
                    label_box = Checkbox(description=displayed_label_name, value=default_label_value)
                    label_box._subset = subset
                    label_box._cube = cube
                    label_box._label = label
                    label_box.observe(on_change, names='value')
                    subset_controls.append(label_box)
            vboxes.append(VBox(subset_controls, layout=box_layout))

        hbox = HBox(vboxes)
        display(hbox)

    def make_cubeset(self, subset=None, dst_labels=None, ext='hdf5', **kwargs):

        if subset is None:
            correspondence = {cube: self.list_main_labels(cube) for cube in self.cubes}
        else:
            correspondence = self.subsets[subset]

        cubes_paths = [f"{self.cubes_dir}/CUBE_{cube}/amplitudes_{cube}.{ext}" for cube in correspondence.keys()]

        dataset = FaciesSeismicCubeset(cubes_paths)

        dst_labels = dst_labels or [label.lower() for label in self.labels]
        dataset.load_labels(label_dir=self.labels_dir, labels_subdirs=self.labels,
                            correspondence=correspondence, dst_labels=dst_labels, **kwargs)

        for function, arguments in self.apply.items():
            indices = [f'amplitudes_{cube}' for cube in arguments.pop('cubes')]
            dataset.apply_to_labels(function=function, indices=indices, src_labels=dst_labels, **arguments)

        return dataset


class FaciesSeismicCubeset(SeismicCubeset):
    """ TODO """


    def load_labels(self, label_dir, labels_subdirs, correspondence, dst_labels=None,
                    main_labels='horizons', label_affix='post', **kwargs):
        """ Load corresponding labels into corresponding dataset attributes.

        Parameters
        ----------
        correspondence : dict
            Correspondence between cube name and a list of patterns for its labels.
        labels_dirs : sequence
            Paths to folders to look corresponding labels with patterns from `correspondence` values for.
            Paths must be relative to cube location.
        dst_labels : sequence
            Names of dataset components to load corresponding labels into.
        main_labels : str
            Which dataset attribute assign to `self.labels`.
        kwargs :
            Passed directly to :meth:`.create_labels`.

        Examples
        --------
        The following argument values may be used to load for labels for 'CUBE_01_XXX':
        - from 'INPUTS/FACIES/FANS_HORIZONS/horizon_01_corrected.char' into `horizons` component;
        - from 'INPUTS/FACIES/FANS/fans_on_horizon_01_corrected_v8.char' into `fans` component,
        and assign `self.horizons` to `self.labels`.

        >>> correspondence = {'CUBE_01_XXX' : ['horizon_01']}
        >>> labels_dirs = ['INPUTS/FACIES/FANS_HORIZONS', 'INPUTS/FACIES/FANS']
        >>> dst_labels = ['horizons', 'fans']
        >>> main_labels = 'horizons'
        """
        self.load_geometries()

        dst_labels = dst_labels or [labels_subdir.lower() for labels_subdir in labels_subdirs]
        for labels_subdir, dst_label in zip(labels_subdirs, dst_labels):
            paths = defaultdict(list)
            for cube_name, labels in correspondence.items():
                full_cube_name = f"amplitudes_{cube_name}"
                cube_path = self.index.get_fullpath(full_cube_name)
                cube_dir = cube_path[:cube_path.rfind('/')]
                for label in labels:
                    label_mask = f"*{label}" if label_affix == 'pre' else f"{label}*" if label_affix == 'post' else None
                    label_mask = '/'.join([cube_dir, label_dir, labels_subdir, label_mask])
                    label_path = glob(label_mask)
                    if len(label_path) == 0:
                        raise ValueError(f"No files found for mask `{label_mask}`")
                    if len(label_path) > 1:
                        raise ValueError('Multiple files match pattern')
                    paths[full_cube_name].append(label_path[0])
            self.create_labels(paths=paths, dst=dst_label, labels_class=Horizon, **kwargs)

        if main_labels is not None:
            if not getattr(self, main_labels, False):
                alt_main_labels = dst_labels[0]
                msg = f"""
                Attribute `{main_labels}` does not exist in dataset.
                Cubeset `labels` automatically set to point to `{alt_main_labels}`.
                To avoid this behaviour specify attribute existing in dataset.
                """
                warn(msg)
                main_labels = alt_main_labels
            self.labels = getattr(self, main_labels) #pylint: disable=attribute-defined-outside-init

    def show_labels(self, indices=None, main_labels='labels', overlay_labels=None, attributes=None, correspondence=None,
                    scale=10, colorbar=True, main_cmap='viridis', overlay_cmap='autumn', overlay_alpha=0.7,
                    suptitle_size=25, title_size=15, transpose=True, plot_figures=True, return_figures=False):
        """ Show specific attributes for labels of selected cubes with optional overlay by other attributes.

        Parameters
        ----------
        indices : list of int, list of str or None
            Cubes indices to show labels for. If None, show labels for all cubes. Defaults to None.
        main_labels : str
            Name of cubeset attribute to get labels from for the main image.
        overlay_labels : str
            Name of cubeset attribute to get labels from for the overlay image.
        attributes : str or list of str
            Names of label attribute to show in a row (incompatible with `correspondence` arg).
        correspondence : list of dicts
            Alternative plotting specifications allowing nuanced vizualizations (incompatible with `attributes` arg).
            Each item of a list must be a dict defining label-attribute correspondence for specific subplot in a row.
            This dict should consist of `str` keys with the names of cubeset attributes to get labels for plotting from
            and `dict` values, specifying at least an attribute to plot for selected label. This allows plotting both
            non-overlayed and overlayed by mask images of specific label attribute, e.g.:
            >>> [
            >>>     {
            >>>         'labels' : dict(attribute='amplitudes', cmap='tab20c')
            >>>     },
            >>>     {
            >>>         'labels' : dict(attribute='amplitudes', cmap='tab20c'),
            >>>         'channels': dict(attribute='masks', cmap='Blues', alpha=0.5)
            >>>     }
            >>> ]
        scale : number
            How much to scale the figure.
        colorbar : bool
            Whether plot colorbar for every subplot.
            May be overriden for specific subplot by `colorbar` from `correspondence` arg if specified there.
        main_cmap : str
            Default name of colormap for main image.
            May be overriden for specific subplot by `cmap` from `correspondence` arg if specified there.
        overlay_cmap : str
            Default name of colormap for overlay image.
            May be overriden for specific subplot by `cmap` from `correspondence` arg if specified there.
        overlay_alpha : float from [0, 1]
            Default opacity for overlay image.
            May be overriden for specific subplot by `alpha` from `correspondence` arg if specified there.
        suptitle_size : int
            Fontsize of suptitle for a row of images.
        title_size : int
            Size of titles for every subplot in a row.
        transpose : bool
            Whether plot data in `plot_image` way or not.
        """
        #pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt
        from mpl_toolkits import axes_grid1

        def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
            """Add a vertical color bar to an image plot. """
            divider = axes_grid1.make_axes_locatable(im.axes)
            width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
            pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
            cax = divider.append_axes("right", size=width, pad=pad)
            return im.axes.figure.colorbar(im, cax=cax, **kwargs)

        indices = indices or self.indices
        if correspondence is None:
            attributes = ['heights'] if attributes is None else attributes
            attributes = [attributes] if isinstance(attributes, str) else attributes
            correspondence = [{main_labels: dict(attribute=attribute)} for attribute in attributes]
        elif attributes is not None:
            raise ValueError("Can't use both `correspondence` and `attributes`.")

        figures = []
        for idx in indices:
            for label_num, label in enumerate(self[idx, main_labels]):
                x, y = label.matrix.shape
                min_shape = min(x, y)
                figaspect = np.array([x / min_shape * len(correspondence), y / min_shape]) * scale
                fig, axes = plt.subplots(ncols=len(correspondence), figsize=figaspect, constrained_layout=True)
                figures.append(fig)
                axes = axes if isinstance(axes, np.ndarray) else [axes]

                fig.suptitle(f"`{label.name}` on `{label.geometry.displayed_name}`", size=suptitle_size, y=1.1)
                main_label_bounds = tuple(slice(*lims) for lims in label.bbox[:2])
                for ax, src_params in zip(axes, correspondence):
                    if overlay_labels is not None and overlay_labels not in src_params:
                        src_params[overlay_labels] = dict(attribute='masks')
                    attributes = []
                    title = []
                    for layer_num, (src, params) in enumerate(src_params.items()):
                        attribute = params['attribute']
                        attributes.append(attribute)
                        data = self[idx, src, label_num].load_attribute(attribute, fill_value=np.nan)

                        if layer_num == 0:
                            alpha = 1
                            cmap = params.get('cmap', main_cmap)
                        else:
                            alpha = params.get('alpha', overlay_alpha)
                            cmap = params.get('cmap', overlay_cmap)

                        if len(data.shape) > 2 and data.shape[2] != 1:
                            data = data[..., data.shape[2] // 2 + 1]
                        data = data.squeeze()
                        data = data[main_label_bounds]
                        data = data.T if transpose else data

                        im = ax.imshow(data, cmap=cmap, alpha=alpha)
                        local_colorbar = params.get('colorbar', colorbar)
                        if local_colorbar and layer_num == 0:
                            add_colorbar(im)
                        layer_title = params.get('title', None)
                        title += [layer_title] if layer_title else []
                    title = title or np.unique(attributes)
                    title = ', '.join(title)
                    ax.set_title(title, size=title_size)
                if not plot_figures:
                    plt.close(fig)
        if return_figures:
            return figures

    def apply_to_labels(self, function, indices, src_labels, **kwargs):
        """ Call specific function for labels attributes of specific cubes.

        function : str or callable
            If str, name of the function or method to call from the attribute.
            If callable, applied directly to each item of cubeset attribute from `attributes`.
        indices : sequence of str
            For the attributes of which cubes to call `function`.
        src_labels : sequence of str
            For what cube label to call `function`.
        kwargs :
            Passed directly to `function`.

        Examples
        --------
        >>> cubeset.apply_to_labels('smooth_out', ['CUBE_01_XXX', 'CUBE_02_YYY'], ['horizons', 'fans'}, iters=3])
        """
        for src in src_labels:
            for idx in indices:
                items = getattr(self, src).get(idx, None)
                if items is None:
                    warn(f"Can't call `{function} for {src} of cube {idx}, since it is not in index.")
                    continue
                for item in items:
                    res = function(item, **kwargs) if callable(function) else getattr(item, function)(**kwargs)
                    if res is not None:
                        warn(f"Call for {item} returned not None, which is not expected.")

    def show_grid(self, src_labels='labels', labels_indices=None, attribute='cube_values', plot_dict=None):
        """ Plot grid over selected surface to visualize how it overlaps data.

        Parameters
        ----------
        src_labels : str
            Labels to show below the grid.
            Defaults to `labels`.
        labels_indices : str
            Indices of items from `src_labels` to show below the grid.
        attribute : str
            Alias from :attr:`~Horizon.FUNC_BY_ATTR` to show below the grid.
        plot_dict : dict, optional
            Dict of plot parameters, such as:
                figsize : tuple
                    Size of resulted figure.
                title_fontsize : int
                    Font size of title over the figure.
                attr_* : any parameter for `plt.imshow`
                    Passed to attribute plotter
                grid_* : any parameter for `plt.hlines` and `plt.vlines`
                    Passed to grid plotter
                crop_* : any parameter for `plt.hlines` and `plt.vlines`
                    Passed to corners crops plotter
        """
        #pylint: disable=import-outside-toplevel
        from matplotlib import pyplot as plt

        labels = getattr(self, src_labels)[self.grid_info['cube_name']]
        if labels_indices is not None:
            labels_indices = [labels_indices] if isinstance(labels_indices, int) else labels_indices
            labels = [labels[i] for i in labels_indices]

        # Calculate grid lines coordinates
        (x_min, x_max), (y_min, y_max) = self.grid_info['range'][:2]
        x_stride, y_stride = self.grid_info['strides'][:2]
        x_crop, y_crop = self.grid_info['crop_shape'][:2]
        x_lines = list(np.arange(0, x_max, x_stride)) + [x_max - x_crop]
        y_lines = list(np.arange(0, y_max, y_stride)) + [y_max - y_crop]

        default_plot_dict = {
            'figsize': (20 * x_max // y_max, 10),
            'title_fontsize': 18,
            'attr_cmap' : 'tab20b',
            'grid_color': 'darkslategray',
            'grid_linestyle': 'dashed',
            'crop_color': 'crimson',
            'crop_linewidth': 3
        }
        plot_dict = default_plot_dict if plot_dict is None else {**default_plot_dict, **plot_dict}
        attr_plot_dict = {k.split('attr_')[-1]: v for k, v in plot_dict.items() if k.startswith('attr_')}
        attr_plot_dict['zorder'] = 0
        grid_plot_dict = {k.split('grid_')[-1]: v for k, v in plot_dict.items() if k.startswith('grid_')}
        grid_plot_dict['zorder'] = 1
        crop_plot_dict = {k.split('crop_')[-1]: v for k, v in plot_dict.items() if k.startswith('crop_')}
        crop_plot_dict['zorder'] = 2

        _fig, axes = plt.subplots(ncols=len(labels), figsize=plot_dict['figsize'])
        axes = axes if isinstance(axes, np.ndarray) else [axes]

        for ax, label in zip(axes, labels):
            # Plot underlaying attribute
            underlay = label.load_attribute(attribute, transform={'fill_value': np.nan})
            if len(underlay.shape) == 3:
                underlay = underlay[:, :, underlay.shape[2] // 2].squeeze()
            underlay = underlay.T
            ax.imshow(underlay, **attr_plot_dict)
            ax.set_title("Grid over `{}` on `{}`".format(attribute, label.name), fontsize=plot_dict['title_fontsize'])

            # Set limits
            ax.set_xlim([x_min, x_max])
            ax.set_ylim([y_max, y_min])

            # Plot grid
            ax.vlines(x_lines, y_min, y_max, **grid_plot_dict)
            ax.hlines(y_lines, x_min, x_max, **grid_plot_dict)

            # Plot first crop
            ax.vlines(x=x_lines[0] + x_crop, ymin=y_min, ymax=y_crop, **crop_plot_dict)
            ax.hlines(y=y_lines[0] + y_crop, xmin=x_min, xmax=x_crop, **crop_plot_dict)

            # Plot last crop
            ax.vlines(x=x_lines[-1], ymin=y_max - x_crop, ymax=y_max, **crop_plot_dict)
            ax.hlines(y=y_lines[-1], xmin=x_max - y_crop, xmax=x_max, **crop_plot_dict)

    def make_predictions(self, pipeline, crop_shape, overlap_factor, order=(1, 2, 0), src_labels='labels',
                         dst_labels='predictions', pipeline_var='predictions', bar='n', binarize=True):
        """
        Make predictions and put them into dataset attribute.

        Parameters
        ----------
        pipeline : Pipeline
            Inference pipeline.
        crop_shape : sequence
            Passed directly to :meth:`.make_grid`.
        overlap_factor : float or sequence
            Passed directly to :meth:`.make_grid`.
        src_labels : str
            Name of dataset component with items to make grid for.
        dst_labels : str
            Name of dataset component to put predictions into.
        pipeline_var : str
            Name of pipeline variable to get predictions for assemble from.
        order : tuple of int
            Passed directly to :meth:`.assemble_crops`.
        binarize : bool
            Whether convert probability to class label or not.
        """
        # pylint: disable=blacklisted-name
        setattr(self, dst_labels, IndexedDict({ix: [] for ix in self.indices}))
        for idx, labels in getattr(self, src_labels).items():
            for label in labels:
                self.make_grid(cube_name=idx, crop_shape=crop_shape, overlap_factor=overlap_factor,
                               heights=int(label.h_mean), mode='2d')
                pipeline = pipeline << self
                pipeline.run(batch_size=self.size, n_iters=self.grid_iters, bar=bar)
                prediction = self.assemble_crops(pipeline.v(pipeline_var), order=order).squeeze()
                prediction = expit(prediction)
                prediction = prediction.round() if binarize else prediction
                prediction_name = "{}_predicted".format(label.name)
                self[idx, dst_labels] += [Horizon(prediction, label.geometry, prediction_name)]

    def evaluate(self, true_src, pred_src, metrics_fn, output_format='df'):
        """ TODO """
        if output_format == 'dict':
            results = {}
            for idx in self.indices:
                results[idx] = []
                for true, pred in zip(self[idx, true_src], self[idx, pred_src]):
                    true_mask = true.load_attribute('masks', fill_value=0)
                    pred_mask = pred.load_attribute('masks', fill_value=0)
                    result = metrics_fn(true_mask, pred_mask)
                    results[idx].append(result)
        elif output_format == 'df':
            columns = ['cube', 'horizon', 'metrics']
            rows = []
            for idx in self.indices:
                for true, pred in zip(self[idx, true_src], self[idx, pred_src]):
                    true_mask = true.load_attribute('masks', fill_value=0)
                    pred_mask = pred.load_attribute('masks', fill_value=0)
                    metrics_value = metrics_fn(true_mask, pred_mask)
                    row = np.array([idx.lstrip('amplitudes_'), true.name, metrics_value])
                    rows.append(row)
            data = np.stack(rows)
            results = pd.DataFrame(data=data, columns=columns)
        return results
