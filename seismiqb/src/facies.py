""" Containers for storing seismic data and labels with facies-specific interaction model. """
import os
import json
from copy import copy
from warnings import warn
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
import seaborn as sns

from .cubeset import SeismicCubeset
from .horizon import Horizon
from .metrics import METRIC_CMAP
from .plotters import DEPTHS_CMAP
from .utility_classes import IndexedDict
from .utils import get_environ_flag, to_list

from .plotters import plot_image
from ..batchflow import Config

class UpdatableDict(defaultdict):
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


class FaciesInfo():
    """ Class to load and reorder geometry-labels linkage stored in json.

    Initialized from either path to json file or keyword arguments.
    Key-value structure in json file must match parameters setup described below.
    Some items are optional — their defaults are defined in class variable `DEFAULT_INFO`.

    Parameters
    ----------
    json_path : str
        Path to json file containing required parameters.
        If provided, all following arguments are ignored.
    cubes_dir : str, optional
        Path to cubes storage folder.
    cubes : list of str
        Names of cubes to include without 'CUBE_' prefix.
    cubes_extension : str
        Name of valid cube extension.
    labels_dir : str
        Path to folder containing corresponding labels subfolders.
        Must be relative to loaded geometry location.
    labels : list of str
        Path to folders containing corresponding labels.
        Must be relative to `labels_dir` folder.
    main_labels : str, optional
        Name of `labels_dir` subfolder containing labels to put in dataset `labels` attribute.
        Generally, one wants horizon labels to be main.
    labels_extension : str
        Name of valid label extension.
    subsets : nested dict
        keys : str
            Subset names.
        values : dict
            keys : str
                Cubes included in subset.
            values : list of str
                Cube labes included in subset.
    apply : nested dict
        keys : str
            Method name to call from `FaciesCubeset`
        values : dict
            keys : str
                Method arguments names.
            values : any
                Method arguments values.

    Notes
    -----
    Cubes names in parameters above are implied to be in 'XX_YYY' format, i.e. without 'CUBE_' prefix.

    Example of json file
    --------------------
    {
        "cubes_dir": "/data/cubes",
        "cubes": ["01_AAA", "02_BBB"],
        "cubes_extension": ".hdf5",

        "labels_dir": "INPUTS/FACIES",
        "labels": ["FANS_HORIZONS", "FANS"],
        "main_labels": "FANS_HORIZONS",
        "labels_extension": ".char",

        "subsets":
        {
            "train":
            {
                "01_AAA": ["horizon_1"],
                "02_BBB": ["horizon_2"],
            },
            "infer":
            {
                "01_AAA": ["horizon_2"],
                "02_BBB": ["horizon_1"],
            }
        },

        "apply":
        {
            "smooth_out":
            {
                "cubes": ["02_BBB"],
                "preserve_borders": false
            }
        }
    }

    """

    DEFAULT_INFO = {
        'cubes_dir': '/data/seismic_data/seismic_interpretation',
        'cubes_extension': '.hdf5',
        'labels_extension': '.char',
        'main_labels': None,
        'subsets': {},
        'apply': {}
    }

    def __init__(self, json_path=None, **kwargs):
        info = self.DEFAULT_INFO
        if json_path is not None:
            with open(json_path, 'r') as json_info:
                info = {**info, **json.load(json_info)}
        else:
            info = {**info, **kwargs}

        self.cubes_dir = info.pop("cubes_dir")
        self.cubes = info.pop("cubes")
        self.cubes_extension = info.pop("cubes_extension")

        self.labels_dir = info.pop("labels_dir")
        self.labels = info.pop("labels")
        self.labels_extension = info.pop("labels_extension")

        self.main_labels = self.infer_main_labels(info.pop("main_labels"))
        self.subsets = self.make_subsets_storage(info.pop("subsets"))

        self.apply = info.pop("apply")

        if info:
            arguments_source = os.path.basename(json_path) if json_path is not None else 'kwargs'
            warn(f"Unknown arguments ignored in {arguments_source}:\n{list(info.keys())}")

    def infer_main_labels(self, main_labels):
        """ If not provided explicitly, choose main labels folder name from the list of given labels subfolders. """
        if main_labels is None:
            if len(self.labels) > 1:
                horizon_labels = [label for label in self.labels if 'HORIZON' in label]
                if not horizon_labels:
                    msg = f"""
                    Cannot automatically choose main labels directory from {self.labels}.
                    Please specify it explicitly.
                    """
                    raise ValueError(msg)
                main_labels = horizon_labels[0]
            else:
                main_labels = self.labels[0]
            warn(f"Main labels automatically inferred as `{main_labels}`")
        return main_labels

    def list_main_labels(self, cube):
        """ Make a list of existing labels for cube. """
        main_labels_path = f"{self.cubes_dir}/CUBE_{cube}/{self.labels_dir}/{self.main_labels}"
        main_labels = []
        try:
            for file in os.listdir(main_labels_path):
                if file.endswith(self.labels_extension):
                    main_labels.append(file)
        except FileNotFoundError:
            warn(f"Path {main_labels_path} does not exits.")
        if not main_labels:
            msg = f"No labels with {self.labels_extension} extension found in {main_labels_path}."
            warn(msg)
        return main_labels

    def make_subsets_storage(self, subsets):
        """ Wrap subsets linkage info with flexible nested structure.
        Besides given `subsets` cubes-labels linkage create 'all' subset,
        containing all possible labels for every provided cube name.
        """
        storage = {'all': {cube: self.list_main_labels(cube) for cube in self.cubes}}
        for k, v in subsets.items():
            storage[k] = defaultdict(list, v)
        return UpdatableDict(lambda: defaultdict(list), storage)

    def interactive_split(self, subsets=('train', 'infer'), main_subset='all'):
        """ Render interactive menu to include/exclude labels for every name in `subsets`. """
        # pylint: disable=import-outside-toplevel, protected-access
        from ipywidgets import Checkbox, VBox, HBox, Button, Layout

        anonymize = get_environ_flag('SEISMIQB_ANONYMIZE')

        subsets = to_list(subsets)

        box_layout = Layout(display='flex', flex_flow='column', align_items='center', width='23%')

        vboxes = []
        for subset in subsets:
            subset_controls = [Button(description=subset, button_style='info')]
            for cube in self.cubes:

                displayed_cube_name = cube[:cube.rfind('_')] if anonymize else cube
                cube_button = Button(description=displayed_cube_name)
                cube_button._subset = subset
                cube_button.observe(self.subsets.interactive_update, names='value')
                subset_controls.append(cube_button)

                for label in self.subsets[main_subset][cube]:
                    displayed_label_name = label.rstrip(self.labels_extension)
                    default_label_value = label in self.subsets[subset].get(cube, [])
                    label_box = Checkbox(description=displayed_label_name, value=default_label_value)
                    label_box._subset = subset
                    label_box._cube = cube
                    label_box._label = label
                    label_box.observe(self.subsets.interactive_update, names='value')
                    subset_controls.append(label_box)
            vboxes.append(VBox(subset_controls, layout=box_layout))

        hbox = HBox(vboxes)
        display(hbox)

    def get_subset_linkage(self, subset):
        """ Get cubes-labels linkage for given subset name. """
        linkage = {
            cube: sorted(labels)
            for cube, labels in sorted(self.subsets[subset].items())
            if labels
            }

        if not sum(list(linkage.values()), []):
            msg = f"""
            No labels were selected for subset `{subset}`.
            Either define split in loaded json or via `FaciesInfo.interactive_split`.
            """
            raise ValueError(msg)

        return linkage

    def make_cubeset(self, subset='all', dst_labels=None, **kwargs):
        """ Create `FaciesCubeset` instance from cube-labels linkage defined by `subset`. """
        linkage = self.get_subset_linkage(subset)

        cubes_paths = [
            f"{self.cubes_dir}/CUBE_{cube}/amplitudes_{cube}{self.cubes_extension}"
            for cube in linkage.keys()
        ]
        dataset = FaciesCubeset(cubes_paths)

        dst_labels = dst_labels or [label.lower() for label in self.labels]
        dataset.load_labels(label_dir=self.labels_dir, labels_subdirs=self.labels,
                            linkage=linkage, dst_labels=dst_labels, **kwargs)

        for function, arguments in self.apply.items():
            indices = [f'amplitudes_{cube}' for cube in arguments['cubes'] if cube in linkage.keys()]
            arguments = copy(arguments)
            arguments.pop('cubes')
            dataset.apply_to_labels(function=function, indices=indices, src_labels=dst_labels, **arguments)

        return dataset


class FaciesCubeset(SeismicCubeset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    """


    def load_labels(self, label_dir, labels_subdirs, linkage, dst_labels=None,
                    main_labels='horizons', add_subsets=True, **kwargs):
        """ Load corresponding labels into given dataset attributes.
        Optionally add secondary labels as subsets into main labels.
        Adress `FaciesHorizon` docs for details on subsets implementation.

        Parameters
        ----------
        label_dir : str
            Path to folder with corresponding labels subfolders.
            Must be relative to loaded geometry location.
        labels_subdirs : sequence
            Path to folders containing corresponding labels.
            Must be relative to `labels_dir` folder.
        linkage : dict
            Correspondence between cube name and a list of patterns for its labels.
        dst_labels : sequence
            Names of dataset components to load corresponding labels into.
        main_labels : str
            Which dataset attribute assign to `self.labels`.
        add_subsets : bool
            Whether add corresponding labels as subset to main labels or not.
        kwargs :
            Passed directly to :meth:`.create_labels`.

        Examples
        --------
        Given following arguments:

        >>> label_dir = 'INPUTS/FACIES'
        >>> labels_subdirs = ['FANS_HORIZON', 'FANS']
        >>> linkage = {'CUBE_01_AAA' : ['horizon_1.char'], 'CUBE_01_BBB' : ['horizon_2.char']}
        >>> dst_labels = ['horizons', 'fans']
        >>> main_labels = 'horizons'

        Following actions will be performed:
        >>> load labels into `self.horizon` component from:
            - CUBE_01_AAA/INPUTS/FACIES/FANS_HORIZONS/horizon_1.char
            - CUBE_02_BBB/INPUTS/FACIES/FANS_HORIZONS/horizon_2.char
        >>> load labels into `self.fans` component from:
            - CUBE_01_AAA/INPUTS/FACIES/FANS/horizon_1.char
            - CUBE_02_BBB/INPUTS/FACIES/FANS/horizon_2.char
        >>> assign `self.horizons` to `self.labels`.
        """
        self.load_geometries()

        default_dst_labels = [labels_subdir.lower() for labels_subdir in labels_subdirs]
        dst_labels = to_list(dst_labels, default=default_dst_labels)

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

        for labels_subdir, dst_label in zip(labels_subdirs, dst_labels):
            paths = defaultdict(list)
            for cube_name, labels in linkage.items():
                full_cube_name = f"amplitudes_{cube_name}"
                cube_path = self.index.get_fullpath(full_cube_name)
                cube_dir = cube_path[:cube_path.rfind('/')]
                for label in labels:
                    label_path = f"{cube_dir}/{label_dir}/{labels_subdir}/{label}"
                    paths[full_cube_name].append(label_path)
            self.create_labels(paths=paths, dst=dst_label, labels_class=FaciesHorizon, **kwargs)
            if add_subsets and (dst_label != main_labels):
                self.add_subsets(subset_labels=dst_label, main_labels=main_labels)

        setattr(self, 'labels', getattr(self, main_labels))

    def add_subsets(self, subset_labels, main_labels='labels'):
        flatten_main_labels = self.flatten_labels(src_labels=main_labels)
        flatten_subset_labels = self.flatten_labels(src_labels=subset_labels)
        if len(flatten_main_labels) != len(flatten_subset_labels):
            raise ValueError(f"Labels `{subset_labels}` and `{main_labels}` have different lengths.")
        for main_label, subset_label in zip(flatten_main_labels, flatten_subset_labels):
            main_label.add_subset(subset_labels, subset_label)

    def flatten_labels(self, src_labels='labels', indices=None):
        """ Convert given labels attribute from `IndexedDict` to `list`. Optionally filter out required indices. """
        indices = to_list(indices, default=self.indices)
        indices = [idx if isinstance(idx, str) else self.indices[idx] for idx in indices]
        labels_lists = [self[idx, src_labels] for idx in indices]
        return sum(labels_lists, [])

    @property
    def flat_labels(self):
        return self.flatten_labels()

    def apply_to_labels(self, function, indices=None, src_labels='labels', **kwargs):
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
        src_labels = to_list(src_labels)
        results = {}
        for src in src_labels:
            results[src] = {}
            for label in self.flatten_labels(src_labels=src, indices=indices):
                    res = function(label, **kwargs) if callable(function) else getattr(label, function)(**kwargs)
                    results[src][label.short_name] = res
        return results

    def show_labels(self, attributes=None, src_labels='labels', indices=None, linkage=None,
                    plot=True, save=None, return_figures=False, **figure_params):
        figures = self.apply_to_labels(function='show_label', src_labels=src_labels, indices=indices,
                                       attributes=attributes, linkage=linkage, plot=plot, save=save,
                                       return_figure=return_figures, **figure_params)
        return figures if return_figures else None

    def invert_subsets(self, subset, indices=None, src_labels='labels', dst_labels=None, add_subsets=True):
        dst_labels = dst_labels or f"{subset}_inverted"
        inverted = self.apply_to_labels(function='invert_subset', indices=indices, src_labels=src_labels, subset=subset)
        results = IndexedDict({idx: [] for idx in self.indices})
        for _, label in inverted[src_labels].items():
            results[label.cube_name].append(label)
        setattr(self, dst_labels, results)
        if add_subsets:
            self.add_subsets(subset_labels=dst_labels, main_labels=src_labels)

    def add_merged_labels(self, src_labels, dst_labels, indices=None, add_subsets_to='labels'):
        results = IndexedDict({idx: [] for idx in self.indices})
        indices = to_list(indices, default=self.indices)
        for idx in indices:
            to_merge = self[idx, src_labels]
            # since `merge_list` merges all horizons into first object from the list,
            # make a copy of first horizon in list to save merge into its instance
            container = copy(to_merge[0])
            container.name = f"Merged {'/'.join([horizon.short_name for horizon in to_merge])}"
            [container.adjacent_merge(horizon, inplace=True, mean_threshold=999, adjacency=999) for horizon in to_merge]
            container.reset_cache()
            results[idx].append(container)
        setattr(self, dst_labels, results)
        if add_subsets_to:
            self.add_subsets(subset_labels=dst_labels, main_labels=add_subsets_to)

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
            ax.set_title("Grid over `{}` on `{}`".format(attribute, label.short_name), fontsize=plot_dict['title_fontsize'])

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
                         dst_labels='predictions', prefix='_predicted', add_subsets=True,
                         pipeline_variable='predictions', bar='n', binarize=True):
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
        results = IndexedDict({ix: [] for ix in self.indices})
        pbar = tqdm(self.flatten_labels(src_labels))
        pbar.set_description("General progress")
        for label in pbar:
            prediction_name = f"{label.short_name}{prefix}"
            prediction = FaciesHorizon(label.full_matrix, label.geometry, prediction_name)
            cube_name = label.geometry.short_name
            self.make_grid(cube_name=cube_name, crop_shape=crop_shape, overlap_factor=overlap_factor,
                           heights=int(label.h_mean), mode='2d')
            pipeline = pipeline << self
            pipeline.update_config({'src_labels': src_labels})
            pipeline.run(batch_size=self.size, n_iters=self.grid_iters, bar=bar)
            predicted_matrix = expit(self.assemble_crops(pipeline.v(pipeline_variable), order=order).squeeze())
            prediction.filter_matrix(~(predicted_matrix.round().astype(bool)))
            setattr(prediction, "predicted_matrix", predicted_matrix)
            results[cube_name].append(prediction)
        setattr(self, dst_labels, results)
        if add_subsets:
            self.add_subsets(subset_labels=dst_labels, main_labels=src_labels)

    def evaluate(self, true_src, pred_src, metrics_fn, output_format='df'):
        """ TODO """
        pd.options.display.float_format = '{:,.3f}'.format
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
                    row = np.array([idx.lstrip('amplitudes_'), true.short_name, metrics_value])
                    rows.append(row)
            data = np.stack(rows)
            results = pd.DataFrame(data=data, columns=columns)
            results['metrics'] = results['metrics'].values.astype(float)
        return results

    def dump_labels(self, path, src_labels, postfix=None, indices=None):
        postfix = src_labels if postfix is None else postfix
        timestamp = datetime.now().strftime('%b-%d_%H-%M-%S')
        path = f"{path}/{timestamp}_{postfix}/"
        os.makedirs(path)
        self.apply_to_labels(function='dump', indices=indices, src_labels=src_labels, path=path)


class FaciesHorizon(Horizon):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subsets = {}

    def add_subset(self, name, item):
        """ Add item to subsets storage.

        Parameters
        ----------
        name : str
            Key to store given horizon under.
        item : FaciesHorizon
            Instance to store.
        """
        # if not isinstance(item, FaciesHorizon):
        #     msg = f"Only instances of `FaciesHorizon` can be added as subsets, but {type(item)} was given."
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
        except:
            msg = f"Requested subset {name} is missing in subsets storage. Availiable subsets are {list(self.subsets)}."
            raise KeyError(msg)

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
        nested_name = src_attribute.split('/')
        src_attribute = nested_name[-1]

        if src_attribute == 'amplitudes':
            kwargs['window'] = kwargs.get('window', 1)

        data = super().load_attribute(src_attribute=src_attribute, **kwargs)
        if len(nested_name) == 1:
            return data
        if len(nested_name) == 2:
            subset = self.get_subset(nested_name[0])
            location = kwargs.get('location', None)
            mask = subset.load_attribute(src_attribute='masks', location=location, fill_value=0).astype(bool)
            data[~mask] = kwargs.get('fill_value', self.FILL_VALUE)
            return data
        if len(nested_name) > 2:
            raise NotImplementedError("Name nestedness greater than 1 is not currently supported.")

    def show_label(self, attributes=None, linkage=None, show=True, save=False, return_figure=False, **figure_params):
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

        def make_data(label, layer, plot_params):
            default_load_params = {'fill_value': np.nan}
            load = layer.get('load')
            if isinstance(load, np.ndarray):
                data = load
                data_name = 'user data'
            elif isinstance(load, dict):
                load = {**default_load_params, **load}
                data_name = load['src_attribute']
                data = label.load_attribute(**load).squeeze()
                label_bounds = tuple(slice(*lims) for lims in label.bbox[:2])
                data = data[label_bounds]
            else:
                msg = f"Data to load can be either `np.array` or `dict` of params for `{type(label)}.load_attribute`."
                raise ValueError(msg)

            postprocess = layer.get('postprocess', None)
            if postprocess is None:
                pass
            elif callable(postprocess):
                data = postprocess(data)
            elif isinstance(postprocess, dict):
                postprocess_params = copy(postprocess)
                postprocess_func = postprocess_params.pop('func')
                data = postprocess_func(data, **postprocess_params)
            else:
                msg = f"Postprocess can be either `callable` or `dict` with callable under 'func' key and its kwargs."
                raise ValueError(msg)

            if plot_params['mode'] == 'hist':
                data = data.flatten()

            return data, data_name

        def update_plot_params(plot_params, layer, layer_num, axis, data, data_name):
            def generate_default_color(layer_num, mode):
                colors_order = [3, 2, 1, 0, 4, 5, 6, 8, 9, 7]
                default_colors = np.array(sns.color_palette('muted', as_cmap=True))[colors_order]
                color_num = layer_num - 1 if mode == 'overlap' else layer_num
                return default_colors[color_num % len(default_colors)]

            default_cmaps = {
                'depths': DEPTHS_CMAP,
                'metrics': METRIC_CMAP
            }

            mode = plot_params['mode']
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

            plot_params['image'].append(data)
            base_primary_params = ['title_label', 'title_y']
            primary_params = []
            base_secondary_params = ['alpha', 'color', 'legend_label', 'legend_size', 'legend_color']
            secondary_params = []
            if mode == 'overlap':
                if layer_num == 0:
                    primary_params = base_primary_params + ['cmap', 'colorbar', 'aspect', 'fraction', 'xlabel', 'ylabel']
                else:
                    secondary_params = base_secondary_params
            elif mode == 'hist':
                if layer_num == 0:
                    primary_params = base_primary_params + ['bins']
                    secondary_params = base_secondary_params
                else:
                    secondary_params = base_secondary_params
            [plot_params.update({param: show[param]}) for param in primary_params if param in show]
            [plot_params[param].append(show[param]) for param in secondary_params if param in show]

            return plot_params

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
            plot_params = defaultdict(list)
            plot_params['ax'] = axis
            plot_params['show'] = show
            plot_params['savepath'] = save.replace('*', self.short_name) if save else None
            [plot_params.update({k: v}) for k, v in subplot_params.items()]
            for layer_num, layer in enumerate(subplot_layers):
                data, data_name = make_data(label=self, layer=layer, plot_params=plot_params)
                plot_params = update_plot_params(plot_params=plot_params, layer=layer, layer_num=layer_num,
                                                 axis=axis, data=data, data_name=data_name)
            plot_image(**plot_params)
        return fig if return_figure else None

    def __sub__(self, other):
        if type(self) != type(other):
            raise TypeError(f"Subtrahend expected to be of {type(self)} type, but appeared to be {type(other)}.")
        minuend, subtrahend = self.full_matrix, other.full_matrix
        presence = other.presence_matrix
        discrepancies = minuend[presence] != subtrahend[presence]
        if discrepancies.any():
            raise ValueError(f"Horizons have different depths where present.")
        result = minuend.copy()
        result[presence] = self.FILL_VALUE
        name = f"{other.name}_inv"
        return type(self)(result, self.geometry, name)

    def invert_subset(self, subset):
        return self - self.get_subset(subset)

    def dump(self, path):
        if os.path.isdir(path):
            path = f"{path}/{self.name}"
        super().dump(path)

    def reset_cache(self):
        """ Clear cached data. """
        super().reset_cache()
        for subset_label in self.subsets.values():
            subset_label.reset_cache()