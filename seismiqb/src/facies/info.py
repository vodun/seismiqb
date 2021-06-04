""" Seismic facies linkage manager. """
import os
import re
import json
from copy import copy
from warnings import warn
from collections import defaultdict

from .cubeset import FaciesCubeset
from ..utils import get_environ_flag, to_list



class FaciesInfo():
    """ Class to load and reorder geometry-labels linkage stored in json.

    Initialized from path to json file or/and keyword arguments.
    Key-value structure in json file must match parameters setup described below.
    Some items are optional — their defaults are defined in class variable `DEFAULT_INFO`.

    Parameters
    ----------
    json_path : str
        Path to json file containing required parameters.
    cubes_dir : str, optional
        Path to cubes storage folder.
    cubes : list of str
        Names of cubes to include without 'CUBE_' prefix.
    cubes_extension : str
        Name of valid geometry extension.
    labels_dir : str
        Path to folder containing corresponding labels subfolders.
        Must be relative to loaded geometry location.
    labels : list of str
        Path to folders containing corresponding labels.
        Must be relative to `labels_dir` folder.
    labels_extension : str
        Name of valid label extension.
    base_labels : str, optional
        Name of `labels_dir` subfolder containing labels to put in dataset `labels` attribute.
        Generally, one wants horizon labels to be base.
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
        "base_labels": "FANS_HORIZONS",
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
        'cubes': [],
        'cubes_extension': '.qblosc',

        'labels_dir': "INPUTS/HORIZONS",
        'labels': ["RAW"],
        'labels_extension': '.char',
        'base_labels': "RAW",

        'subsets': {},
        'apply': {}
    }

    def __init__(self, json_path=None, **kwargs):
        """ Info from json file has lower priority than from kwargs, so one can easily redefine required arguments."""
        self.info = self._process_arguments(json_path, **kwargs)
        self.subsets = self._make_subsets_storage()

    @classmethod
    def _process_arguments(cls, json_path, **kwargs):
        """ Merge arguments from json and kwargs, filter out unknown arguments and process known. """
        info = {}
        if json_path is not None:
            with open(json_path, 'r') as f:
                info = json.load(f)
        info.update(kwargs)

        # Pop any unexpected keys from `info`
        provided_keys = list(info.keys())
        unrecognized_keys = [info.pop(key) for key in provided_keys if key not in cls.DEFAULT_INFO]
        if unrecognized_keys:
            warn(f"Unknown arguments ignored:\n{unrecognized_keys}")

        info = {**cls.DEFAULT_INFO, **info}
        info['cubes'] = cls._process_cubes_list(info['cubes'], info['cubes_dir'])
        info['base_labels'] = cls._process_base_labels(info['base_labels'], info['labels'])
        info['labels'] = cls._process_labels(info['labels'], info['cubes_dir'], info['cubes'], info['labels_dir'])

        return info

    @classmethod
    def _process_cubes_list(cls, cubes, cubes_dir):
        """ If given intergers sequence, find corresponding cubes folders name. """
        cubes = to_list(cubes)
        all_cubes = [dir.split('CUBE_')[1] for dir in os.listdir(cubes_dir) if dir.startswith('CUBE_')]
        if not cubes:
            chosen_cubes = all_cubes
        elif isinstance(cubes[0], int):
            chosen_cubes = []
            for cube in all_cubes:
                # extract cube numbers from their names
                numbers = re.findall(r'\d+', cube)
                if len(numbers) == 0:
                    warn(f"No numbers occured in cube name {cube}. Skipping.")
                elif len(numbers) != 1:
                    warn(f"Multiple numbers occured in cube name {cube}. Skipping.")
                elif int(numbers[0]) in cubes:
                    chosen_cubes.append(cube)
        else:
            chosen_cubes = cubes
        return chosen_cubes

    @classmethod
    def _process_labels(cls, labels, cubes_dir, cubes, labels_dir):
        labels = to_list(labels)
        for cube in cubes:
            labels_path = f"{cubes_dir}/CUBE_{cube}/{labels_dir}"
            existing_subdirs = next(os.walk(labels_path))[1]
            for label in labels:
                if label not in existing_subdirs:
                    msg = f"""
                        Label subdir {label} was not found in {labels_path}.
                        Available subdirs are {existing_subdirs}.
                    """
                    warn(msg)
                    labels.pop(label)
        return labels

    @classmethod
    def _process_base_labels(cls, base_labels, labels):
        """ If not provided explicitly, choose base labels folder name from the list of given labels subfolders. """
        if base_labels not in labels:
            msg = f"""
            Main labels {base_labels} are not in {labels}.
            """
            if len(labels) > 1:
                horizon_labels = [label for label in labels if 'HORIZON' in label]
                if not horizon_labels:
                    msg += f"""
                    Cannot automatically choose new base labels directory from {labels}.
                    Please specify it explicitly.
                    """
                    raise ValueError(msg)
                base_labels = horizon_labels[0]
            else:
                base_labels = labels[0]
            msg += f"""
            Main labels automatically inferred as `{base_labels}`.
            """
            warn(msg)
        return base_labels

    def __getattr__(self, name):
        """ If attribute is not present in dataset, try retrieving it from dict-like storage. """
        return self.info[name]

    def _get_cube_labels(self, cube):
        """ Make a list of existing labels for cube. """
        base_labels_path = f"{self.cubes_dir}/CUBE_{cube}/{self.labels_dir}/{self.base_labels}"

        base_labels = []
        if os.path.exists(base_labels_path):
            for file in os.listdir(base_labels_path):
                if file.endswith(self.labels_extension):
                    base_labels.append(file)
        else:
            warn(f"Path {base_labels_path} does not exist.")

        if not base_labels:
            msg = f"No labels with {self.labels_extension} extension found in {base_labels_path}."
            warn(msg)

        return base_labels

    def _make_subsets_storage(self):
        """ Wrap subsets linkage info with flexible nested structure.
        Besides cubes-labels linkage given in `self.info['subsets']`,
        create subset containing all possible labels for every cube name under 'all' key.
        """
        result = defaultdict(lambda: defaultdict(list))

        for cube in self.cubes:
            result['all'][cube] = self._get_cube_labels(cube)
        for subset, linkage in self.subsets.items():
            result[subset] = defaultdict(list, linkage)

        return result

    def _update_on_event(self, event):
        """ Method to pass to ipywidgets `observe` call. """
        # pylint: disable=protected-access
        event_name = type(event).__name__

        if event_name == 'Bunch':
            # if event is a checkbox click
            subset = event['owner']._subset
            cube = event['owner']._cube
            label = event['owner']._label
            interact = 'append' if event['new'] else 'remove'
            getattr(self.subsets[subset][cube], interact)(label)
        elif event_name == 'Button':
            # if event is a button click
            cubes = getattr(event, '_cubes', None)
            if cubes is None:
                # if cube button
                boxes = getattr(event, '_labels', None)
            else:
                # if subset button
                boxes = sum([cube._labels for cube in cubes], [])
            first_box_value = boxes[0].value
            for box in boxes:
                box.value = not first_box_value

    def interactive_split(self, subsets=('train', 'infer'), main_subset='all'):
        """ Render interactive menu to include/exclude labels for every name in `subsets`. """
        # pylint: disable=import-outside-toplevel, protected-access
        from ipywidgets import Checkbox, VBox, HBox, Button, Layout
        from IPython.display import display

        anonymize = get_environ_flag('SEISMIQB_ANONYMIZE')

        subsets = to_list(subsets)

        box_layout = Layout(display='flex', flex_flow='column', align_items='center', width='23%')

        vboxes = []
        for subset in subsets:
            subset_controls = []
            subset_button = Button(description=subset, button_style='info')
            subset_button._cubes = []
            subset_button.on_click(self._update_on_event)
            subset_controls.append(subset_button)
            for cube in self.cubes:

                displayed_cube_name = cube[:cube.rfind('_')] if anonymize else cube
                cube_button = Button(description=displayed_cube_name)
                cube_button._labels = []
                cube_button._controls = subset_controls
                cube_button._subset = subset
                cube_button.on_click(self._update_on_event)
                subset_button._cubes.append(cube_button)
                subset_controls.append(cube_button)

                for label in self.subsets[main_subset][cube]:
                    displayed_label_name = label.rstrip(self.labels_extension)
                    default_label_value = label in self.subsets[subset].get(cube, [])
                    label_box = Checkbox(description=displayed_label_name, value=default_label_value)
                    label_box._subset = subset
                    label_box._cube = cube
                    label_box._label = label
                    label_box.observe(self._update_on_event, names='value')
                    cube_button._labels.append(label_box)
                    subset_controls.append(label_box)
            vboxes.append(VBox(subset_controls, layout=box_layout))

        hbox = HBox(vboxes)
        display(hbox)

    def _get_subset_linkage(self, subset):
        """ Get cubes-labels linkage for given subset name. """
        linkage = {
            cube: sorted(labels)
            for cube, labels in sorted(self.subsets[subset].items())
            if labels
            }

        if not sum(linkage.values(), []):
            msg = f"""
            No labels were selected for subset `{subset}`.
            Either choose non-empty subset or add some labels current one.
            Labels can be added in either loaded json or via `FaciesInfo.interactive_split`.
            """
            raise ValueError(msg)

        return linkage

    def make_cubeset(self, subset='all', dst_labels=None, cube_file_prefix='', **kwargs):
        """ Create `FaciesCubeset` instance from cube-labels linkage defined by `subset`. """
        linkage = self._get_subset_linkage(subset)

        cubes_paths = [
            f"{self.cubes_dir}/CUBE_{cube}/{cube_file_prefix}{cube}{self.cubes_extension}"
            for cube in linkage.keys()
        ]

        linkage = {f"{cube_file_prefix}{k}": v for k, v in linkage.items()}
        dataset = FaciesCubeset(cubes_paths)

        dst_labels = dst_labels or [label.lower() for label in self.labels]
        dataset.load_labels(label_dir=self.labels_dir, labels_subdirs=self.labels,
                            linkage=linkage, dst_labels=dst_labels, **kwargs)

        for function, arguments in self.apply.items():
            cubes = self._process_cubes_list(arguments['cubes'], self.cubes_dir)
            indices = [cube for cube in cubes if cube in linkage.keys()]
            arguments = copy(arguments)
            arguments.pop('cubes')
            dataset.apply_to_labels(function=function, indices=indices, src_labels=dst_labels, **arguments)

        return dataset

    def dump(self, path):
        """ Save info. """
        with open(path, 'w') as f:
            json.dump(self.info, f)
