""" Container for storing seismic data and labels with facies-specific interaction model. """
from copy import copy
from warnings import warn
from datetime import datetime
from collections import defaultdict

import pandas as pd

from .facies import Facies
from .batch import FaciesBatch
from ..cubeset import SeismicCubeset
from ..utility_classes import IndexedDict
from ..utils import to_list



class FaciesCubeset(SeismicCubeset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    """
    # pylint: disable=useless-super-delegation
    def __init__(self, *args, batch_class=FaciesBatch, **kwargs):
        super().__init__(*args, batch_class=batch_class, **kwargs)


    def load_labels(self, label_dir, labels_subdirs, linkage, dst_labels=None,
                    base_labels='horizons', add_subsets=True, **kwargs):
        """ Load corresponding labels into given dataset attributes.
        Optionally add secondary labels as subsets into base labels.
        Adress `Facies` docs for details on subsets implementation.

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
        base_labels : str
            Which dataset attribute assign to `self.labels`.
        add_subsets : bool
            Whether add corresponding labels as subset to base labels or not.
        kwargs :
            Passed directly to :meth:`.create_labels`.

        Examples
        --------
        Given following arguments:

        >>> label_dir = 'INPUTS/FACIES'
        >>> labels_subdirs = ['FANS_HORIZON', 'FANS']
        >>> linkage = {'CUBE_01_AAA' : ['horizon_1.char'], 'CUBE_02_BBB' : ['horizon_2.char']}
        >>> dst_labels = ['horizons', 'fans']
        >>> base_labels = 'horizons'

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
        dst_labels = to_list(dst_labels or default_dst_labels)

        if base_labels not in dst_labels:
            alt_base_labels = dst_labels[0]
            msg = f"Provided base_labels `{base_labels}` are not in `dst_labels` and were set automatically "\
                  f"to `{alt_base_labels}`. That means, that dataset `labels` will point to `{alt_base_labels}`. "\
                  f"To override this behaviour provide `base_labels` from `dst_labels={dst_labels}`."
            warn(msg)
            base_labels = alt_base_labels

        for labels_subdir, dst_label in zip(labels_subdirs, dst_labels):
            paths = defaultdict(list)
            for cube_name, labels in linkage.items():
                cube_path = self.index.get_fullpath(cube_name)
                cube_dir = cube_path[:cube_path.rfind('/')]
                for label in labels:
                    label_path = f"{cube_dir}/{label_dir}/{labels_subdir}/{label}"
                    paths[cube_name].append(label_path)
            self.create_labels(paths=paths, dst=dst_label, labels_class=Facies, **kwargs)
            if add_subsets and (dst_label != base_labels):
                self.add_subsets(subset_labels=dst_label, base_labels=base_labels)

        setattr(self, 'labels', getattr(self, base_labels))

    def add_subsets(self, subset_labels, base_labels='labels'):
        """ Add nested labels. """
        flat_base_labels = getattr(self, base_labels).flat
        flat_subset_labels = getattr(self, subset_labels).flat
        if len(flat_base_labels) != len(flat_subset_labels):
            raise ValueError(f"Labels `{subset_labels}` and `{base_labels}` have different lengths.")
        for base_label, subset_label in zip(flat_base_labels, flat_subset_labels):
            base_label.add_subset(subset_labels, subset_label)

    def map_labels(self, function, indices=None, src_labels='labels', **kwargs):
        """ Call function for every item from labels list of requested cubes and return produced results.

        Parameters
        ----------
        function : str or callable
            If str, name of label method to call.
            If callable, applied to labels of chosen cubes.
        indices : sequence of str
            Which cubes' labels to map.
        src_labels : str
            Attribute with labels to map.
        kwargs :
            Passed directly to `function`.

        Returns
        -------
        IndexedDict where keys are cubes names and values are lists of results obtained by applied map.
        If all lists in result values are empty, None is returned instead.

        Examples
        --------
        >>> cubeset.map_labels('smooth_out', ['CUBE_01_AAA', 'CUBE_02_BBB'], 'horizons', iters=3)
        """
        results = IndexedDict({idx: [] for idx in self.indices})
        for label in getattr(self, src_labels).flatten(keys=indices):
            if isinstance(function, str):
                res = getattr(label, function)(**kwargs)
            elif callable(function):
                res = function(label, **kwargs)
            if res is not None:
                results[label.geometry.short_name].append(res)
        return results if len(results.flat) > 0 else None

    def show(self, attributes='depths', src_labels='labels', indices=None, **kwargs):
        """ Show attributes of multiple dataset labels. """
        return self.map_labels(function='show', src_labels=src_labels, indices=indices, attributes=attributes, **kwargs)

    def invert_subsets(self, subset, src_labels='labels', dst_labels=None, add_subsets=True):
        """ Apply `invert_subset` for every given label and put it into cubeset. """
        dst_labels = dst_labels or f"{subset}_inverted"
        inverted = self.map_labels(function='invert_subset', indices=None, src_labels=src_labels, subset=subset)

        setattr(self, dst_labels, inverted)
        if add_subsets:
            self.add_subsets(subset_labels=dst_labels, base_labels=src_labels)

    def add_merged_labels(self, src_labels, dst_labels, indices=None, add_subsets_to='labels'):
        """ Merge given labels and put result into cubeset. """
        results = IndexedDict({idx: [] for idx in self.indices})
        indices = to_list(indices or self.indices)
        for idx in indices:
            to_merge = self[idx, src_labels]
            # since `merge_list` merges all horizons into first object from the list,
            # make a copy of first horizon in list to save merge into its instance
            container = copy(to_merge[0])
            container.name = f"Merged {'/'.join([horizon.short_name for horizon in to_merge])}"
            _ = [container.adjacent_merge(horizon, inplace=True, mean_threshold=999, adjacency=999)
                 for horizon in to_merge]
            container.reset_cache()
            results[idx].append(container)
        setattr(self, dst_labels, results)
        if add_subsets_to:
            self.add_subsets(subset_labels=dst_labels, base_labels=add_subsets_to)

    def evaluate(self, src_true, src_pred, metrics_fn, indices=None, src_labels='labels'):
        """ TODO """
        metrics_values = self.map_labels(function='evaluate', src_labels=src_labels, indices=indices,
                                              src_true=src_true, src_pred=src_pred, metrics_fn=metrics_fn)
        return pd.concat(metrics_values.flat)

    def dump_labels(self, path, src_labels, postfix=None, indices=None):
        """ TODO """
        postfix = src_labels if postfix is None else postfix
        timestamp = datetime.now().strftime('%b-%d_%H-%M-%S')
        path = f"{path}/{timestamp}_{postfix}/"
        self.map_labels(function='dump', indices=indices, src_labels=src_labels, path=path)
