""" Container for storing seismic data and labels with facies-specific interaction model. """
from copy import copy

import pandas as pd

from ..cubeset import SeismicCubeset
from ..utility_classes import IndexedDict
from ..utils import to_list



class FaciesCubeset(SeismicCubeset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    Most methods basically call methods of the same name for every label stored in requested attribute.
    """

    def add_subsets(self, src_subset, dst_base='labels'):
        """ Add nested labels.

        Parameters
        ----------
        src_labels : str
            Name of dataset attribute with labels to add as subsets.
        dst_base: str
            Name of dataset attribute with labels to add subsets to.
        """
        subset_labels = getattr(self, src_subset)
        base_labels = getattr(self, dst_base)
        if len(subset_labels.flat) != len(base_labels.flat):
            raise ValueError(f"Labels `{src_subset}` and `{dst_base}` have different lengths.")
        for subset, base in zip(subset_labels, base_labels):
            base.add_subset(name=src_subset, item=subset)

    def map_labels(self, function, indices=None, src_labels='labels', **kwargs):
        """ Call function for every item from labels list of requested cubes and return produced results.

        Parameters
        ----------
        function : str or callable
            If str, name of label method to call.
            If callable, applied to labels of chosen cubes.
        indices : str or sequence of str
            Indices of cubes which labels to map.
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
        """ Show attributes of requested dataset labels. """
        return self.map_labels(function='show', src_labels=src_labels, indices=indices, attributes=attributes, **kwargs)

    def invert_subsets(self, subset, src_labels='labels', dst_labels=None, add_subsets=True):
        """ Invert matrices of requested dataset labels and store resulted labels in cubeset. """
        dst_labels = dst_labels or f"{subset}_inverted"
        inverted = self.map_labels(function='invert_subset', indices=None, src_labels=src_labels, subset=subset)

        setattr(self, dst_labels, inverted)
        if add_subsets:
            self.add_subsets(src_subset=dst_labels, dst_base=src_labels)

    def add_merged_labels(self, src_labels, dst_labels, indices=None, dst_base='labels'):
        """ Merge requested labels and store resulted labels in cubeset. """
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
        if dst_base:
            self.add_subsets(src_subset=dst_labels, dst_base=dst_base)

    def evaluate(self, src_true, src_pred, metrics_fn, metrics_names=None, indices=None, src_labels='labels'):
        """ Apply given function to 'masks' attribute of requested labels and return merged dataframe of results.

        Parameters
        ----------
        src_true : str
            Name of `labels` subset to load true mask from.
        src_pred : str
            Name of `labels` subset to load prediction mask from.
        metrics_fn : callable or list of callable
            Metrics function(s) to calculate.
        metrics_name : str, optional
            Name of the column with metrics values in resulted dataframe.
        """
        metrics_values = self.map_labels(function='evaluate', src_labels=src_labels, indices=indices,
                                              src_true=src_true, src_pred=src_pred, metrics_fn=metrics_fn)
        return pd.concat(metrics_values.flat)
