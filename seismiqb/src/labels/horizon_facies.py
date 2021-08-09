""" Seismic facies container. """
# pylint: disable=too-many-statements
from copy import copy

import numpy as np
import pandas as pd


from .horizon import Horizon
from ..utils import to_list



class Facies(Horizon):
    """ Extends base class functionality, allowing interaction with label subsets.

    Parameters
    ----------
    subsets : dict, optional
        Subsets of the initialized label. Keys are subset names and values are `Facies` instances.
    args, kwargs : misc
        For `Horizon` base class.

    Methods
    -------
    - Class methods here rely heavily on the concept of nested subset storage. Facies are labeled along the horizon and
    therefore can be viewed as subsets of those horizons, if compared as sets of triplets of points they consist of.

    Main methods for interaction with label subsets are `add_subset` and `get_subset`. First allows adding given label
    instance under provided name into parent subsets storage. Second returns the subset label under the requested name.

    - If facies are added as subsets to their base horizons then their attributes can be accessed via the base label.
    This is how `Facies` is different from `Horizon`, where different labels are stored in separate class attributes.

    - Facies predictions might be stored as separate subsets of base labels (a horizon they are segmented along).
    Method `evaluate` is used for comparison of ground truth labels and predicted ones, if both are stored as subsets.

    - Since attributes calculation is cached, there may be a need to free up storage. Method `reset_caches` does that.

    - To save facies call `dump`.
    """

    # Correspondence between attribute alias and the class function that calculates it

    def __init__(self, storage, geometry, name=None, dtype=np.int32, subsets=None, **kwargs):
        super().__init__(storage=storage, geometry=geometry, name=name, dtype=dtype, **kwargs)
        self.subsets = subsets or {}


    # Subsets interaction
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

    def cut_subset(self, matrix, name=None):
        """ Make new label with points matrix filtered by given presense matrix.

        Parameters
        ----------
        matrix : np.array
            Presense matrix of labels points. Must be in full cubes coordinates.
            If consists of 0 and 1, keep points only where values are 1.
            If consists of values from [0, 1] interval, keep points where values are greater than 0.5.
        name : str or None
            Name for new label. If None, original label name used.

        Returns
        -------
        New `Facies` instance with filtered points matrix.
        """
        other = copy(self)
        other.name = name or self.name

        filtering_matrix = (matrix < 0.5).astype(int)
        other.filter_matrix(filtering_matrix)

        return other

    # Matrix inversion
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


    # Load attributes from subsets
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
            Passed directly to attribute-evaluating methods from :attr:`.ATTRIBUTE_TO_METHOD` depending on `src`.

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

        data = super().load_attribute(src=src, location=location, **kwargs)

        if subset_name:
            subset = self.get_subset(subset_name)
            # pylint: disable=protected-access
            mask = subset.load_attribute(src='masks', location=location, fill_value=0).astype(np.bool)
            data[~mask] = kwargs.get('fill_value', self.FILL_VALUE)

        return data


    # Metrics evaluations
    def evaluate(self, src_true, src_pred, metrics_fn, metrics_names=None):
        """ Apply given function to 'masks' attribute of requested labels.

        Parameters
        ----------
        src_true : str
            Name of `labels` subset to load true mask from.
        src_pred : str
            Name of `labels` subset to load prediction mask from.
        metrics_fn : callable or list of callable
            Metrics function(s) to calculate.
        metrics_name : str, optional
            Name of the columns with metrics values in resulted dataframe.
        """
        pd.options.display.float_format = '{:,.3f}'.format

        labeled_traces = self.load_attribute('masks', fill_value=0).astype(bool)
        true = self.load_attribute(f"{src_true}/masks", fill_value=0)[labeled_traces]
        pred = self.load_attribute(f"{src_pred}/masks", fill_value=0)[labeled_traces]

        metrics_fn = to_list(metrics_fn)
        values = [fn(true, pred) for fn in metrics_fn]
        names = to_list(metrics_names) if metrics_names is not None else [fn.__name__ for fn in metrics_fn]
        data = dict(zip(names, values))

        index = pd.MultiIndex.from_arrays([[self.geometry.displayed_name], [self.short_name]],
                                          names=['geometry_name', 'horizon_name'])
        df = pd.DataFrame(index=index, data=data)
        return df


    # Manage data
    def dump(self, path, ext='.char', log=True):
        """ Save facies. """
        path = self.make_savepath(path, ext)
        super().dump(path)
        if log:
            print(f"`{self.short_name}` saved to `{path}`")

    def reset_cache(self):
        """ Clear cached data. """
        super().reset_cache()
        for subset_label in self.subsets.values():
            subset_label.reset_cache()
