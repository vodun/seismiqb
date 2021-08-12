""" !!. """
import os
import re
from glob import glob
from difflib import get_close_matches

import numpy as np

from .visualization import VisualizationMixin
from ..geometry import SeismicGeometry
from ..labels import Horizon, Fault, Facies
from ..utils import AugmentedList, transformable



class Field(VisualizationMixin):
    """ !!. """
    def __init__(self, geometry, labels=None, labels_class=None, geometry_kwargs=None, labels_kwargs=None, **kwargs):
        # Attributes
        self.labels = []
        self.horizons, self.facies, self.fans, self.channels, self.faults = [], [], [], [], []
        self.loaded_labels = []

        # Geometry: description and convenient API to a seismic cube
        if isinstance(geometry, str):
            geometry_kwargs = geometry_kwargs or {}
            geometry = SeismicGeometry(geometry, **{**kwargs, **geometry_kwargs})
        self.geometry = geometry

        # Labels: objects on a field
        if labels:
            labels_kwargs = labels_kwargs or {}
            self.load_labels(labels, labels_class, **{**kwargs, **labels_kwargs})


    # Label initialization inner workings
    METHOD_TO_NAMES = {
        '_load_horizons': ['horizon', Horizon],
        '_load_faults': ['fault', Fault],
        '_load_facies': ['facies', 'fans', 'channels', Facies],
        '_load_geometries': ['geometries', 'geometry',  SeismicGeometry],
    }
    NAME_TO_METHOD = {name: method for method, names in METHOD_TO_NAMES.items() for name in names}

    def load_labels(self, labels=None, labels_class=None, **labels_kwargs):
        """ !!. """
        if isinstance(labels, str):
            labels = glob(labels)
        if isinstance(labels, (tuple, list)):
            labels = {'labels': labels}
        if not isinstance(labels, dict):
            raise TypeError('TYPE SHOULD BE: str / sequence / dict')

        # Labels class: make a dictionary
        if labels_class is None:
            labels_class_dict = {label_dst : None for label_dst in labels.keys()}
        if isinstance(labels_class, (type, str)):
            labels_class_dict = {label_dst : labels_class for label_dst in labels.keys()}
        if isinstance(labels_class, dict):
            labels_class_dict = labels_class

        for label_dst, label_src in labels.items():
            # Try getting provided `labels_class`, else fallback on NAME_TO_CLASS closest match
            label_class = labels_class_dict.get(label_dst)

            if label_class is None:
                # Roughly equivalent to ``label_class = self.NAME_TO_METHOD.get(label_dst)``
                str_names = [name for name in (self.NAME_TO_METHOD.keys())
                             if isinstance(name, str)]
                matched = get_close_matches(label_dst, str_names, n=1)
                if matched:
                    label_class = matched[0]

            if label_class is None:
                raise TypeError(f"Can't determine the label class for `{label_dst}`!")

            # Process paths: get rid of service files
            if isinstance(label_src, str):
                label_src = glob(label_src)
            if not isinstance(label_src, (tuple, list)):
                label_src = [label_src]
            label_src = [item for item in label_src
                         if (isinstance(item, str) and ('.dvc' not in item) \
                             and ('.gitignore' not in item) and ('.meta' not in item)) \
                            or not isinstance(item, str)]

            # Load desired labels, based on class
            method_name = self.NAME_TO_METHOD[label_class]
            method = getattr(self, method_name)
            result = method(label_src, **labels_kwargs)

            setattr(self, label_dst, result)
            self.loaded_labels.append(label_dst)

            if 'labels' not in labels and not self.labels:
                setattr(self, 'labels', result)


    def _load_horizons(self, paths, filter=True, interpolate=False, sort=True, **kwargs):
        #pylint: disable=redefined-builtin
        horizons = []
        for path in paths:
            if isinstance(path, str):
                horizon = Horizon(path, field=self, **kwargs)
                if filter:
                    horizon.filter()
                if interpolate:
                    horizon.interpolate()

            elif isinstance(path, Horizon):
                horizon = path
            horizons.append(horizon)

        if sort:
            sort = sort if isinstance(sort, str) else 'h_mean'
            horizons.sort(key=lambda label: getattr(label, sort))
        return horizons

    def _load_faults(self, paths, **kwargs):
        print('IN _LOAD_FAULTS', paths)
        return []

    def _load_facies(self, paths, **kwargs):
        print('IN _LOAD_FACIES', paths)
        return []

    def _load_geometries(self, paths, **kwargs):
        if len(paths) > 1:
            raise ValueError('!!.')
        path = paths[0]
        return SeismicGeometry(path, **kwargs)

    # Other methods of initialization
    @classmethod
    def from_horizon(cls, horizon):
        """ !!. """
        return cls(geometry=horizon.geometry, labels={'horizons': horizon})

    @classmethod
    def from_dvc(cls, tag, dvc_path=''):
        """ !!. """


    # Inner workings
    def __getattr__(self, key):
        if hasattr(self.geometry, key):
            return getattr(self.geometry, key)
        raise AttributeError(f'Attribute `{key}` does not exist in either Field or associated Geometry!')

    def __getattribute__(self, key):
        result = super().__getattribute__(key)
        if isinstance(result, list) and not isinstance(result, AugmentedList):
            result = AugmentedList(result)
            setattr(self, key, result)
        return result


    # Public methods. Usually, used by Batch class
    def load_seismic(self, location, slicing='custom', src='geometry', **kwargs):
        """ !!. """
        geometry = getattr(self, src)

        if slicing == 'native':
            seismic_crop = geometry[tuple(location)]
        elif slicing == 'custom':
            seismic_crop = geometry.load_crop(location, **kwargs)
        else:
            raise ValueError(f"Slicing must be either 'native' or 'custom', not {slicing}!.")
        return seismic_crop

    def make_mask(self, location, shape, indices='all', width=3, src='labels', **kwargs):
        """ !!. """
        mask = np.zeros(shape, dtype=np.float32)

        labels = getattr(self, src)
        labels = [labels] if not isinstance(labels, (tuple, list)) else labels
        if len(labels) == 0:
            return mask

        indices = [indices] if isinstance(indices, int) else indices
        if isinstance(indices, (tuple, list, np.ndarray)):
            labels = [labels[idx] for idx in indices]
        elif indices in ['single', 'random']:
            labels = np.random.shuffle(labels)[0]


        for label in labels:
            mask = label.add_to_mask(mask, locations=location, width=width)
            if indices in ['single', 'random'] and mask.sum() > 0.0:
                break
        return mask


    # Attribute retrieval
    def matrix_fill_to_num(self, matrix, value):
        """ Change the matrix values at points where field is absent to a supplied one. """
        matrix[np.isnan(matrix)] = value
        return matrix

    def matrix_normalize(self, matrix, mode):
        """ Normalize matrix values.

        Parameters
        ----------
        mode : bool, str, optional
            If `min-max` or True, then use min-max scaling.
            If `mean-std`, then use mean-std scaling.
            If False, don't scale matrix.
        """
        values = matrix[~np.isnan(matrix)]

        if mode in ['min-max', True]:
            min_, max_ = np.nanmin(values), np.nanmax(values)
            matrix = (matrix - min_) / (max_ - min_)
        elif mode == 'mean-std':
            mean, std = np.nanmean(values), np.nanstd(values)
            matrix = (matrix - mean) / std
        else:
            raise ValueError(f'Unknown normalization mode `{mode}`.')
        return matrix

    def load_attribute(self, src, **kwargs):
        """ !!. """
        # 'zero_traces'
        # 'horizons:0/amplitudes'
        # 'horizons:0/channels/masks'
        src = src.strip('/')

        if '/' not in src:
            data = self.get_property(src=src, **kwargs)

        else:
            label_id, *src = src.split('/')
            label_attr, label_idx = re.split(':|_|-', label_id)

            if label_attr not in self.loaded_labels:
                matched = get_close_matches(label_attr, self.loaded_labels, n=1)
                if matched:
                    label_attr = matched[0]
                else:
                    raise ValueError(f"Can't determine the label attribute for `{label_attr}`!")
            label_idx = int(label_idx)
            label = getattr(self, label_attr)[label_idx]

            src = '/'.join(src)
            data = label.load_attribute(src, **kwargs)
        return data

    @transformable
    def get_property(self, src):
        """ !!. """
        return getattr(self, src)


    # Utility functions
    def make_savepath(self, path, name=None, makedirs=True):
        """ !!. """
        basedir = os.path.dirname(self.path)
        name = name or self.short_name

        path = (path.replace('**', basedir)
                    .replace('%', basedir)
                    .replace('*', name)
                    .replace('//', '/'))

        if makedirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        return path


    # TODO: cache resets/introspection
    def reset_cache(self):
        """ Clear cached data from underlying entities. """
        for attribute in set(['geometry'] + self.loaded_labels):
            getattr(self, attribute).reset_cache()

    @property
    def cache_size(self):
        """ Total size of cached data. """
        size = self.geometry.cache_size
        for attribute in set(self.loaded_labels):
            size += sum(getattr(self, attribute).cache_size)
        return size


    # TODO: subsets
    def add_subsets(self, src_subset, dst_base='labels'):
        """ Add nested labels.

        Parameters
        ----------
        src_labels : str
            Name of field attribute with labels to add as subsets.
        dst_base: str
            Name of field attribute with labels to add subsets to.
        """
        subset_labels = getattr(self, src_subset)
        base_labels = getattr(self, dst_base)
        if len(subset_labels.flat) != len(base_labels.flat):
            raise ValueError(f"Labels `{src_subset}` and `{dst_base}` have different lengths.")

        for subset, base in zip(subset_labels, base_labels):
            base.add_subset(name=src_subset, item=subset)

    def invert_subsets(self, subset, src='labels', dst=None, add_subsets=True):
        """ !!. """
        dst = dst or f"{subset}_inverted"
        inverted = getattr(self, src).invert_subset(subset=subset)

        setattr(self, dst, inverted)
        if add_subsets:
            self.add_subsets(src_subset=dst, dst_base=src)
