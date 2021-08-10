""" !!. """
from glob import glob
from difflib import get_close_matches

import numpy as np

from ..geometry import SeismicGeometry
from ..labels import Horizon, Fault, Facies
from ..utils import DelegatingList



class Field:
    """ !!. """
    def __init__(self, geometry, labels=None, labels_class=None, geometry_kwargs=None, labels_kwargs=None, **kwargs):
        # Attributes
        self.labels = []
        self.horizons, self.facies, self.fans, self.channels, self.faults = [], [], [], [], []

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
    CLASS_TO_METHOD = {
        Horizon: '_load_horizons',
        Fault: '_load_faults',
        Facies: '_load_facies',
        SeismicGeometry: '_load_geometries'
    }

    CLASS_TO_NAMES = {
        Horizon: ['horizon'],
        Fault: ['fault'],
        Facies: ['facies', 'fans', 'channels'],
        SeismicGeometry: ['geometries'],
    }
    NAME_TO_CLASS = {name: class_ for class_, names in CLASS_TO_NAMES.items() for name in names}

    def load_labels(self, labels=None, labels_class=None, **labels_kwargs):
        """ !!. """
        if isinstance(labels, str):
            labels = glob(labels)
        if isinstance(labels, (tuple, list)):
            labels = {'labels': labels}
        if not isinstance(labels, dict):
            raise TypeError('TYPE SHOULD BE: str / sequence / dict')

        self._labels = {**labels} #TODO: debug, remove?

        # Labels class: make a dictionary-like object
        if labels_class is None:
            labels_class_dict = {label_dst : None for label_dst in labels.keys()}
        if isinstance(labels_class, type):
            labels_class_dict = {label_dst : labels_class for label_dst in labels.keys()}
        if isinstance(labels_class, dict):
            labels_class_dict = labels_class

        for label_dst, label_src in labels.items():
            # Try getting provided `labels_class`, else fallback on NAME_TO_CLASS closest match
            label_class = labels_class_dict.get(label_dst)

            if label_class is None:
                # label_class = self.NAME_TO_CLASS.get(label_dst)
                name = get_close_matches(label_dst, list(self.NAME_TO_CLASS.keys()), n=1)
                if name:
                    label_class = self.NAME_TO_CLASS[name[0]]

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
            method_name = self.CLASS_TO_METHOD[label_class]
            method = getattr(self, method_name)
            result = method(label_src, **labels_kwargs)
            setattr(self, label_dst, result)

            if 'labels' not in labels and not self.labels:
                setattr(self, 'labels', result)


    def _load_horizons(self, paths, filter=True, interpolate=False, sort=True, **kwargs):
        #pylint: disable=redefined-builtin
        print('IN _LOAD_HORIZONS')
        horizons = []
        for path in paths:
            if isinstance(path, str):
                horizon = Horizon(path, geometry=self.geometry, **kwargs)
                if filter:
                    horizon.filter()
                if interpolate:
                    horizon.interpolate()

            elif isinstance(path, Horizon):
                horizon = path
            horizons.append(horizon)

        return horizons

    def _load_faults(self, paths, **kwargs):
        print('IN _LOAD_FAULTS', paths)
        return []

    def _load_facies(self, paths, **kwargs):
        print('IN _LOAD_FACIES', paths)
        return []

    def _load_geometries(self, paths, **kwargs):
        print('IN _LOAD_GEOMETRIES', paths)
        return []

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
        if not key in vars(self) and hasattr(self.geometry, key):
            return getattr(self.geometry, key)
        raise AttributeError(f'Attribute `{key}` does not exist in either Field or associated Geometry!')

    def __getattribute__(self, key):
        result = super().__getattribute__(key)
        result = DelegatingList(result) if isinstance(result, list) else result
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

    # TODO: cache resets/introspection
    # TODO: visualization

    def __repr__(self):
        return f"""<Field `{self.displayed_name}` at {hex(id(self))}>"""

    def __str__(self):
        processed_prefix = 'un' if self.geometry.has_stats is False else ''
        labels_prefix = ':' if self.labels else ''
        msg = f'Field `{self.displayed_name}` with {processed_prefix}processed geometry{labels_prefix}\n'
        for label in self.labels:
            msg += f'    {label.name}\n'
        return msg
