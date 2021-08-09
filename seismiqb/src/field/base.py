""" !!. """
from collections import defaultdict

from glob import glob
from difflib import get_close_matches

from ..geometry import SeismicGeometry
from ..labels import Horizon, Fault, Facies



class Field:
    """ !!. """
    def __init__(self, geometry, labels=None, labels_class=None, geometry_kwargs=None, labels_kwargs=None, **kwargs):
        # Geometry: description and convenient API to a seismic cube
        if isinstance(geometry, str):
            geometry_kwargs = geometry_kwargs or {}
            geometry = SeismicGeometry(geometry, **{**kwargs, **geometry_kwargs})
        self.geometry = geometry

        # Labels: objects on a field
        if labels:
            self.load_labels(labels, labels_class, **{**kwargs, **labels_kwargs})


    # Label initialization inner workings
    CLASS_TO_METHOD = {Horizon: '_load_horizons', Fault: '_load_faults', Facies: '_load_facies'}

    CLASS_TO_NAMES = {
        Horizon: ['horizon'],
        Fault: ['fault'],
        Facies: ['facies', 'fans', 'channels'],
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
            labels_class_dict = defaultdict(lambda: None)
        if isinstance(labels_class, type):
            labels_class_dict = defaultdict(lambda: labels_class)
        if isinstance(labels_class, dict):
            labels_class_dict = labels_class

        for label_dst, label_src in labels.items():
            # Try getting provided `labels_class`, else fallback on NAME_TO_CLASS closest match
            label_class = labels_class_dict.get(label_dst)
            if label_class is None:
                name = get_close_matches(label_dst, list(self.NAME_TO_CLASS.keys()), n=1)
                if name:
                    label_class = self.NAME_TO_CLASS[name[0]]

            if label_class is None:
                raise TypeError(f"Can't determine the label class for {label_dst}!")

            # Process paths: get rid of service files
            if isinstance(label_src, str):
                label_src = glob(label_src)
            label_src = [item for item in label_src
                         if (isinstance(item, str) and self.is_label_path(item)) or not isinstance(item, str)]

            # Load desired labels, based on class
            method_name = self.CLASS_TO_METHOD[label_class]
            method = getattr(self, method_name)
            lst = method(label_src, **labels_kwargs)
            setattr(self, label_dst, lst)


    @staticmethod
    def is_label_path(path):
        """ !!. """
        return ('.dvc' not in path) and ('.gitignore' not in path) and ('.meta' not in path)

    def _load_horizons(self, paths, filter=True, interpolate=False, sort=True, **kwargs):
        #pylint: disable=redefined-builtin
        print('IN _LOAD_HORIZONS')
        horizons = []
        for path in paths:
            horizon = Horizon(path, geometry=self.geometry, **kwargs)
            if filter:
                horizon.filter()
            if interpolate:
                horizon.interpolate()
            horizons.append(horizon)

        return horizons

    def _load_faults(self, paths, **kwargs):
        print('IN _LOAD_FAULTS', paths)
        return []

    def _load_facies(self, paths, **kwargs):
        print('IN _LOAD_FACIES', paths)
        return []

    # Other methods of initialization
    @classmethod
    def from_geometry(cls, geometry):
        """ !!. """

    @classmethod
    def from_horizon(cls, horizon):
        """ !!. """

    @classmethod
    def from_dvc(cls, tag, dvc_path=''):
        """ !!. """
