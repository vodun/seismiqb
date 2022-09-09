""" A mixin to dump/load attributes to an external file. """
import os

import numpy as np
import pandas as pd

import h5py
import hdf5plugin



class MetaMixin:
    """ A mixin to store / load instance attributes.

    To be used correctly, the instance should provide following attributes:
        - `meta_path` defines the path to dump attributes / load them from.
        If not provided, but the instance has `path` attribute, we try to infer the meta path from it:
            - if `path` references the `hdf5` file, we use it as meta container.
            - otherwise, we append `_meta` prefix to the `path` and use it as meta path.

        - `PRESERVED` with sequence of names of attributes to dump/load.
        They are then loaded at subsequent instance initialization.

        - `PRESERVED_LAZY` with sequence of names of attributes to dump/load.
        They are then loaded on demand, at the time of the first access.
    """
    #pylint: disable=redefined-argument-from-local
    PRESERVED = []              # loaded at instance initialization
    PRESERVED_LAZY = []         # loaded on demand

    META_OPENER = h5py.File     # constructor for files

    @property
    def meta_paths(self):
        """ Paths to the files with stored meta. """
        paths = []
        if hasattr(self, 'meta_path') and self.meta_path is not None:
            paths.append(self.meta_path)

        if hasattr(self, 'path'):
            if 'hdf5' in self.path:
                paths.append(self.path)
            paths.append(self.path + '_meta')
        return paths

    @property
    def existing_meta_paths(self):
        """ Existing paths to the files with stored meta. """
        paths = self.meta_paths
        paths = [path for path in paths if os.path.exists(path)]
        return paths

    @property
    def meta_exists(self):
        """ Whether at least one meta path exists. """
        return len(self.existing_meta_paths) > 0

    def has_meta_item(self, key, path=None):
        """ Check whether `key` is present. """
        key = key + '/'
        key = key.replace('//', '/')

        meta_paths = self.meta_paths
        if path is not None:
            meta_paths.insert(0, path)

        for path in meta_paths:
            with self.META_OPENER(path, mode='a') as src:
                if key in src:
                    return True
        return False


    # Dump
    def dump_meta(self, path=None, names=None, overwrite=True):
        """ Save attributes to meta file.

        Parameters
        ----------
        path : str or None
            If provided, then path to the meta file. Otherwise, uses the first path in `meta_paths` property.
        names : sequence or None
            If provided, then attributes to dump. Otherwise, dumps all `PRESERVED` and `PRESERVED_LAZE` attributes.
        overwrite : bool
            Whether to overwrite the file.
        """
        names = names or self.PRESERVED + self.PRESERVED_LAZY

        # Dump each attribute
        for name in names:
            if hasattr(self, name) and getattr(self, name) is not None:
                self.dump_meta_item(key=f'/meta/{name}', value=getattr(self, name),
                                    path=path, overwrite=overwrite)
            else:
                if hasattr(self, 'meta_list_failed_to_dump'):
                    self.meta_list_failed_to_dump.add(name)

    def dump_meta_item(self, key, value, path=None, overwrite=True):
        """ Save one `value` as `key` into the meta file.
        Unlike native `h5py`, works with sequences, dataframes and arrays with `object` dtype.
        """
        key = key + '/'
        key = key.replace('//', '/')

        meta_path = path or self.meta_paths[0]

        with self.META_OPENER(meta_path, mode='a') as dst:
            if overwrite and key in dst:
                del dst[key]

            if isinstance(value, (tuple, list)) or (isinstance(value, np.ndarray) and value.dtype == object):
                # Sequence: store its length and type separately, then dump each item to its own group
                dst[key + 'is_sequence'] = 1
                dst[key + 'length'] = len(value)

                types = {tuple: 0, list: 1, np.ndarray: 2}
                type_ = types[type(value)]
                dst[key + 'type'] = type_

                for i, v in enumerate(value):
                    self.dump_meta_item(path=meta_path, key=key+str(i), value=v)

            elif isinstance(value, pd.DataFrame):
                # Dataframe: store column/index names and values separately
                # TODO: would not work with arbitrary index. Can be improved by dumping index values directly
                dst[key + 'is_dataframe'] = 1
                dst.attrs[key + 'columns'] = list(value.columns)

                index_names = list(value.index.names)
                if index_names[0]:
                    dst.attrs[key + 'index_names'] = index_names
                    values_ = value.reset_index().values
                else:
                    values_ = value.values
                self.dump_meta_item(path=meta_path, key=key+'values', value=values_)
            elif isinstance(value, np.ndarray):
                dst.create_dataset(key.strip('/'), data=value,
                                   **hdf5plugin.Blosc(cname='lz4hc', clevel=6, shuffle=0))
            else:
                dst[key] = value

    # Loading
    def load_meta(self, path=None, names=None):
        """ Load attributes from meta file.

        Parameters
        ----------
        path : str or None
            If provided, then path to the meta file. Otherwise, uses `meta_path` property.
        names : sequence or None
            If provided, then attributes to load. Otherwise, loads all `PRESERVED` attributes.
        """
        names = names or self.PRESERVED

        for name in names:
            value = self.load_meta_item(f'/meta/{name}', path=path)
            setattr(self, name, value)
            if hasattr(self, 'meta_list_loaded'):
                self.meta_list_loaded.add(name)

    def load_meta_item(self, key, path=None):
        """ Load one `key` from meta.
        Unlike native `h5py`, works with sequences, dataframes and arrays with `object` dtype.
        """
        meta_paths = self.existing_meta_paths
        if path is not None:
            meta_paths.insert(0, path)

        key = key + '/'
        key = key.replace('//', '/')

        for path in meta_paths:
            with self.META_OPENER(path, mode='a') as src:
                if key + 'is_sequence' in src:
                    length = src[key + 'length'][()]
                    type_ = src[key + 'type'][()]

                    value = [self.load_meta_item(key=key + str(i), path=path) for i in range(length)]

                    types = {0: tuple, 1: list, 2: np.array}
                    value = types[type_](value)
                elif key + 'is_dataframe' in src:
                    values = src[key + 'values'][()]
                    columns = src.attrs[key + 'columns']

                    value = pd.DataFrame(data=values, columns=columns)

                    if key + 'index_names' in src:
                        index_names = src.attrs[key + 'index_names']
                        value.set_index(index_names, inplace=True)

                elif key in src:
                    value = src[key][()]
                else:
                    continue
                return value
        raise KeyError(f'No key `{key}` in meta! Meta files: {meta_paths}!')


    # Reset
    def reset_meta(self, path=None, names=None):
        """ Delete all keys in meta file. """
        meta_path = path or self.existing_meta_paths[0]
        names = names or (self.PRESERVED + self.PRESERVED_LAZY)

        with self.META_OPENER(meta_path, mode='a') as meta_file:
            for name in names:
                if name in meta_file:
                    del meta_file[name]

                name_ = f'/meta/{name}'
                if name_ in meta_file:
                    del meta_file[name_]

    def remove_meta(self, path=None):
        """ Remove meta file entirely. """
        meta_path = path or self.existing_meta_paths[0]
        os.remove(meta_path)

    # Introspection
    def print_meta_tree(self, path=None):
        """ Print textual representation of meta. """
        meta_path = path or self.meta_paths[0]
        self.META_OPENER(meta_path, mode='r').visit(self._print_meta_tree)

    def _print_meta_tree(self, name):
        """ Print one meta item. """
        shift = name.count('/') * ' ' * 4
        item_name = name.split('/')[-1]
        print(shift + item_name)
