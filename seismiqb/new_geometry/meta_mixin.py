""" A mixin to dump/load attributes to an external file. """
import os

import numpy as np
import pandas as pd

import h5py
import hdf5plugin



class MetaMixin:
    """ !!. """
    #pylint: disable=redefined-argument-from-local
    PRESERVED = []
    PRESERVED_LAZY = []

    META_OPENER = h5py.File

    @property
    def meta_path(self):
        """ !!. """
        if hasattr(self, '_meta_path') and self._meta_path is not None:
            return self._meta_path
        if 'hdf5' in self.path:
            return self.path
        return self.path + '_meta'

    def has_meta_item(self, key, src=None):
        """ !!. """
        key = key + '/'
        key = key.replace('//', '/')

        with src or self.META_OPENER(self.meta_path, mode='a') as src:
            return key in src


    # Dump
    def dump_meta(self, path=None, names=None, overwrite=True):
        """ !!. """
        meta_path = path or self.meta_path
        names = names or self.PRESERVED + self.PRESERVED_LAZY

        # Remove file, if exists: h5py can't do that
        if overwrite and os.path.exists(meta_path):
            os.remove(meta_path)

        #
        with self.META_OPENER(meta_path, mode='a') as meta_file:
            for name in names:
                if hasattr(self, name) and getattr(self, name) is not None:
                    self.dump_meta_item(key=f'/meta/{name}', value=getattr(self, name), dst=meta_file)
                else:
                    if hasattr(self, 'meta_list_failed_to_dump'):
                        self.meta_list_failed_to_dump.add(name)

    def dump_meta_item(self, key, value, dst=None):
        """ !!. """
        key = key + '/'
        key = key.replace('//', '/')

        with dst or self.META_OPENER(self.meta_path, mode='a') as dst:
            if isinstance(value, (tuple, list)) or (isinstance(value, np.ndarray) and value.dtype == object):
                # Sequence: store its length and type separately, then dump each item to its own group
                dst[key + 'is_sequence'] = 1
                dst[key + 'length'] = len(value)

                types = {tuple: 0, list: 1, np.ndarray: 2}
                type_ = types[type(value)]
                dst[key + 'type'] = type_

                for i, v in enumerate(value):
                    self.dump_meta_item(dst=dst, key=key+str(i), value=v)

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
                self.dump_meta_item(dst=dst, key=key+'values', value=values_)
            elif isinstance(value, np.ndarray):
                dst.create_dataset(key.strip('/'), data=value,
                                **hdf5plugin.Blosc(cname='lz4hc', clevel=2, shuffle=0))
            else:
                dst[key] = value

    # Loading
    def load_meta(self, path=None, names=None):
        """ !!. """
        meta_path = path or self.meta_path
        names = names or self.PRESERVED

        with self.META_OPENER(meta_path, mode='r') as meta_file:
            for name in names:
                value = self.load_meta_item(f'/meta/{name}', src=meta_file)
                setattr(self, name, value)
                if hasattr(self, 'meta_list_loaded'):
                    self.meta_list_loaded.add(name)

    def load_meta_item(self, key, src=None):
        """ !!. """
        key = key + '/'
        key = key.replace('//', '/')

        with src or self.META_OPENER(self.meta_path, mode='a') as src:
            if key + 'is_sequence' in src:
                length = src[key + 'length'][()]
                type_ = src[key + 'type'][()]

                value = [self.load_meta_item(key=key + str(i), src=src) for i in range(length)]

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
                raise KeyError(f'No key `{key}` in {src}!')
            return value


    # Reset
    def reset_meta(self, path=None, names=None):
        """ !!. """
        meta_path = path or self.meta_path
        names = names or (self.PRESERVED + self.PRESERVED_LAZY)

        with self.META_OPENER(meta_path, mode='a') as meta_file:
            for name in names:
                if name in meta_file:
                    del meta_file[name]

                name_ = f'/meta/{name}'
                if name_ in meta_file:
                    del meta_file[name_]

    # Introspection
    def print_meta_tree(self, path=None):
        """ !!. """
        meta_path = path or self.meta_path
        self.META_OPENER(meta_path, mode='r').visit(self._print_meta_tree)

    def _print_meta_tree(self, name):
        """ !!. """
        shift = name.count('/') * ' ' * 4
        item_name = name.split('/')[-1]
        print(shift + item_name)
