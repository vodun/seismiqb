""" Thread-safe lru cache class and cache mixin. """
import os
from copy import copy
from functools import wraps
from hashlib import blake2b
from inspect import ismethod
from threading import RLock
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd


class _GlobalCacheClass:
    """ Methods for global cache management.

    Note, it controls only objects which use :class:`~.lru_cache` and :class:`~.cached_property`.
    """
    def __init__(self):
        """ Initialize container with objects and their cache references.

        Note, the container is filled on the modules import stage."""
        self.cache_references = {}

    @staticmethod
    def _get_cache_length(cache_instance):
        """ Evaluate cache length for the specific object. """
        cache_length = 0

        for cache in cache_instance.cache.values():
            cache_length += len(cache)

        return cache_length

    @staticmethod
    def _get_cache_size(cache_instance):
        """ Evaluate cache size for the specific object. """
        cache_size = 0

        for cache in cache_instance.cache.values():
            for value in cache.values():
                if isinstance(value, np.ndarray):
                    cache_size += value.nbytes / (1024 ** 3)

        return cache_size

    @property
    def length(self):
        """ Total cache length. """
        cache_length = 0

        for cache_instance in self.cache_references.values():
            cache_length += self._get_cache_length(cache_instance=cache_instance)

        return cache_length

    @property
    def size(self):
        """ Total cache size. """
        cache_size = 0

        for cache_instance in self.cache_references.values():
            cache_size += self._get_cache_size(cache_instance=cache_instance)

        return cache_size

    def get_object_cache_repr(self, object_name):
        """ Create cache representation for the specific object. """
        cache_instance = self.cache_references[object_name]

        length = self._get_cache_length(cache_instance=cache_instance)
        if length > 0:
            size = self._get_cache_size(cache_instance=cache_instance)

        cache_repr_ = {'length': length, 'size': size} if length > 0 else None
        return cache_repr_

    def get_cache_repr(self, format='dict'):
        """ Create global cache representation.

        Cache representation consists of names of objects that use data caching,
        information about cache length, size, and arguments for each method.

        Parameters
        ----------
        format : {'dict', 'df'}
            Return value format. 'df' means pandas DataFrame.
        """
        cache_repr_ = {}

        for method_name in self.cache_references.keys():
            object_cache_repr = self.get_object_cache_repr(object_name=method_name)
            if object_cache_repr is not None:
                cache_repr_[method_name] = object_cache_repr

        # Conversion to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            cache_repr_ = pd.DataFrame.from_dict(cache_repr_, orient='index')
            cache_repr_ = cache_repr_.loc[:, ['length', 'size']] # Columns sort
        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def repr(self):
        """ Global cache representation"""
        return self.get_cache_repr(format='df')

    def reset(self):
        """ Clear all cache. """
        for cache_instance in self.cache_references.values():
            cache_instance.reset()

GlobalCache = _GlobalCacheClass() # No sense in multiple instances, use this for cache control

class lru_cache:
    """ Thread-safe least recent used cache. Must be applied to a class methods.
    Adds the `use_cache` argument to the decorated method to control whether the caching logic is applied.
    Stored values are individual for each instance of a class.

    Parameters
    ----------
    maxsize : int
        Maximum amount of stored values.
    attributes: None, str or sequence of str
        Attributes to get from object and use as additions to key.
    apply_by_default : bool
        Whether the cache logic is on by default.
    copy_on_return : bool
        Whether to copy the object on retrieving from cache.

    Examples
    --------
    Store loaded slides::

    @lru_cache(maxsize=128)
    def load_slide(cube_name, slide_no):
        pass

    Specify cache size on class instantiation::
    def __init__(self, maxsize):
        self.method = lru_cache(maxsize)(self.method)

    Notes
    -----
    All arguments to a decorated method must be hashable.
    """
    #pylint: disable=invalid-name, attribute-defined-outside-init
    def __init__(self, maxsize=128, attributes=None, apply_by_default=True, copy_on_return=False):
        self.maxsize = maxsize
        self.apply_by_default = apply_by_default
        self.copy_on_return = copy_on_return

        # Parse `attributes`
        if isinstance(attributes, str):
            self.attributes = [attributes]
        elif isinstance(attributes, (tuple, list)):
            self.attributes = attributes
        else:
            self.attributes = False

        self.default = Singleton
        self.lock = RLock()
        self.reset()

    def reset(self, instance=None):
        """ Clear cache and stats. """
        if instance is None:
            self.cache = defaultdict(OrderedDict)
            self.is_full = defaultdict(lambda: False)
            self.stats = defaultdict(lambda: {'hit': 0, 'miss': 0})
        else:
            instance_hash = self.compute_hash(instance)
            self.cache[instance_hash] = OrderedDict()
            self.is_full[instance_hash] = False
            self.stats[instance_hash] = {'hit': 0, 'miss': 0}

    def make_key(self, instance, args, kwargs):
        """ Create a key from a combination of method args and instance attributes. """
        key = list(args[1:] if args[0] is instance else args)
        if kwargs:
            for k, v in sorted(kwargs.items()):
                if isinstance(v, slice):
                    v = (v.start, v.stop, v.step)
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(instance, attr))
                key.append(attr_hash)
        return flatten_nested(key)

    @staticmethod
    def compute_hash(obj):
        """ Compute `obj` hash. If not provided by the object, rely on objects identity. """
        #pylint: disable=bare-except
        try:
            result = hash(obj)
        except:
            result = id(obj)
        return result

    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(*args, **kwargs):
            # if a bound method, get class instance from function else from arguments
            instance = func.__self__ if ismethod(func) else args[0]

            use_cache = kwargs.pop('use_cache', self.apply_by_default)
            copy_on_return = kwargs.pop('copy_on_return', self.copy_on_return)

            if os.getenv('SEISMIQB_DISABLE_CACHE', ""):
                use_cache = False

            # Skip the caching logic and evaluate function directly
            if not use_cache:
                result = func(*args, **kwargs)
                return result

            key = self.make_key(instance, args, kwargs)
            instance_hash = getattr(instance, '_hash', self.compute_hash(instance))

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                result = self.cache[instance_hash].get(key, self.default)
                if result is not self.default:
                    del self.cache[instance_hash][key]
                    self.cache[instance_hash][key] = result
                    self.stats[instance_hash]['hit'] += 1
                    return copy(result) if copy_on_return else result

            # The result was not found in cache: evaluate function
            result = func(*args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats[instance_hash]['miss'] += 1
                if key in self.cache[instance_hash]:
                    pass
                elif self.is_full[instance_hash]:
                    self.cache[instance_hash].popitem(last=False)
                    self.cache[instance_hash][key] = result
                else:
                    self.cache[instance_hash][key] = result
                    self.is_full[instance_hash] = (len(self.cache[instance_hash]) >= self.maxsize)
            return copy(result) if copy_on_return else result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        wrapper.reset_instance = lambda instance: self.reset(instance=instance)
        wrapper.cache_instance = self

        GlobalCache.cache_references[func.__qualname__] = self
        return wrapper

class cached_property:
    """ Mock for using :class:`~.lru_cache` for properties. """
    def __init__(self, func):
        self.cached_func = lru_cache()(func)

    def __get__(self, instance, owner=None):
        _ = owner
        return self.cached_func(instance)


class SingletonClass:
    """ There must be only one! """
Singleton = SingletonClass()


def stable_hash(key):
    """ Hash that stays the same between different runs of Python interpreter. """
    if not isinstance(key, (str, bytes)):
        key = ''.join(sorted(str(key)))
    if not isinstance(key, bytes):
        key = key.encode('ascii')
    return str(blake2b(key).hexdigest())

def flatten_nested(iterable):
    """ Recursively flatten nested structure of tuples, list and dicts. """
    result = []
    if isinstance(iterable, (tuple, list)):
        for item in iterable:
            result.extend(flatten_nested(item))
    elif isinstance(iterable, dict):
        for key, value in sorted(iterable.items()):
            result.extend((*flatten_nested(key), *flatten_nested(value)))
    else:
        return (iterable,)
    return tuple(result)


class CacheMixin:
    """ Methods for cache management.

    You can use this mixin for cache introspection and cached data cleaning on instance level.
    """
    #pylint: disable=redefined-builtin
    @property
    def _hash(self):
        """ Object hash value which is used for cache control. """
        if not hasattr(self, '_hash_value'):
            setattr(self, '_hash_value', lru_cache.compute_hash(self))
        return self._hash_value

    @property
    def cached_objects(self):
        """ All object names that use caching. """
        if not hasattr(self.__class__, '_cached_objects'):
            cached_objects = [obj_qualname for obj_qualname in GlobalCache.cache_references.keys()
                              if obj_qualname.split('.')[-1] in dir(self.__class__)]

            setattr(self.__class__, '_cached_objects', tuple(cached_objects))
        return self.__class__._cached_objects

    def get_cache_length(self, name=None):
        """ Get cache length for specified objects.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then get total cache length.
        """
        names = (name,) if name is not None else self.cached_objects
        cache_length_accumulator = 0

        for name in names:
            cache = GlobalCache.cache_references[name].cache.get(self._hash, ())
            cache_length_accumulator += len(cache)

        return cache_length_accumulator

    def get_cache_size(self, name=None):
        """ Get cache size for specified objects.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then get total cache size.
        """
        names = (name,) if name is not None else self.cached_objects
        cache_size_accumulator = 0

        # Accumulate cache size over all cached objects: each term is a size of cached numpy array
        for name in names:
            cache = GlobalCache.cache_references[name].cache.get(self._hash, {})
            values = list(cache.values())

            for value in values:
                if isinstance(value, np.ndarray):
                    cache_size_accumulator += value.nbytes / (1024 ** 3)

        return cache_size_accumulator

    @property
    def cache_length(self):
        """ Total amount of cached objects. """
        return self.get_cache_length()

    @property
    def cache_size(self):
        """ Total size of cached objects. """
        return self.get_cache_size()

    def _get_object_cache_repr(self, name):
        """ Make repr of object's cache if its length is nonzero else return None. """
        object_cache_length = self.get_cache_length(name=name)
        if object_cache_length == 0:
            return None

        object_cache_size = self.get_cache_size(name=name)

        cache = GlobalCache.cache_references[name].cache.get(self._hash, None)

        arguments = list(cache.keys())[0][1:]
        arguments = dict(zip(arguments[::2], arguments[1::2]))

        object_cache_repr = {
            'length': object_cache_length,
            'size': object_cache_size,
            'arguments': arguments
        }

        return object_cache_repr

    def get_cache_repr(self, format='dict'):
        """  Create instance cache representation.

        Cache representation consists of names of objects that use data caching,
        information about cache length, size, and arguments for each method.

        Parameters
        ----------
        format : {'dict', 'df'}
            Return value format. 'df' means pandas DataFrame.
        """
        cache_repr_ = {}

        # Creation dictionary of cache representation for each object
        # with cache_length, cache_size and arguments
        for name in self.cached_objects:
            object_cache_repr = self._get_object_cache_repr(name=name)
            if object_cache_repr is not None:
                cache_repr_[name] = object_cache_repr

        # Conversion to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            cache_repr_ = pd.DataFrame.from_dict(cache_repr_, orient='index')
            cache_repr_ = cache_repr_.loc[:, ['length', 'size', 'arguments']] # Columns sort

        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def cache_repr(self):
        """ DataFrame with cache representation that contains names, cache_length
        and cache_size for each cached object.
        """
        df = self.get_cache_repr(format='df')
        if df is not None:
            df = df.loc[:, ['length', 'size']]
        return df

    def reset_cache(self, name=None):
        """ Clear cached data.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then clean cache of all cached objects.
        """
        names = (name,) if name is not None else self.cached_objects
        for name in names:
            cache_reference = GlobalCache.cache_references[name]
            if self._hash in cache_reference.cache:
                cache_reference.reset()
