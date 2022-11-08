""" Thread-safe lru cache class and cache mixin. """
import os
from copy import copy
from functools import wraps
from hashlib import blake2b
from inspect import ismethod
from threading import RLock
from collections import OrderedDict, defaultdict
from weakref import WeakSet

import numpy as np
import pandas as pd


class _GlobalCacheClass:
    """ Methods for global cache management.

    Note, it controls only objects which use :class:`~.lru_cache` and :class:`~.cached_property`.
    """
    def __init__(self):
        """ Initialize containers with cache references and instances with cached objects.

        Note, the `cache_references` container is filled on the modules import stage."""
        self.cache_references = {}
        self.instances_with_cache = WeakSet()

    @property
    def length(self):
        """ Total cache length. """
        cache_length = 0

        for instance in self.instances_with_cache:
            cache_length += instance.cache_length

        return cache_length

    @property
    def size(self):
        """ Total cache size. """
        cache_size = 0

        for instance in self.instances_with_cache:
            cache_size += instance.cache_size

        return cache_size

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

        for instance in self.instances_with_cache:
            instance_cache_repr = instance.get_cache_repr()

            if instance_cache_repr is not None:
                if instance.__class__ not in cache_repr_:
                    cache_repr_[instance.__class__.__name__] = {}

                cache_repr_[instance.__class__.__name__][id(instance)] = instance_cache_repr

        # Conversion to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            # Dataframe index columns are (class_name, instance_id, method_name), expand values for them:
            cache_repr_ = pd.DataFrame.from_dict({
                (class_name, instance_id, method_name): cache_repr_[class_name][instance_id][method_name]
                    for class_name in cache_repr_.keys() 
                    for instance_id in cache_repr_[class_name].keys()
                    for method_name in cache_repr_[class_name][instance_id].keys()},
            orient='index')

            cache_repr_ = cache_repr_.loc[:, ['length', 'size']] # Columns sort
        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def repr(self):
        """ Global cache representation"""
        return self.get_cache_repr(format='df')

    def reset(self):
        """ Clear all cache. """
        for instance in self.instances_with_cache:
            instance.reset_cache()

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
            self.stats = defaultdict(lambda: {'hit': 0, 'miss': 0})
        else:
            if hasattr(instance, self.attrname):
                delattr(instance, self.attrname)

            instance_hash = self.compute_hash(instance)
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

            # Init cache and reference on it in the GlobalCache controller
            if self.attrname not in instance.__dict__:
                instance.__setattr__(self.attrname, OrderedDict())

            GlobalCache.instances_with_cache.add(instance)

            key = self.make_key(instance, args, kwargs)
            instance_hash = getattr(instance, '_hash', self.compute_hash(instance))

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                cache = instance.__dict__[self.attrname]
                result = cache.get(key, self.default)

                if result is not self.default:
                    del cache[key]
                    cache[key] = result
                    self.stats[instance_hash]['hit'] += 1
                    return copy(result) if copy_on_return else result

            # The result was not found in cache: evaluate function
            result = func(*args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats[instance_hash]['miss'] += 1

                if key in cache:
                    pass
                elif len(cache) >= self.maxsize:
                    cache.popitem(last=False)
                    cache[key] = result
                else:
                    cache[key] = result

            return copy(result) if copy_on_return else result

        self.attrname = '_cache_' + func.__name__ # name of the attribute, which store the cache

        wrapper.__name__ = func.__name__
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        wrapper.reset_instance = lambda instance: self.reset(instance=instance)

        GlobalCache.cache_references[func.__qualname__] = self
        return wrapper

class cached_property:
    """ Mock for using :class:`~.lru_cache` for properties. """
    def __init__(self, func):
        self.cached_func = lru_cache(maxsize=1)(func)

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
    @property
    def cached_objects(self):
        """ All object names that use caching. """
        if not hasattr(self.__class__, '_cached_objects'):
            class_dir = dir(self.__class__)
            cached_objects = [obj_qualname for obj_qualname in GlobalCache.cache_references.keys()
                              if obj_qualname.split('.')[-1] in class_dir]

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
            cache_attrname = '_cache_' + name.split('.')[-1]
            cache = self.__dict__.get(cache_attrname, ())

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
            cache_attrname = '_cache_' + name.split('.')[-1]
            cache = self.__dict__.get(cache_attrname, {})
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

        cache_attrname = '_cache_' + name.split('.')[-1]
        cache = self.__dict__.get(cache_attrname, {})

        # The class saves cache for the same method with different arguments values
        # Get them all in a desired format: list of dicts
        all_arguments = []
        for arguments in cache.keys():
            arguments = dict(zip(arguments[::2], arguments[1::2])) # tuple ('name', value, ...) to dict
            all_arguments.append(arguments)

        # Expand extra scopes
        if len(all_arguments) == 1:
            all_arguments = all_arguments[0]

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
            cache_attrname = '_cache_' + name.split('.')[-1]
            if hasattr(self, cache_attrname):
                delattr(self, cache_attrname)
