""" Thread-safe lru cache class and cache mixin. """
import os
from copy import copy
from functools import wraps
from functools import cached_property as functools_cached_property
from hashlib import blake2b
from inspect import ismethod
from threading import RLock
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd


class _GlobalCacheClass:
    """ Global cache representation and introspection.

    Note, the container saves only info for objects which use :class:`~.lru_cache` and :class:`~.cached_property`.

    TODO: proper removal instances control

    cache_instances = {
        'methods': {'name': cache_instance, ...},
        'properties': {'name': {owner_instance, ...}, ...}
    }
    """
    def __init__(self):
        self.cache_instances = {'properties': {}, 'methods': {}} # Evaluated on the modules init

    @staticmethod
    def _get_method_length(cache_instance):
        """ .. """
        cache_length = 0
        for cache in cache_instance.cache.values():
            cache_length += len(cache)
        return cache_length

    @staticmethod
    def _get_property_length(name, owners):
        """ .. """
        cache_length = 0
        for owner in owners:
            if name in owner.__dict__:
                cache_length += 1
        return cache_length

    @property
    def length(self):
        """ Total cache length. """
        cache_length = 0
        for cache_instance in self.cache_instances['methods'].values():
            cache_length += _GlobalCacheClass._get_method_length(cache_instance=cache_instance)

        for name, owners in self.cache_instances['properties'].items():
            cache_length += _GlobalCacheClass._get_property_length(name=name, owners=owners)

        return cache_length

    @staticmethod
    def _get_method_size(cache_instance):
        """ .. """
        cache_size = 0
        for cache in cache_instance.cache.values():
            for value in cache.values():
                if isinstance(value, np.ndarray):
                    cache_size += value.nbytes / (1024 ** 3)
        return cache_size

    @staticmethod
    def _get_property_size(name, owners):
        """ .. """
        cache_size = 0
        for owner in owners:
            if name in owner.__dict__:
                property_value = getattr(owner, name, None)
                if isinstance(property_value, np.ndarray):
                    cache_size += property_value.nbytes / (1024 ** 3)
        return cache_size

    @property
    def size(self):
        """ Total cache size. """
        cache_size = 0
        for cache_instance in self.cache_instances['methods'].values():
            cache_size += _GlobalCacheClass._get_method_size(cache_instance=cache_instance)

        for name, owners in self.cache_instances['properties'].items():
            cache_size += _GlobalCacheClass._get_property_size(name=name, owners=owners)

        return cache_size

    def get_object_cache_repr(self, object_name, object_type=None):
        """ .. """
        if object_type not in ('method', 'property'):
            if object_name in self.cache_instances['methods'].keys():
                object_type = 'method'
            elif object_name in self.cache_instances['properties'].keys():
                object_type = 'property'
            else:
                raise ValueError("Invalid `object_name`: cache doesn't contain it.")

        if object_type == 'method':
            cache_instance = self.cache_instances['methods'][object_name]

            length = _GlobalCacheClass._get_method_length(cache_instance=cache_instance)
            if length > 0:
                size = _GlobalCacheClass._get_method_size(cache_instance=cache_instance)

        else:
            owners = self.cache_instances['properties'][object_name]

            length = _GlobalCacheClass._get_property_length(name=object_name, owners=owners)
            if length > 0:
                size = _GlobalCacheClass._get_property_size(name=object_name, owners=owners)

        cache_repr_ = {'length': length, 'size': size} if length > 0 else None
        return cache_repr_

    def get_cache_repr(self, format='dict'):
        """ .. """
        cache_repr_ = {}

        for method_name in self.cache_instances['methods'].keys():
            object_cache_repr = self.get_object_cache_repr(object_name=method_name, object_type='method')
            if object_cache_repr is not None:
                cache_repr_[method_name] = object_cache_repr

        for property_name in self.cache_instances['properties'].keys():
            object_cache_repr = self.get_object_cache_repr(object_name=property_name, object_type='property')
            if object_cache_repr is not None:
                cache_repr_[property_name] = object_cache_repr

        # Conversion to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            cache_repr_ = pd.DataFrame.from_dict(cache_repr_, orient='index')
            cache_repr_ = cache_repr_.loc[:, ['length', 'size']] # Columns sort
        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def repr(self):
        """ .. !!.."""
        return self.get_cache_repr(format='df')

    def reset(self):
        """ .. !!.."""
        for cache_instance in self.cache_instances['methods'].values():
            cache_instance.reset()

        for property_name, owners in self.cache_instances['properties'].items():
            for owner in owners:
                if property_name in owner.__dict__:
                    delattr(owner, property_name)

GlobalCache = _GlobalCacheClass() # No sense in multiple instances, use this for cache introspection and reset


class cached_property:
    """ ..  modification of functools.cached_property for GlobalCache """
    def __init__(self, func):
        """ .. save and decorate """
        self.name = func.__name__
        self.functools_func = functools_cached_property(func)

        GlobalCache.cache_instances['properties'][self.name] = set()

    def __set_name__(self, owner, name):
        """ .. __set_name__ call for correct implementation"""
        self.functools_func.__set_name__(owner, name)

    def __get__(self, instance, owner=None):
        """ .. save and get value"""
        GlobalCache.cache_instances['properties'][self.name].add(instance)
        return self.functools_func.__get__(instance, owner)


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
            instance_hash = self.compute_hash(instance)

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

        GlobalCache.cache_instances['methods'][func.__name__] = self
        return wrapper

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

    You can use this mixin for cache introspection and cached data cleaning.
    """
    #pylint: disable=redefined-builtin
    @property
    def cached_objects(self):
        """ All properties and methods names that use caching. """
        if getattr(self.__class__, '_cached_objects', None) is None:
            cached_properties = [obj for obj in GlobalCache.cache_instances['properties'].keys() if hasattr(self, obj)]
            cached_methods = [obj for obj in GlobalCache.cache_instances['methods'].keys() if hasattr(self, obj)]

            self.__class__._cached_objects = {'properties': cached_properties,
                                              'methods': cached_methods}
        return self.__class__._cached_objects

    def _parse_name(self, name=None):
        """ Map attribute name to its type (property or method). """
        if name is not None:
            properties = (name,) if name in self.cached_objects['properties'] else ()
            methods = (name,) if name in self.cached_objects['methods'] else ()
        else:
            properties = self.cached_objects['properties']
            methods = self.cached_objects['methods']
        return properties, methods

    def reset_cache(self, name=None):
        """ Clear cached data.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then reset cache of all cached objects.
        """
        reset_properties, reset_methods = self._parse_name(name)

        for property_name in reset_properties:
            if property_name in self.__dict__:
                delattr(self, property_name)

        for method_name in reset_methods:
            getattr(self, method_name).reset(instance=self)

    def get_cache_length(self, name=None):
        """ Get total amount of cached objects for specified properties and methods.

        Parameters
        ----------
        name: str, optional
            Attribute name. If None, then get total cache length.
        """
        cached_properties, cached_methods = self._parse_name(name)

        cache_length_accumulator = 0

        for property_name in cached_properties:
            if property_name in self.__dict__:
                cache_length_accumulator += 1

        for method_name in cached_methods:
            method_cache = getattr(self, method_name).cache()
            cache_length_accumulator += len(method_cache[lru_cache.compute_hash(self)])

        return cache_length_accumulator

    def get_cache_size(self, name=None):
        """ Get total size of cached objects for specified properties and methods.

        Parameters:
        ----------
        name: str, optional
            Attribute name. If None, then get total cache size.
        """
        cached_properties, cached_methods = self._parse_name(name)

        cache_size_accumulator = 0

        # Accumulate cache size over all cached methods and properties
        # Each term is a size of cached numpy array
        for property_name in cached_properties:
            if property_name in self.__dict__:
                property_value = getattr(self, property_name)

                if isinstance(property_value, np.ndarray):
                    cache_size_accumulator += property_value.nbytes / (1024 ** 3)

        for method_name in cached_methods:
            method_cache = getattr(self, method_name).cache()
            method_values = list(method_cache[lru_cache.compute_hash(self)].values())

            for values in method_values:
                if isinstance(values, np.ndarray):
                    cache_size_accumulator += values.nbytes / (1024 ** 3)

        return cache_size_accumulator

    @property
    def cache_length(self):
        """ Total amount of cached objects. """
        return self.get_cache_length()

    @property
    def cache_size(self):
        """ Total size of cached objects. """
        return self.get_cache_size()

    def _get_object_cache_repr(self, object_name, object_type):
        """ Make repr of object's cache if its length is nonzero else return None. """
        object_cache_length = self.get_cache_length(name=object_name)
        if object_cache_length == 0:
            return None

        object_cache_size = self.get_cache_size(name=object_name)

        if object_type == 'property':
            arguments = None
        elif object_type == 'method':
            method_cache = getattr(self, object_name).cache()
            arguments = list(method_cache[lru_cache.compute_hash(self)].keys())[0][1:]
            arguments = dict(zip(arguments[::2], arguments[1::2]))

        object_cache_repr = {
            'length': object_cache_length,
            'size': object_cache_size,
            'arguments': arguments
        }

        return object_cache_repr

    def get_cache_repr(self, format='dict'):
        """ Cache representation that consists of names of methods that cache data,
        information about cache length, size, and arguments for each method.

        Parameters:
        ----------
        format : {'dict', 'df'}
            Return value format. 'df' means pandas DataFrame.
        """
        cache_repr_ = {}

        # Creation of a dictionary of cache representation for each method and property
        # with cache_length, cache_size and arguments
        for property_name in self.cached_objects['properties']:
            property_cache_repr = self._get_object_cache_repr(object_name=property_name, object_type='property')
            if property_cache_repr is not None:
                cache_repr_[property_name] = property_cache_repr

        for method_name in self.cached_objects['methods']:
            method_cache_repr = self._get_object_cache_repr(object_name=method_name, object_type='method')
            if method_cache_repr is not None:
                cache_repr_[method_name] = method_cache_repr

        # Conversion to pandas dataframe
        if format == 'df' and len(cache_repr_) > 0:
            cache_repr_ = pd.DataFrame.from_dict(cache_repr_, orient='index')
            cache_repr_ = cache_repr_.loc[:, ['length', 'size', 'arguments']] # Columns sort

        return cache_repr_ if len(cache_repr_) > 0 else None

    @property
    def cache_repr(self):
        """ DataFrame with cache representation that contains of names, cache_length
        and cache_size for each cached method.
        """
        df = self.get_cache_repr(format='df')
        if df is not None:
            df = df.loc[:, ['length', 'size']]
        return df