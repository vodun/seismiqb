""" Thread-safe lru cache class and cache mixin. """
import os
from copy import copy
from functools import wraps, cached_property
from hashlib import blake2b
from threading import RLock
from collections import OrderedDict, defaultdict

import numpy as np
import pandas as pd



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

    Notes
    -----
    All arguments to a decorated method must be hashable.
    """
    #pylint: disable=invalid-name, attribute-defined-outside-init
    def __init__(self, maxsize=None, attributes=None, apply_by_default=True, copy_on_return=False):
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
            self.cache[instance] = OrderedDict()
            self.is_full[instance] = False
            self.stats[instance] = {'hit': 0, 'miss': 0}

    def make_key(self, instance, args, kwargs):
        """ Create a key from a combination of instance reference, method args, and instance attributes. """
        key = [instance] + list(args)
        if kwargs:
            for k, v in sorted(kwargs.items()):
                key.append((k, v))

        if self.attributes:
            for attr in self.attributes:
                attr_hash = stable_hash(getattr(instance, attr))
                key.append(attr_hash)
        return flatten_nested(key)


    def __call__(self, func):
        """ Add the cache to the function. """
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            use_cache = kwargs.pop('use_cache', self.apply_by_default)
            copy_on_return = kwargs.pop('copy_on_return', self.copy_on_return)

            if os.getenv('SEISMIQB_DISABLE_CACHE', ""):
                use_cache = False

            # Skip the caching logic and evaluate function directly
            if not use_cache:
                result = func(instance, *args, **kwargs)
                return result

            key = self.make_key(instance, args, kwargs)

            # If result is already in cache, just retrieve it and update its timings
            with self.lock:
                result = self.cache[instance].get(key, self.default)
                if result is not self.default:
                    del self.cache[instance][key]
                    self.cache[instance][key] = result
                    self.stats[instance]['hit'] += 1
                    return copy(result) if copy_on_return else result

            # The result was not found in cache: evaluate function
            result = func(instance, *args, **kwargs)

            # Add the result to cache
            with self.lock:
                self.stats[instance]['miss'] += 1
                if key in self.cache[instance]:
                    pass
                elif self.is_full[instance]:
                    self.cache[instance].popitem(last=False)
                    self.cache[instance][key] = result
                else:
                    self.cache[instance][key] = result
                    self.is_full[instance] = (len(self.cache[instance]) >= self.maxsize)
            return copy(result) if copy_on_return else result

        wrapper.__name__ = func.__name__
        wrapper.cache = lambda: self.cache
        wrapper.stats = lambda: self.stats
        wrapper.reset = self.reset
        wrapper.reset_instance = lambda instance: self.reset(instance=instance)
        wrapper.cache_instance = self
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

    You can use this mixin for cache introspection and clearing cached data.
    """
    #pylint: disable=redefined-builtin
    def get_cached_objects(self, properties, methods):
        """ Get a list of methods and properties that use caching."""
        if properties is False:
            get_properties = False
            properties = []
        elif properties is True:
            get_properties = True
            properties = []
        else:
            get_properties = False

        if methods is False:
            get_methods = False
            methods = []
        elif methods is True:
            get_methods = True
            methods = []
        else:
            get_methods = False

        if not get_properties and not get_methods:
            return properties, methods

        class_ = self.__class__
        for name in dir(self):
            if name.startswith("__"):
                continue

            class_obj = getattr(class_, name, None)
            if isinstance(class_obj, property):
                continue

            if get_properties and isinstance(class_obj, cached_property):
                properties.append(name)
                continue

            instance_obj = getattr(self, name)
            if get_methods and callable(instance_obj) and hasattr(instance_obj, 'cache'):
                methods.append(instance_obj)

        return properties, methods

    def reset_cache(self, properties=True, methods=True):
        """ Clear cached data.

        Parameters:
        ----------
        properties: bool or list
            If a bool, indicates whether every cached class property must be reset.
            If a list, must contain names of cached properties to reset.
            By default, reset all cached properties of the class.

        methods: bool or list
            If a bool, indicates whether every cached class method must be reset.
            If a list, must contain names of cached methods to reset.
            By default, reset all cached methods of the class.
        """
        reset_properties, reset_methods = self.get_cached_objects(properties, methods)

        for property_ in reset_properties:
            print(property_)
            delattr(self, property_)

        for method in reset_methods:
            method.reset(instance=self)

    def get_cache_length(self, properties=True, methods=True):
        """ Clear cached data.

        Parameters:
        ----------
        properties: bool or list
            If a bool, indicates whether every cached class property must be accounted.
            If a list, must contain names of cached properties to account.
            By default, account all cached properties of the class.

        methods: bool or list
            If a bool, indicates whether every cached class method must be accounted.
            If a list, must contain names of cached methods to account.
            By default, account all cached methods of the class.
        """
        cached_properties, cached_methods = self.get_cached_objects(properties, methods)

        cache_length_accumulator = 0

        for property_ in cached_properties:
            if property_ in self.__dict__:
                cache_length_accumulator += 1

        for method in cached_methods:
            cache_length_accumulator += len(method.cache()[self])

        return cache_length_accumulator

    def get_cache_size(self, properties=True, methods=True):
        """ Total size of cached objects for specified methods.

        Parameters:
        ----------
        properties: bool or list
            If a bool, indicates whether every cached class property must be accounted.
            If a list, must contain names of cached properties to account.
            By default, account all cached properties of the class.

        methods: bool or list
            If a bool, indicates whether every cached class method must be accounted.
            If a list, must contain names of cached methods to account.
            By default, account all cached methods of the class.
        """
        cached_properties, cached_methods = self.get_cached_objects(properties, methods)

        cache_size_accumulator = 0

        # Accumulate cache size over all cached methods and properties
        # Each term is a size of cached numpy array
        for property_ in cached_properties:
            property_value = getattr(self, property_)

            if isinstance(property_value, np.ndarray):
                cache_size_accumulator += property_value.nbytes / (1024 ** 3)

        for method in cached_methods:
            method_values = list(method.cache()[self].values())

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

    def make_cache_repr(self, format='dict'):
        """ Cache representation that consists of names of methods that cache data,
        information about cache length, size, and arguments for each method.

        Parameters:
        ----------
        format : str
            Return value format. Can be 'dict' or 'df'. 'df' means pandas DataFrame.
        """
        cached_properties, cached_methods = self.get_cached_objects(properties=True, methods=True)

        cache_repr_ = {}

        # Creation of a dictionary of cache representation for each method and property
        # with cache_length, cache_size and arguments
        for property_ in cached_properties:
            property_cache_lenth = self.get_cache_length(properties=[property_], methods=False)

            if property_cache_lenth:
                property_cache_size = self.get_cache_size(properties=[property_], methods=False)

                cache_repr_[property_] = {
                    'cache_length': property_cache_lenth,
                    'cache_size': property_cache_size,
                    'arguments': None
                }

        for method in cached_methods:
            method_cache_length = self.get_cache_length(properties=False, methods=[method])

            if method_cache_length:
                method_cache_size = self.get_cache_size(properties=False, methods=[method])

                arguments = list(method.cache()[self].keys())[0][1:]
                arguments = dict(zip(arguments[::2], arguments[1::2]))

                cache_repr_[method.__name__] = {
                    'cache_length': method_cache_length,
                    'cache_size': method_cache_size,
                    'arguments': arguments
                }

        # Convertation to pandas dataframe
        if format == 'df':
            cache_repr_ = pd.DataFrame.from_dict(cache_repr_, orient='index')

            # Columns sort
            if len(cache_repr_) > 0:
                cache_repr_ = cache_repr_.loc[:, ['cache_length', 'cache_size', 'arguments']]

        return cache_repr_

    @property
    def cache_repr(self):
        """ DataFrame with cache representation that contains of names, cache_length
        and cache_size for each cached method.
        """
        df = self.make_cache_repr(format='df')

        if len(df) > 0:
            return df.loc[:, ['cache_length', 'cache_size']]
        return {}
