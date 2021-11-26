""" A cache mixin. """


class CacheMixin:
    """ Methods for cache management.

    You can use this mixin for introspection caching and cleaning cached data.
    """
    def get_cached_methods(self):
        """ Get a list of methods which use caching."""
        methods = []

        for name in dir(self):
            is_property = isinstance(getattr(self.__class__, name, None), property)
            if name.startswith("__") or 'cache' in name or is_property:
                continue

            method = getattr(self, name)

            if callable(method):
                if hasattr(method, 'cache'):
                    methods.append(method)

        return methods

    def reset_cache(self, cached_methods=None):
        """ Clear cached data.

        Parameters:
        ----------
        cached_methods: list
            A list of methods of `self` which cache data.
        """
        if cached_methods is None:
            cached_methods = self.get_cached_methods()

        for method in cached_methods:
            method.reset(instance=self)

    def get_cache_length(self, cached_methods=None):
        """ Total amount of cached objects for specified methods.

        Parameters:
        ----------
        cached_methods: list
            A list of methods of `self` which cache data.
        """
        if cached_methods is None:
            cached_methods = self.get_cached_methods()

        cache_length_accumulator = 0

        for method in cached_methods:
            cache_length_accumulator += len(method.cache()[self])

        return cache_length_accumulator

    def get_cache_size(self, cached_methods=None):
        """ Total size of cached objects for specified methods.

        Parameters:
        ----------
        cached_methods: list
            A list of methods of `self` which cache data.
        """
        if cached_methods is None:
            cached_methods = self.get_cached_methods()

        cache_size_accumulator = 0

        for method in cached_methods:
            cache_size_accumulator += sum(item.nbytes / (1024 ** 3) for item in method.cache()[self].values())

        return cache_size_accumulator

    @property
    def cache_length(self):
        """ Total amount of cached objects. """
        return self.get_cache_length()

    @property
    def cache_size(self):
        """ Total size of cached objects. """
        return self.get_cache_size()

    @property
    def cache_repr(self):
        """ Cache representation presented by a dictionary with names of methods with cached data as keys.
        Values are dictionaries with information about cache length, size and method arguments.
        """
        cached_methods = self.get_cached_methods()

        cache_repr_ = {}

        for method in cached_methods:
            method_cache_length = self.get_cache_length(cached_methods=[method])

            if method_cache_length:
                method_cache_size = self.get_cache_size(cached_methods=[method])

                arguments = list(method.cache()[self].keys())[0][1:]
                arguments = dict(zip(arguments[::2], arguments[1::2]))

                cache_repr_[method.__name__] = {
                    'cache_length': method_cache_length,
                    'cache_size': method_cache_size,
                    'arguments': arguments
                }

        return cache_repr_
