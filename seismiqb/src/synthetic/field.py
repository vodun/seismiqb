""" Placeholder for a synthetic field. """
import numpy as np

from .generator import SyntheticGenerator
from ..utils import lru_cache

class GeometryMock:
    def __getattr__(self, _):
        return None


class SyntheticField:
    """ A wrapper around `SyntheticGenerator` to provide the same API, as regular `:class:.Field`.

    The intended use of this class is:
        - define a `param_generator` function, that returns a dictionary with parameters of seismic generation.
        The parameters may be randomized, so the generated data is different each time.
        - initialize an instance of this class.
        - use `get_attribute` method to get synthetic images, horizon/fault masks, impedance models.
        In order to ensure that they match one another, supply the same `location` (triplet of slices), for example:
        >>> location = (slice(10, 11), slice(100, 200), slice(1000, 1500))
        >>> synthetic = synthetic_field.get_attribute(location=location, attribute='synthetic')
        >>> impedance = synthetic_field.get_attribute(location=location, attribute='impedance')
        would make synthetic and impedance images for the same underlying data.

    Methods `load_seismic` and `make_masks` are thin wrappers around `get_attribute` to make API of this class
    identical to that of `:class:.Field`. Despite that, we don't use abstract classes or other ways to enforce it.

    Under the hood, we keep track of internal cache (with `location` as key) to use the same instance of generator
    multiple times. The size of cache is parametrized at initialization and should be bigger than the batch size.
    Other than that, the `location` is also used to infer shape of requested synthetic data,
    if it is not provided at initialization / from `param_generator`.

    Parameters
    ----------
    param_generator : callable, optional
        If provided, should return a dictionary with parameters to generate synthetic.
        Refer to `:meth:.default_param_generator` for example of implementation.
        Can be omitted if the `data_generator` is supplied instead.
    data_generator : callable, optional
        If provided, then a callable to populate an instance of `SyntheticGenerator` with data.
        Should take `generator` as the only required argument. Disables the `param_generator` option.
        Note that the logic of keep
    attribute : str
        Attribute to get from the generator if `labels` are requested.
    crop_shape : tuple of int
        Default shape of the generated synthetic images.
        If not provided, we use the shape from `param_generator` or `location`.
    name : str
        Name of the the field. Used to comply with `:class:.Field` API.
    cache_maxsize : int
        Number of cached generators. Should be equal or bigger than the batch size.
    """
    #pylint: disable=method-hidden, protected-access
    def __init__(self, param_generator=None, data_generator=None, attribute=None, crop_shape=(256, 256),
                 name='synthetic_field', cache_maxsize=128):
        #
        self.attribute = attribute
        self.crop_shape = crop_shape
        self.param_generator = param_generator if param_generator is not None else self.default_param_generator
        self.data_generator = data_generator
        self._make_generator = lru_cache(maxsize=cache_maxsize)(self._make_generator)

        # String info
        self.path = self.short_path = f'{name}_path'
        self.name = self.short_name = self.displayed_name = name
        self.index_headers = self.axis_names = ['INLINE_3D', 'CROSSLINE_3D']

        # Attributes to comply with `:class:.Field` API
        self.geometry = GeometryMock()
        self.spatial_shape = (-1, -1)
        self.shape = self.cube_shape = (-1, -1, -1)
        self.ilines_len = self.xlines_len = self.depth = -1
        self.zero_traces = self.mean_matrix = self.std_matrix = None

        # Properties
        self._normalization_stats = None

    @property
    def labels(self):
        """ Property for sampler creation. Used as a signal that this field is in fact synthetic. """
        return self

    # Generator creation
    def get_generator(self, location=None, shape=None):
        """ Get a generator with data of a given `shape`.
        If called with the same parameters twice, returns the same instance: `location` is used as a hash value.
        """
        if location is not None:
            hash_value = hash(tuple((slc.start, slc.stop, slc.step) for slc in location))
        else:
            hash_value = hash(np.random.randint(0, 1000000))

        generator = self._make_generator(hash_value)
        self._populate_generator(generator=generator, location=location, shape=shape) # Works in-place!
        return generator

    # @lru_cache
    def _make_generator(self, hash_value):
        """ Create a generator instance. During initialization, wrapped in `lru_cache`. """
        return SyntheticGenerator()

    def _populate_generator(self, generator, location=None, shape=None):
        """ Call `generator` methods to populate it with data: impedance model, horizon surfaces, faults, etc. """
        if hasattr(generator, '_populated'):
            return None

        if self.data_generator is not None:
            # Change `generator` in-place!
            self.data_generator(generator)

        else:
            # Generate parameters, use them to populate `generator` in-place
            params = self.param_generator()

            if shape is None:
                shape = params.get('shape', self.crop_shape)
                if shape is None and location is not None:
                    shape = tuple(slc.stop - slc.start for slc in location)

            shape = tuple(s for s in shape if s != 1)

            (generator
             .make_velocities(num_reflections=params['num_reflections'],
                              horizon_heights=params['horizon_heights'],
                              horizon_multipliers=params['horizon_multipliers'],
                              velocity_limits=params['velocity_limits'])
             .make_velocity_model(shape=shape, grid_shape=params['grid_shape'])
             .make_density_model(density_noise_lims=params['density_noise_lims'])
             .make_reflectivity()
             .make_synthetic(ricker_width=params['ricker_width'],
                             ricker_points=params['ricker_points'])
             .postprocess_synthetic(noise_mul=params['noise_mul'])
            )
            generator.params = params

        generator._populated = True
        return None


    @staticmethod
    def default_param_generator(seed=None):
        """ Sample parameters for synthetic generation. """
        rng = np.random.default_rng(seed)
        num_horizons = rng.integers(low=2, high=7, endpoint=True)

        return {
            'velocity_limits': (2000, 6000),

            # Horizons
            'num_reflections': rng.integers(low=15, high=30, endpoint=True),
            'horizon_heights': np.sort(rng.uniform(low=.15, high=.95, size=num_horizons)),
            'horizon_multipliers': (rng.choice([-1, 1], size=num_horizons) *
                                    rng.uniform(4, 9, size=num_horizons)),
            # Faults

            # Impedance creation
            'grid_shape': rng.integers(low=5, high=10, size=(1,)),
            'density_noise_lims': rng.uniform(low=(0.9, 1.0), high=(1.0, 1.1)),

            # Conversion to seismic
            'ricker_width': rng.uniform(low=3.5, high=5.1),
            'ricker_points': rng.integers(low=50, high=130, endpoint=True),
            'noise_mul': rng.uniform(low=0.1, high=0.3),
        }


    # Getting data
    def get_attribute(self, location=None, shape=None, attribute='synthetic', **kwargs):
        """ Output requested `attribute`. If `location` is not provided, generates a new instance each time.
        For the same `location` values, uses the same generator instance (and the same reflectivity model):
        >>> location = (slice(10, 11), slice(100, 200), slice(1000, 1500))
        >>> synthetic = synthetic_field.get_attribute(location=location, attribute='synthetic')
        >>> impedance = synthetic_field.get_attribute(location=location, attribute='impedance')
        """
        _ = kwargs
        generator = self.get_generator(location=location, shape=shape)

        if attribute == 'labels':
            attribute = generator.params.get('attribute', self.attribute)

        if attribute in ['synthetic', 'geometry', 'image']:
            result = generator.synthetic
        elif 'upward' in attribute:
            generator.make_upward_velocities().make_upward_velocity_model()
            result = getattr(generator, attribute)
        elif 'reflections' in attribute:
            result = generator.fetch_horizons(mode=slice(1, None, 1), horizon_format='mask',
                                              width=kwargs.get('width', 3))
        elif 'reflect' in attribute:
            result = generator.reflectivity_coefficients
        elif 'horizon' in attribute:
            result = generator.fetch_horizons(mode='horizons', horizon_format='mask',
                                              width=kwargs.get('width', 3))
        elif 'impedance' in attribute:
            result = generator.velocity_model
        else:
            result = getattr(generator, attribute)

        if result.dtype != np.float32:
            result = result.astype(np.float32)

        if location is not None:
            shape = tuple(slc.stop - slc.start for slc in location)
        return result.reshape(shape)


    def load_seismic(self, location=None, shape=None, src='synthetic', **kwargs):
        """ Wrapper around `:meth:.get_attribute` to comply with `:class:.Field` API. """
        return self.get_attribute(location=location, shape=shape, attribute=src, **kwargs)

    def make_mask(self, location=None, shape=None, src='labels', **kwargs):
        """ Wrapper around `:meth:.get_attribute` to comply with `:class:.Field` API. """
        return self.get_attribute(location=location, shape=shape, attribute=src, **kwargs)

    def load_slide(self):
        """ !!. """
        pass


    # Normalization
    def make_normalization_stats(self, n=1000, shape=None, attribute='synthetic'):
        """ Compute normalization stats (`mean`, `std`, `min`, `max`, quantiles) from `n` generated `attributes`. """
        data = [self.get_attribute(shape=shape or self.crop_shape, attribute=attribute) for _ in range(n)]
        data = np.array(data)

        q01, q05, q95, q99 = np.quantile(data, (0.01, 0.05, 0.95, 0.99))

        normalization_stats = {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data),
            'q_01': q01,
            'q_05': q05,
            'q_95': q95,
            'q_99': q99,
        }
        self._normalization_stats = normalization_stats
        return normalization_stats

    @property
    def normalization_stats(self):
        """ Property with default normalization stats for synthetic images. """
        if self._normalization_stats is None:
            self.make_normalization_stats()
        return self._normalization_stats

    # Vis
    def __repr__(self):
        return f"""<SyntheticField `{self.displayed_name}` at {hex(id(self))}>"""

    def __str__(self):
        msg = f"SyntheticField `{self.displayed_name}`"

        attribute = self.attribute
        if attribute is None and self.param_generator is not None:
            attribute = self.param_generator.get('attribute')
        if attribute is not None:
            msg += f':\n    - labels: attribute `{attribute}`'
        return msg

    def show_slide(self):
        """ !!. """
        pass
