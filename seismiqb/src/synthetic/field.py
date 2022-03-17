""" Placeholder for a synthetic field. """
import numpy as np

from .generator import SyntheticGenerator
from ..utils import lru_cache
from ..plotters import plot_image
from ...batchflow import Config

class GeometryMock:
    """ Mock for SeismicGeometry. """
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
        would make synthetic and impedance images for the same underlying synthetic model.

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
        Note that the logic of keeping the same instance of `generator` for multiple calls with the same `location`
        is performed by class internals and still available in that case.
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
        # Data generation
        self.param_generator = param_generator if param_generator is not None else self.default_param_generator
        self.data_generator = data_generator
        self._make_generator = lru_cache(maxsize=cache_maxsize)(self._make_generator)
        self._cache_maxsize = cache_maxsize

        # Defaults for retrieving attributes
        self.attribute = attribute
        self.crop_shape = crop_shape

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
        return SyntheticGenerator(seed=abs(hash_value))

    def _populate_generator(self, generator, location=None, shape=None):
        """ Call `generator` methods to populate it with data: impedance model, horizon surfaces, faults, etc. """
        if hasattr(generator, '_populated'):
            return None

        if self.data_generator is not None:
            self.data_generator(generator)

        else:
            # Generate parameters, use them to populate `generator` in-place
            params = self.param_generator(rng=generator.rng)
            params = Config(params)

            # Parse shape: priority is `params['shape']` -> `self.crop_shape` -> `location.shape`
            if shape is None:
                if location is not None:
                    shape = tuple(slc.stop - slc.start for slc in location)
                elif 'shape' in params:
                    shape = params['shape']
                else:
                    shape = self.crop_shape

            # Compute velocity model, using the velocity vector and horizon matrices
            (generator
             .make_velocity_vector(**params['make_velocity_vector'])
             .make_horizons(shape=shape, **params['make_horizons'])
             .make_velocity_model(**params['make_velocity_model'])
             )

            # Faults
            for fault_params in params['make_fault_2d']:
                generator.make_fault_2d(**fault_params)

            # Finalize synthetic creation
            (generator
             .make_density_model(**params['make_density_model'])
             .make_impedance_model(**params['make_impedance_model'])
             .make_reflectivity_model(**params['make_reflectivity_model'])

             .make_synthetic(**params['make_synthetic'])
             .postprocess_synthetic(**params['postprocess_synthetic'])
             .cleanup(**params['cleanup'])
            )

            generator.params = params

        generator._populated = True
        return None


    @staticmethod
    def default_param_generator(rng=None):
        """ Sample parameters for synthetic generation. """
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        num_faults = rng.choice([0, 1, 2, 3], p=[0.7, 0.2, 0.1, 0.0])
        fault_params = [{'coordinates': 'random',
                         'max_shift': rng.uniform(20, 40),
                         'shift_sign': rng.choice([-1, 1]),
                         'width': rng.uniform(1.0, 4.0)}
                        for _ in range(num_faults)]

        return {
            'make_velocity_vector': {'num_horizons': 17,                                            # scales with depth
                                     'limits': (2000, 6000),
                                     'randomization_scale': 0.3,
                                     'amplify_probability': 0.2, 'amplify_range': (2.0, 4.0)},
            'make_horizons': {'interval_randomization': 'uniform',
                              'interval_randomization_scale': 0.5,
                              'interval_min': rng.uniform(0.4, 0.6),
                              'randomization1_scale': 0.25,
                              'randomization2_scale': 0.25, 'locs_n_range': (3, 5)},
            'make_velocity_model': {},

            'make_fault_2d': fault_params,

            'make_density_model': {'randomization_limits': (0.95, 1.05)},
            'make_impedance_model': {},
            'make_reflectivity_model': {},

            'make_synthetic': {'ricker_width': rng.uniform(4.7, 5.3),
                               'ricker_points': 100},
            'postprocess_synthetic': {'sigma': 0.5,
                                      'noise_mode': 'normal', 'noise_mul': rng.uniform(0.2, 0.6)},
            'cleanup': {'delete': []},
        }


    # Getting data
    def get_attribute(self, location=None, shape=None, attribute='synthetic', **kwargs):
        """ Output requested `attribute`. If `location` is not provided, generates a new instance each time.
        For the same `location` values, uses the same generator instance (with the same reflectivity model):
        >>> location = (slice(10, 11), slice(100, 200), slice(1000, 1500))
        >>> synthetic = synthetic_field.get_attribute(location=location, attribute='synthetic')
        >>> impedance = synthetic_field.get_attribute(location=location, attribute='impedance')
        """
        _ = kwargs
        generator = self.get_generator(location=location, shape=shape)

        if attribute == 'labels':
            attribute = generator.params.get('attribute', self.attribute)

        # Main: velocity, reflectivity, synthetic
        if attribute in ['synthetic', 'geometry', 'image']:
            result = generator.get_attribute(attribute='synthetic')
        elif 'impedance' in attribute:
            result = generator.get_attribute(attribute='velocity_model')
        elif 'reflect' in attribute:
            result = generator.get_attribute(attribute='reflectivity_model')
        elif 'upward' in attribute:
            result = generator.get_increasing_impedance_model()

        # Labels: horizons and faults
        elif 'horizons' in attribute:
            result = generator.get_horizons(indices='all', format='mask', width=kwargs.get('width', 3))
        elif 'amplified' in attribute:
            result = generator.get_horizons(indices='amplified', format='mask', width=kwargs.get('width', 3))
        elif 'fault' in attribute:
            result = generator.get_faults(format='mask', width=kwargs.get('width', 3))

        # Fallback
        else:
            result = generator.get_attribute(attribute=attribute)

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


    # Utilities
    @classmethod
    def velocity_to_seismic(cls, velocity, ricker_width=4.3):
        """ Generate synthetic seismic out of velocity predictions. """
        result = []
        for velocity_array in velocity:
            generator = SyntheticGenerator()

            # Generating synthetic out of predicted velocity for all items
            generator.velocity_model = velocity_array
            generator.shape = generator.shape_padded = velocity_array.shape
            generator.depth = generator.depth_padded = velocity_array.shape[-1]

            (generator
                .make_density_model(randomization=None)
                .make_impedance_model()
                .make_reflectivity_model()
                .make_synthetic(ricker_width=ricker_width, ricker_points=100))
            result.append(generator.synthetic)

        return np.stack(result).astype(np.float32)

    # Normalization
    def make_normalization_stats(self, n=100, shape=None, attribute='synthetic'):
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

    # Visualization
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

    def show_slide(self, location=None, shape=None, **kwargs):
        """ Create one generator and show underlying models, synthetic and masks. """
        generator = self.get_generator(location=location, shape=shape)
        self._last_generator = generator
        return generator.show_slide(**kwargs)

    def show_roll(self, attribute='synthetic', n=25, **kwargs):
        """ Show attribute-images for a number of generators. """
        data = [[self.get_attribute(attribute=attribute)[0]] for _ in range(n)]
        cmap = 'gray'
        titles = list(range(n))

        # Display images
        plot_params = {
            'suptitle': f'Roll of `{attribute}`',
            'title': titles,
            'cmap': cmap,
            'colorbar': True,
            'ncols': 5,
            'scale': 0.5,
            **kwargs
        }
        return plot_image(data, **plot_params)
