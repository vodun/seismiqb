""" Pre-defined parameter generators. """
import numpy as np
from ...batchflow import Config


class ParamGeneratorFactory:
    """ Create a `param_generator` function with desired behavior.
    Allows to easily change parts of the sampled configs by passing `config` variable,
    which takes priority over default created parameters.

    Each of the values in `config` can be a callable: in this case, it will be called with no arguments to make config.
    """
    def __init__(self, config=None, attribute='labels', horizon_frequency=25, faults_p=(0.7, 0.3), faults_config=None):
        config = config if config is not None else {}
        self.config = Config(config)

        self.horizon_frequency = horizon_frequency

        self.faults_p = faults_p
        self.faults_a = np.arange(len(faults_p))
        self.faults_config = faults_config if faults_config is not None else {}


    def make_fault_params(self, rng):
        """ !!. """
        return {'coordinates': 'random',
                'max_shift': rng.uniform(20, 40),
                'shift_sign': rng.choice([-1, 1]),
                'width': rng.uniform(1.0, 4.0),
                'update_horizon_matrices': True,
                **self.faults_config}

    def __call__(self, shape=None, rng=None):
        rng = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

        num_horizons = shape[-1] // self.horizon_frequency + rng.integers(-3, +3)

        num_faults = rng.choice(a=self.faults_a, p=self.faults_p)
        fault_params = [self.make_fault_params(rng=rng) for _ in range(num_faults)]

        config = Config({
            'attribute': self.attribute,
            'make_velocity_vector': {'num_horizons': num_horizons,                                  # scales with depth
                                     'limits': (2000, 6000),
                                     'randomization': 'uniform', 'randomization_scale': 0.3,
                                     'amplify_probability': 0.2, 'amplify_range': (2.0, 4.0),
                                     'amplify_sign_probability': 0.8},
            'make_horizons': {'shape': shape,
                              'interval_randomization': 'uniform',
                              'interval_randomization_scale': 0.3,
                              'interval_min': rng.uniform(0.2, 0.6),
                              'randomization1_scale': 0.25, 'num_nodes': 10,
                              'randomization2_scale': 0.25, 'locs_n_range': (3, 5),
                              'output_range': (-0.2, 0.8)},
            'make_velocity_model': {},

            'make_fault_2d': fault_params,

            'make_density_model': {'randomization': 'uniform',
                                   'randomization_limits': (0.95, 1.05)},
            'make_impedance_model': {},
            'make_reflectivity_model': {},

            'make_synthetic': {'ricker_width': rng.uniform(4.7, 5.3),
                               'ricker_points': 100},
            'postprocess_synthetic': {'sigma': 0.5, 'clip': True,
                                      'noise_mode': 'normal', 'noise_mul': rng.uniform(0.1, 0.3)},
            'cleanup': {'delete': []},
        })

        self_config = Config({key : value() if callable(value) else value
                              for key, value in self.config.flatten().items()})
        config += self_config
        return config


default_param_generator = ParamGeneratorFactory()
