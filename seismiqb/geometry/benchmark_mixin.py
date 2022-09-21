""" Collection of tools for benchmarking and testing array-like geometries. """
import time
import psutil

import numpy as np



class BenchmarkMixin:
    """ Methods for testing and benchmarking the geometry. """
    def equal(self, other, return_explanation=False):
        """ Check if two geometries are equal: have the same shape, headers and values. """
        condition = (self.shape == other.shape).all()
        if condition is False:
            explanation = f'Different shapes, {self.shape}  != {other.shape}'
            return (False, explanation) if return_explanation else False

        condition = (self.headers == other.headers).all().all()
        if condition is False:
            explanation = 'Different `headers` dataframes!'
            return (False, explanation) if return_explanation else False

        condition = (self.mean_matrix == other.mean_matrix).all()
        if condition is False:
            explanation = 'Different `mean_matrix` values!'
            return (False, explanation) if return_explanation else False

        for i in range(self.shape[0]):
            condition = (self[i] == other[i]).all()
            if condition is False:
                explanation = f'Different values in slide={i}!'
                return (False, explanation) if return_explanation else False

        return (True, '') if return_explanation else True

    @staticmethod
    def make_random_slide_locations(bounds, allowed_axis=(0, 1, 2), rng=None):
        """ Create random slide locations along one of the axis. """
        rng = rng or np.random.default_rng(rng)
        axis = rng.choice(a=allowed_axis)
        index = rng.integers(*bounds[axis])

        locations = [slice(None), slice(None), slice(None)]
        locations[axis] = slice(index, index + 1)
        return locations

    @staticmethod
    def make_random_crop_locations(bounds, size_min=10, size_max=100, rng=None):
        """ Create random crop locations. """
        rng = rng or np.random.default_rng(rng)
        if isinstance(size_min, int):
            size_min = (size_min, size_min, size_min)
        if isinstance(size_max, int):
            size_max = (size_max, size_max, size_max)

        point = rng.integers(*bounds)
        shape = rng.integers(low=size_min, high=size_max)
        locations = [slice(start, np.clip(start+size, bound_min, bound_max))
                     for start, size, bound_min, bound_max in zip(point, shape, bounds[0], bounds[1])]
        return locations


    def benchmark(self, n_slides=300, projections='ixd', n_crops=300, crop_size_min=5, crop_size_max=200, seed=42):
        """ Calculate average loading timings.
        Output is user, system and wall timings in milliseconds for slides and crops.

        Parameters
        ----------
        n_slides : int
            Number of slides to load.
        projections : str or sequence of int or str
            Allowed projections to generate slides along.
        n_crops : int
            Number of crops to load.
        crop_size_min : int or tuple of int
            A minimum size of generated crops.
            If tuple, then each number corresponds to size along each axis.
        crop_size_max : int or tuple of int
            A maximum size of generated crops.
            If tuple, then each number corresponds to size along each axis.
        seed : int
            Seed for the random numbers generator.
        """
        # Parse parameters
        projections = [self.parse_axis(proj) for proj in projections]
        if isinstance(crop_size_min, int):
            crop_size_min = (crop_size_min, crop_size_min, crop_size_min)
        if isinstance(crop_size_max, int):
            crop_size_max = (crop_size_max, crop_size_max, crop_size_max)

        rng = np.random.default_rng(seed)
        timings = {}

        # Calculate the average loading slide time
        if n_slides:
            self.reset_cache()

            timestamp_start, wall_start = psutil.cpu_times(), time.perf_counter()
            for _ in range(n_slides):
                slide_locations = self.make_random_slide_locations(bounds=self.bbox, rng=rng,
                                                                   allowed_axis=projections)
                slide_locations = tuple(slide_locations)
                _ = self[slide_locations]
            timestamp_end, wall_end = psutil.cpu_times(), time.perf_counter()

            timings['slide'] = {
                'user': 1000 * (timestamp_end[0] - timestamp_start[0]) / n_slides,
                'system': 1000 * (timestamp_end[2] - timestamp_start[2]) / n_slides,
                'wall': 1000 * (wall_end - wall_start) / n_slides
            }

        # Calculate the average loading crop time
        if n_crops:
            self.reset_cache()

            timestamp_start, wall_start = psutil.cpu_times(), time.perf_counter()
            for _ in range(n_crops):
                crop_locations = self.make_random_crop_locations(self.bbox.T, rng=rng,
                                                                size_min=crop_size_min, size_max=crop_size_max)
                crop_locations = tuple(crop_locations)
                _ = self[crop_locations]
            timestamp_end, wall_end = psutil.cpu_times(), time.perf_counter()

            timings['crop'] = {
                'user': 1000 * (timestamp_end[0] - timestamp_start[0]) / n_crops,
                'system': 1000 * (timestamp_end[2] - timestamp_start[2]) / n_crops,
                'wall': 1000 * (wall_end - wall_start) / n_crops
            }

        self.reset_cache()
        return timings
