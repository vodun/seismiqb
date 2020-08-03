""" Interpolate horizon from a carcass. """
import numpy as np

from .base import BaseController



class CarcassInterpolator(BaseController):
    """ Detector with convenient defaults to create a 2D surface from a sparce labeled carcass. """

    def train(self, dataset=None, horizon=None, **kwargs):
        """ Train model on a sparce labeled carcass. """
        if dataset is None and horizon is not None:
            dataset = self.make_dataset_from_horizon(horizon)

        horizon = dataset.labels[0][0]
        geometry = dataset.geometries[0]

        horizon_grid = (horizon.full_matrix != horizon.FILL_VALUE).astype(int)
        grid_coverage = (np.nansum(horizon_grid) /
                         (np.prod(geometry.cube_shape[:2]) - np.nansum(geometry.zero_traces)))
        self.log(f'Coverage of carcass is {grid_coverage}')

        self.make_sampler(dataset,
                          bins=np.array([500, 500, 100]),
                          use_grid=True, grid_src=horizon_grid)

        return super().train(dataset, use_grid=True, grid_src=horizon_grid, **kwargs)

    def inference(self, dataset, version=1, orientation='i', overlap_factor=2, heights_range=None,
                  batch_size_multiplier=1, **kwargs):
        """ Keep only the biggest horizon; name it after the carcass. """
        super().inference(dataset=dataset, version=version, orientation=orientation,
                          overlap_factor=overlap_factor, heights_range=heights_range,
                          batch_size_multiplier=batch_size_multiplier, **kwargs)

        self.predictions = [self.predictions[0]]
        self.predictions[0].name = f'from_{dataset.labels[0][0].name}'



class GridInterpolator(BaseController):
    """ Detector with convenient defaults to create a sparce carcass from a horizon by using a quality
    grid with supplied frequencies. Then, spread it to the whole cube spatial range.
    """
    def train(self, dataset=None, horizon=None, frequencies=(200, 200), **kwargs):
        """ Create a grid for a horizon, then train model on it. """
        if dataset is None and horizon is not None:
            dataset = self.make_dataset_from_horizon(horizon)

        horizon = dataset.labels[0][0]

        grid_coverages = self.make_grid(dataset, frequencies, iline=True, xline=True, margin=30)
        self.log(f'Coverage of grid with {frequencies} is {grid_coverages}')

        self.make_sampler(dataset,
                          bins=np.array([500, 500, 100]),
                          use_grid=True)

        return super().train(dataset, use_grid=True, **kwargs)


    def inference(self, dataset, version=1, orientation='i', overlap_factor=2, heights_range=None,
                  batch_size_multiplier=1, **kwargs):
        """ Keep only the biggest horizon; name it after the gridded horizon. """
        super().inference(dataset=dataset, version=version, orientation=orientation,
                          overlap_factor=overlap_factor, heights_range=heights_range,
                          batch_size_multiplier=batch_size_multiplier, **kwargs)

        self.predictions = [self.predictions[0]]
        self.predictions[0].name = f'from_gridded_{dataset.labels[0][0].name}'
