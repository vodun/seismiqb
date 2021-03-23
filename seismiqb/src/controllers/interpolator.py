""" Interpolate horizon from a carcass. """
#pylint: disable=attribute-defined-outside-init
import numpy as np

from .horizon import HorizonController


class Interpolator(HorizonController):
    """ Horizon detector with automated choise between Carcass and Grid interpolation. """
    def train(self, dataset=None, cube_paths=None, horizon_paths=None, horizon=None, frequencies=None, **kwargs):
        """ Either train on a carcass or cut carcass from a full horizon and train on it. """
        if dataset is None:
            dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)

        geometry = dataset.geometries[0]
        horizon = dataset.labels[0][0]

        if horizon.is_carcass:
            # Already a carcass
            self.from_carcass = True
            self.log('Using CARCASS mode of Interpolator')

            grid = (horizon.full_matrix != horizon.FILL_VALUE).astype(int)
            grid_coverage = (np.nansum(grid) /
                            (np.prod(geometry.cube_shape[:2]) - np.nansum(geometry.zero_traces)))
            self.log(f'Coverage of carcass is {grid_coverage}')

            self.make_sampler(dataset,
                              bins=np.array([500, 500, 100]),
                              use_grid=True, grid_src=grid)
        else:
            # Cut a carcass out of the horizon: used for tests
            self.from_carcass = False
            self.log('Using GRID mode of Interpolator')
            grid = 'quality_grid'

            self.make_grid(dataset=dataset, frequencies=frequencies, iline=True, xline=True, margin=30)

            self.make_sampler(dataset,
                            bins=np.array([500, 500, 100]),
                            use_grid=True, grid_src=grid)

        return super().train(dataset=dataset, adaptive_slices=True, grid_src=grid, **kwargs)

    def inference(self, dataset, model, **kwargs):
        """ Inference with different naming conventions. """
        prediction = super().inference(dataset=dataset, model=model, **kwargs)[0]

        if self.from_carcass:
            prediction.name = f'from_{dataset.labels[0][0].name}'
        else:
            prediction.name = f'from_gridded_{dataset.labels[0][0].name}'
        return prediction

    # One method to rule them all
    def run(self, cube_paths=None, horizon_paths=None, horizon=None, frequencies=None, **kwargs):
        """ Run the entire procedure of horizon detection: from loading the carcass/grid to outputs. """
        dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)
        model = self.train(dataset=dataset, frequencies=frequencies, **kwargs)

        prediction = self.inference(dataset, model)
        prediction = self.postprocess(prediction)
        self.evaluate(prediction, dataset=dataset)
        return prediction
