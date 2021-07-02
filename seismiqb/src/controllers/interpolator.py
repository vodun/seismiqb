""" Interpolate horizon from a carcass. """
#pylint: disable=attribute-defined-outside-init
from .horizon import HorizonController


class Interpolator(HorizonController):
    """ !!. """
    def train(self, dataset=None, cube_paths=None, horizon_paths=None, horizon=None, **kwargs):
        """ !!. """
        if dataset is None:
            dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)

        horizon = dataset.labels[0][0]
        self.log(f'Coverage of carcass is {horizon.coverage}')

        sampler = self.make_sampler(dataset)
        return super().train(dataset=dataset, sampler=sampler, **kwargs)

    def inference(self, dataset, model, config=None, name=None, **kwargs):
        """ !!. """
        prediction = super().inference(dataset=dataset, model=model, **kwargs)[0]

        if name is None:
            if len(dataset.labels[0]) > 0:
                name = dataset.labels[0][0].name
            else:
                name = f'prediction_{int(prediction.h_mean)}'

        prediction.name = f'from_{name}'
        return prediction

    # One method to rule them all
    def run(self, cube_paths=None, horizon_paths=None, horizon=None, **kwargs):
        """ Run the entire procedure of horizon detection: from loading the carcass/grid to outputs. """
        dataset = self.make_dataset(cube_paths=cube_paths, horizon_paths=horizon_paths, horizon=horizon)
        model = self.train(dataset=dataset, **kwargs)

        prediction = self.inference(dataset, model)
        prediction = self.postprocess(prediction)
        self.evaluate(prediction, dataset=dataset)
        return prediction
