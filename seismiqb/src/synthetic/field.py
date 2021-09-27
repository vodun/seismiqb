""" Placeholder for a synthetic field. """


class SyntheticField:
    """ Placeholder for a synthetic field. """
    def __init__(self):
        """ Store all the parameters for later usage, as well as reference to a synthetic generator. """

    def make_sampler(self):
        """ Create a sampler to generate pseudo-locations. """

    def load_seismic(self):
        """ Create a synthetic seismic slide. """

    def make_mask(self, src=('horizons', 'faults')):
        """ Make segmentation mask. """
        _ = src

    def show(self):
        """ A simple slide visualization. """

    def __repr__(self):
        pass

    def __str__(self):
        pass
