""" Collection of tools for benchmarking and testing array-like geometries. """

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

        for i in range(self.shape[0]):
            condition = (self[i] == other[i]).all()
            if condition is False:
                explanation = f'Different values in slide={i}!'
                return (False, explanation) if return_explanation else False

        return (True, '') if return_explanation else True
