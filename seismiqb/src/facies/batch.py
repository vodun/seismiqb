""" Facies batch. """
import numpy as np

from ..crop_batch import SeismicCropBatch
from ...batchflow import action, inbatch_parallel



class FaciesBatch(SeismicCropBatch):
    """ Batch with facies-specific data loading. """
    def get_label(self, ix, src_labels, label_name):
        """ Get label by cube index, its attribute name and short name. """
        all_labels = self.dataset[ix, src_labels]
        labels = [label for label in all_labels if label.short_name == label_name]
        if len(labels) == 0:
            raise ValueError(f"Cannot find label with `{label_name}` name among {all_labels}.")
        if len(labels) > 1:
            raise ValueError(f"Cannot choose between several labels with identical names: {labels}.")
        return labels[0]


    @action
    @inbatch_parallel(init='indices', post='_assemble', target='for')
    def load(self, ix, dst, attribute=None, src_labels='labels', locations='locations', res_ndim=3, **kwargs):
        """ Load attribute for coordinates and label defined in given locations.

        Parameters
        ----------
        attribute : str
            A keyword from :attr:`~Horizon.ATTRIBUTE_TO_METHOD` keys, defining label attribute to make crops from.
        src_labels : str
            Dataset attribute with labels dict.
        locations : str
            Component of batch with locations of crops to load.
        res_ndim : 2 or 3
            Number of dimensions returned result should have.
        kwargs :
            Passed directly to either:
            - one of attribute-evaluating methods from :attr:`~Horizon.ATTRIBUTE_TO_METHOD` depending on `attribute`
            - or attribute-transforming method :meth:`~Horizon.transform_where_present`.

        Notes
        -----
        This method loads rectified data, e.g. amplitudes are croped relative
        to horizon and will form a straight plane in the resulting crop.
        """
        location, label_name = self.get(ix, locations)
        label = self.get_label(ix, src_labels, label_name)
        res = label.load_attribute(attribute=attribute, location=location, **kwargs)
        if res_ndim == 3 and res.ndim == 2:
            res = res[..., np.newaxis]
        elif res_ndim != res.ndim:
            raise ValueError(f"Expected returned crop to have {res_ndim} dimensions, but got {res.ndim}.")
        return res
