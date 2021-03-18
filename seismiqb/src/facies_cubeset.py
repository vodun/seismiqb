""" Container for storing seismic data and labels with facies-specific interaction model. """
from glob import glob
from warnings import warn
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.special import expit

from .cubeset import SeismicCubeset
from .horizon import Horizon
from .utility_classes import IndexedDict


class FaciesSeismicCubeset(SeismicCubeset):
    """ TODO """


    def load_labels(self, info, dst_labels=None, main_labels='horizons', label_affix='post', **kwargs):
        """ Load corresponding labels into corresponding dataset attributes.

        Parameters
        ----------
        correspondence : dict
            Correspondence between cube name and a list of patterns for its labels.
        labels_dirs : sequence
            Paths to folders to look corresponding labels with patterns from `correspondence` values for.
            Paths must be relative to cube location.
        dst_labels : sequence
            Names of dataset components to load corresponding labels into.
        main_labels : str
            Which dataset attribute assign to `self.labels`.
        kwargs :
            Passed directly to :meth:`.create_labels`.

        Examples
        --------
        The following argument values may be used to load for labels for 'CUBE_01_XXX':
        - from 'INPUTS/FACIES/FANS_HORIZONS/horizon_01_corrected.char' into `horizons` component;
        - from 'INPUTS/FACIES/FANS/fans_on_horizon_01_corrected_v8.char' into `fans` component,
        and assign `self.horizons` to `self.labels`.

        >>> correspondence = {'CUBE_01_XXX' : ['horizon_01']}
        >>> labels_dirs = ['INPUTS/FACIES/FANS_HORIZONS', 'INPUTS/FACIES/FANS']
        >>> dst_labels = ['horizons', 'fans']
        >>> main_labels = 'horizons'
        """
        self.load_geometries()

        label_dir = info['PATHS']["LABELS"]
        labels_subdirs = info['PATHS']["SUBDIRS"]
        dst_labels = dst_labels or [labels_subdir.lower() for labels_subdir in labels_subdirs]
        for labels_subdir, dst_label in zip(labels_subdirs, dst_labels):
            paths = defaultdict(list)
            for cube_name, labels in info['CUBES'].items():
                full_cube_name = f"amplitudes_{cube_name}"
                cube_path = self.index.get_fullpath(full_cube_name)
                cube_dir = cube_path[:cube_path.rfind('/')]
                for label in labels:
                    label_mask = f"*{label}" if label_affix == 'pre' else f"{label}*" if label_affix == 'post' else None
                    label_mask = '/'.join([cube_dir, label_dir, labels_subdir, label_mask])
                    label_path = glob(label_mask)
                    if len(label_path) == 0:
                        raise ValueError(f"No files found for mask `{label_mask}`")
                    if len(label_path) > 1:
                        raise ValueError('Multiple files match pattern')
                    paths[full_cube_name].append(label_path[0])
            self.create_labels(paths=paths, dst=dst_label, labels_class=Horizon, **kwargs)
        if main_labels is None:
            main_labels = dst_labels[0]
            warn("""Cubeset `labels` point now to `{}`.
                    To suppress this warning, explicitly pass value for `main_labels`.""".format(main_labels))
        self.labels = getattr(self, main_labels) #pylint: attribute-defined-outside-init

    def make_predictions(self, pipeline, crop_shape, overlap_factor, order=(1, 2, 0), src_labels='labels',
                         dst_labels='predictions', pipeline_var='predictions', bar='n', binarize=True):
        """
        Make predictions and put them into dataset attribute.

        Parameters
        ----------
        pipeline : Pipeline
            Inference pipeline.
        crop_shape : sequence
            Passed directly to :meth:`.make_grid`.
        overlap_factor : float or sequence
            Passed directly to :meth:`.make_grid`.
        src_labels : str
            Name of dataset component with items to make grid for.
        dst_labels : str
            Name of dataset component to put predictions into.
        pipeline_var : str
            Name of pipeline variable to get predictions for assemble from.
        order : tuple of int
            Passed directly to :meth:`.assemble_crops`.
        binarize : bool
            Whether convert probability to class label or not.
        """
        # pylint: disable=blacklisted-name
        setattr(self, dst_labels, IndexedDict({ix: [] for ix in self.indices}))
        for idx, labels in getattr(self, src_labels).items():
            for label in labels:
                self.make_grid(cube_name=idx, crop_shape=crop_shape, overlap_factor=overlap_factor,
                               heights=int(label.h_mean), mode='2d')
                pipeline = pipeline << self
                pipeline.run(batch_size=self.size, n_iters=self.grid_iters, bar=bar)
                prediction = self.assemble_crops(pipeline.v(pipeline_var), order=order).squeeze()
                prediction = expit(prediction)
                prediction = prediction.round() if binarize else prediction
                prediction_name = "{}_predicted".format(label.name)
                self[idx, dst_labels] += [Horizon(prediction, label.geometry, prediction_name)]

    def evaluate(self, true_src, pred_src, metrics_fn, output_format='df'):
        """ TODO """
        if output_format == 'dict':
            results = {}
            for idx in self.indices:
                results[idx] = []
                for true, pred in zip(self[idx, true_src], self[idx, pred_src]):
                    true_mask = true.load_attribute('masks', fill_value=0)
                    pred_mask = pred.load_attribute('masks', fill_value=0)
                    result = metrics_fn(true_mask, pred_mask)
                    results[idx].append(result)
        elif output_format == 'df':
            columns = ['cube', 'horizon', 'metrics']
            rows = []
            for idx in self.indices:
                for true, pred in zip(self[idx, true_src], self[idx, pred_src]):
                    true_mask = true.load_attribute('masks', fill_value=0)
                    pred_mask = pred.load_attribute('masks', fill_value=0)
                    metrics_value = metrics_fn(true_mask, pred_mask)
                    row = np.array([idx.lstrip('amplitudes_'), true.name, metrics_value])
                    rows.append(row)
            data = np.stack(rows)
            results = pd.DataFrame(data=data, columns=columns)
        return results
