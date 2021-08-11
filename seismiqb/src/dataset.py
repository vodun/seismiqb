""" Container for storing seismic data and labels. """
#pylint: disable=too-many-lines, too-many-arguments
import os
from copy import copy
from textwrap import indent

import numpy as np
import pandas as pd

from ..batchflow import DatasetIndex, Dataset, Pipeline

from .field import Field
from .geometry import SeismicGeometry
from .plotters import plot_image, show_3d
from .crop_batch import SeismicCropBatch
from .utils import to_list, IndexedDict


class SeismicDataset(Dataset):
    """ !!.
    A common container for data entities: usually, seismic cubes and some type of labels.
    Entities are stored in dict-like objects, which can be indexed with either cube names or ordinal integer.

    Can be initialized with either:
        - `FilesIndex` instance
        - path(s) to cubes
        - from an instance of Horizon

    For batch generation and pipeline usage, we use `:meth:~SeismicCropBatch.make_locations` to convert batch indices
    from individual cubes to crops.

    Attributes
    ----------
    geometries : IndexedDict
        Storage of geometries, where keys are cube names and values are `SeismicGeometry` instances.
    labels : IndexedDict
        Nested storage of labels, where keys are cube names and values are sequences of labels.
    """
    #pylint: disable=keyword-arg-before-vararg
    def __init__(self, index, batch_class=SeismicCropBatch, *args, **kwargs):
        if args:
            raise TypeError('Positional args are not allowed for `SeismicDataset` initialization!')

        # Convert `index` to a dictionary
        if isinstance(index, (str, SeismicGeometry, Field)):
            index = [index]
        if isinstance(index, (tuple, list, DatasetIndex)):
            index = {item : None for item in index}

        if isinstance(index, dict):
            self.fields = IndexedDict()
            for geometry, labels in index.items():
                field = Field(geometry=geometry, labels=labels, **kwargs)
                self.fields[field.short_name] = field
        else:
            raise TypeError('!!.')

        dataset_index = DatasetIndex(list(self.fields.keys()))
        super().__init__(dataset_index, batch_class=batch_class)


    @classmethod
    def from_horizon(cls, horizon):
        """ Create dataset from an instance of Horizon. """
        return cls({horizon.field.geometry : {'horizons': [horizon]}})


    # Inner workings
    def gen_batch(self, batch_size=None, shuffle=False, n_iters=None, n_epochs=None, drop_last=False, **kwargs):
        """ Remove `n_epochs`, `shuffle` and `drop_last` from passed arguments.
        Set default value `batch_size` to the size of current dataset, removing the need to
        pass it to `next_batch` and `run` methods.
        """
        if (n_epochs is not None and n_epochs != 1) or shuffle or drop_last:
            raise TypeError(f'`SeismicCubeset` does not work with `n_epochs`, `shuffle` or `drop_last`!'
                            f'`{n_epochs}`, `{shuffle}`, `{drop_last}`')

        batch_size = batch_size or len(self)
        return super().gen_batch(batch_size, n_iters=n_iters, **kwargs)


    def get_nested_iterable(self, attribute):
        """ !!. """
        return IndexedDict({idx : getattr(field, attribute) for idx, field in self.fields.items()})

    def get_flat_iterable(self, attribute):
        """ !!. """
        return self.get_nested_iterable(attribute=attribute).flat

    def __getitem__(self, key):
        """ !!. """
        if isinstance(key, (int, str)):
            return self.fields[key]
        raise KeyError(f'Unsupported key for subscripting, {key}')

    def __getattr__(self, key):
        """ !!. """
        if isinstance(key, str) and key not in self.indices:
            return self.get_nested_iterable(key)
        raise AttributeError(f'Unknown attribute {key}')

    @property
    def geometries(self):
        """ !!. """
        return self.get_nested_iterable('geometry')



    # Default pipeline and batch for fast testing / introspection
    def data_pipeline(self, sampler, batch_size=4, width=4):
        """ Pipeline with default actions of creating locations, loading seismic images and corresponding masks. """
        return (self.p
                .make_locations(generator=sampler, batch_size=batch_size)
                .create_masks(dst='masks', width=width)
                .load_cubes(dst='images')
                .adaptive_reshape(src=['images', 'masks'])
                .normalize(src='images'))

    def data_batch(self, sampler, batch_size=4, width=4):
        """ Get one batch of `:meth:.data_pipeline` with `images` and `masks`. """
        return self.data_pipeline(sampler=sampler, batch_size=batch_size, width=width).next_batch()



    def dump_labels(self, path, name='points', separate=True):
        #TODO: remove?
        """ Dump label points to file. """
        for idx, labels_list in self.labels.items():
            for label in labels_list:
                dirname = os.path.dirname(self.index.get_fullpath(idx))
                if path[0] == '/':
                    path = path[1:]
                dirname = os.path.join(dirname, path)
                if not os.path.exists(dirname):
                    os.makedirs(dirname)
                name = label.name if separate else name
                save_to = os.path.join(dirname, name + '.npz')
                label.dump_points(save_to)

    def reset_caches(self, attributes=None):
        # TODO: rewrite for fields
        """ Reset lru cache for cached class attributes.

        Parameters
        ----------
        attributes : sequence of str
            Class attributes to reset cache in.
            If not supplied, reset in `geometries` and attributes added by `create_labels`.
        """
        cached_attributes = attributes or self._cached_attributes

        for attr in cached_attributes:
            for idx in self.indices:
                cached_attr = getattr(self, attr)[idx]
                cached_attr = cached_attr if isinstance(cached_attr, list) else [cached_attr]
                _ = [item.reset_cache() for item in cached_attr]


    # Textual and visual representation of dataset contents
    def __str__(self):
        msg = f'Seismic Dataset with {len(self)} field{"s" if len(self) > 1 else ""}:\n'
        for field in self.fields.values():
            msg += indent(f'{str(field)}\n', prefix='    ')
        return msg[:-1]

    # TODO: move to Field
    def show_3d(self, idx=0, src='labels', aspect_ratio=None, zoom_slice=None,
                 n_points=100, threshold=100, n_sticks=100, n_nodes=10,
                 slides=None, margin=(0, 0, 20), colors=None, **kwargs):
        """ Interactive 3D plot for some elements of cube. Roughly, does the following:
            - take some faults and/or horizons
            - select `n` points to represent the horizon surface and `n_sticks` and `n_nodes` for each fault
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface
            - draw few slides of the cube if needed

        Parameters
        ----------
        idx : int, str
            Cube index.
        src : str, Horizon-instance or list
            Items to draw, by default, 'labels'. If item of list (or `src` itself) is str, then all items of
            that dataset attribute will be drawn.
        aspect_ratio : None, tuple of floats or Nones
            Aspect ratio for each axis. Each None in the resulting tuple will be replaced by item from
            `(geometry.cube_shape[0] / geometry.cube_shape[1], 1, 1)`.
        zoom_slice : tuple of slices or None
            Crop from cube to show. By default, the whole cube volume will be shown.
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : number
            Threshold to remove triangles with bigger height differences in vertices.
        n_sticks : int
            Number of sticks for each fault.
        n_nodes : int
            Number of nodes for each stick.
        slides : list of tuples
            Each tuple is pair of location and axis to load slide from seismic cube.
        margin : tuple of ints
            Added margin for each axis, by default, (0, 0, 20).
        colors : dict or list
            Mapping of label class name to color defined as str, by default, all labels will be shown in green.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : number
            Size of the image.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        src = src if isinstance(src, (tuple, list)) else [src]
        geometry = self.geometries[idx]
        coords = []
        simplices = []

        if zoom_slice is None:
            zoom_slice = [slice(0, geometry.cube_shape[i]) for i in range(3)]
        else:
            zoom_slice = [
                slice(item.start or 0, item.stop or stop) for item, stop in zip(zoom_slice, geometry.cube_shape)
            ]
        zoom_slice = tuple(zoom_slice)
        triangulation_kwargs = {
            'n_points': n_points,
            'threshold': threshold,
            'n_sticks': n_sticks,
            'n_nodes': n_nodes,
            'slices': zoom_slice
        }

        labels = [getattr(self, src_)[idx] if isinstance(src_, str) else [src_] for src_ in src]
        labels = sum(labels, [])

        if isinstance(colors, dict):
            colors = [colors.get(type(label).__name__, colors.get('all', 'green')) for label in labels]

        simplices_colors = []
        for label, color in zip(labels, colors):
            x, y, z, simplices_ = label.make_triangulation(**triangulation_kwargs)
            if x is not None:
                simplices += [simplices_ + sum([len(item) for item in coords])]
                simplices_colors += [[color] * len(simplices_)]
                coords += [np.stack([x, y, z], axis=1)]

        simplices = np.concatenate(simplices, axis=0)
        coords = np.concatenate(coords, axis=0)
        simplices_colors = np.concatenate(simplices_colors)
        title = geometry.displayed_name

        default_aspect_ratio = (geometry.cube_shape[0] / geometry.cube_shape[1], 1, 1)
        aspect_ratio = [None] * 3 if aspect_ratio is None else aspect_ratio
        aspect_ratio = [item or default for item, default in zip(aspect_ratio, default_aspect_ratio)]

        axis_labels = (geometry.index_headers[0], geometry.index_headers[1], 'DEPTH')

        images = []
        if slides is not None:
            for loc, axis in slides:
                image = geometry.load_slide(loc, axis=axis)
                if axis == 0:
                    image = image[zoom_slice[1:]]
                elif axis == 1:
                    image = image[zoom_slice[0], zoom_slice[-1]]
                else:
                    image = image[zoom_slice[:-1]]
                images += [(image, loc, axis)]

        show_3d(coords[:, 0], coords[:, 1], coords[:, 2], simplices, title, zoom_slice, simplices_colors, margin=margin,
                aspect_ratio=aspect_ratio, axis_labels=axis_labels, images=images, **kwargs)

    def show_points(self, idx=0, src_labels='labels', **kwargs):
        """ Plot 2D map of points. """
        map_ = np.zeros(self.geometries[idx].cube_shape[:-1])
        denum = np.zeros(self.geometries[idx].cube_shape[:-1])
        for label in getattr(self, src_labels)[idx]:
            map_[label.points[:, 0], label.points[:, 1]] += label.points[:, 2]
            denum[label.points[:, 0], label.points[:, 1]] += 1
        denum[denum == 0] = 1
        map_ = map_ / denum
        map_[map_ == 0] = np.nan

        labels_class = type(getattr(self, src_labels)[idx][0]).__name__
        kwargs = {
            'title_label': f'{labels_class} on {self.indices[idx]}',
            'xlabel': self.geometries[idx].index_headers[0],
            'ylabel': self.geometries[idx].index_headers[1],
            'cmap': 'Reds',
            **kwargs
        }
        return plot_image(map_, **kwargs)

    def show_slide(self, loc, idx=0, axis='iline', zoom_slice=None, src_labels='labels', **kwargs):
        """ Show slide of the given cube on the given line.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        axis : int or str
            Number or name of axis to load slide along.
        zoom_slice : tuple of slices
            Tuple of slices to apply directly to 2d images.
        idx : str, int
            Number of cube in the index to use.
        src_labels : str
            Dataset components to show as labels.
        """
        components = ('images', 'masks') if getattr(self, src_labels)[idx] else ('images',)
        cube_name = self.indices[idx]
        geometry = self.geometries[cube_name]
        crop_shape = np.array(geometry.cube_shape)

        axis = geometry.parse_axis(axis)
        crop_shape[axis] = 1

        location = np.zeros((1, 9), dtype=np.int32)
        location[0, axis + 3] = loc
        location[0, axis + 6] = loc
        location[0, [6, 7, 8]] += crop_shape

        # Fake generator with one point only
        generator = lambda batch_size: location
        generator.to_names = lambda array: np.array([[cube_name, 'unknown']])

        pipeline = (Pipeline()
                    .make_locations(generator=generator)
                    .load_cubes(dst='images', src_labels=src_labels)
                    .normalize(src='images'))

        if 'masks' in components:
            use_labels = kwargs.pop('use_labels', 'all')
            width = kwargs.pop('width', crop_shape[-1] // 100)
            labels_pipeline = (Pipeline()
                               .create_masks(src_labels=src_labels, dst='masks', width=width, use_labels=use_labels))

            pipeline = pipeline + labels_pipeline

        batch = (pipeline << self).next_batch()
        imgs = [np.squeeze(getattr(batch, comp)) for comp in components]
        xmin, xmax, ymin, ymax = 0, imgs[0].shape[0], imgs[0].shape[1], 0

        if zoom_slice:
            imgs = [img[zoom_slice] for img in imgs]
            xmin = zoom_slice[0].start or xmin
            xmax = zoom_slice[0].stop or xmax
            ymin = zoom_slice[1].stop or ymin
            ymax = zoom_slice[1].start or ymax

        # Plotting defaults
        header = geometry.axis_names[axis]
        total = geometry.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = geometry.index_headers[0]
            ylabel = geometry.index_headers[1]

        kwargs = {
            'title_label': f'Data slice on cube `{geometry.displayed_name}`\n {header} {loc} out of {total}',
            'title_y': 1.01,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': False, # TODO: Make every horizon mask creation individual to allow their distinction while plot.
            **kwargs
        }
        return plot_image(imgs, **kwargs)



class FaciesCubeset(SeismicDataset):
    """ Storage extending `SeismicCubeset` functionality with methods for interaction with labels and their subsets.

    Most methods basically call methods of the same name for every label stored in requested attribute.
    """

    def add_subsets(self, src_subset, dst_base='labels'):
        """ Add nested labels.

        Parameters
        ----------
        src_labels : str
            Name of dataset attribute with labels to add as subsets.
        dst_base: str
            Name of dataset attribute with labels to add subsets to.
        """
        subset_labels = getattr(self, src_subset)
        base_labels = getattr(self, dst_base)
        if len(subset_labels.flat) != len(base_labels.flat):
            raise ValueError(f"Labels `{src_subset}` and `{dst_base}` have different lengths.")
        for subset, base in zip(subset_labels, base_labels):
            base.add_subset(name=src_subset, item=subset)

    def map_labels(self, function, indices=None, src_labels='labels', **kwargs):
        """ Call function for every item from labels list of requested cubes and return produced results.

        Parameters
        ----------
        function : str or callable
            If str, name of label method to call.
            If callable, applied to labels of chosen cubes.
        indices : str or sequence of str
            Indices of cubes which labels to map.
        src_labels : str
            Attribute with labels to map.
        kwargs :
            Passed directly to `function`.

        Returns
        -------
        IndexedDict where keys are cubes names and values are lists of results obtained by applied map.
        If all lists in result values are empty, None is returned instead.

        Examples
        --------
        >>> cubeset.map_labels('smooth_out', ['CUBE_01_AAA', 'CUBE_02_BBB'], 'horizons', iters=3)
        """
        results = IndexedDict({idx: [] for idx in self.indices})
        for label in getattr(self, src_labels).flatten(keys=indices):
            if isinstance(function, str):
                res = getattr(label, function)(**kwargs)
            elif callable(function):
                res = function(label, **kwargs)
            if res is not None:
                results[label.geometry.short_name].append(res)
        return results if len(results.flat) > 0 else None

    def show(self, attributes='depths', src_labels='labels', indices=None, **kwargs):
        """ Show attributes of requested dataset labels. """
        return self.map_labels(function='show', src_labels=src_labels, indices=indices, attributes=attributes, **kwargs)

    def invert_subsets(self, subset, src_labels='labels', dst_labels=None, add_subsets=True):
        """ Invert matrices of requested dataset labels and store resulted labels in cubeset. """
        dst_labels = dst_labels or f"{subset}_inverted"
        inverted = self.map_labels(function='invert_subset', indices=None, src_labels=src_labels, subset=subset)

        setattr(self, dst_labels, inverted)
        if add_subsets:
            self.add_subsets(src_subset=dst_labels, dst_base=src_labels)

    def add_merged_labels(self, src_labels, dst_labels, indices=None, dst_base='labels'):
        """ Merge requested labels and store resulted labels in cubeset. """
        results = IndexedDict({idx: [] for idx in self.indices})
        indices = to_list(indices or self.indices)
        for idx in indices:
            to_merge = self[idx, src_labels]
            # since `merge_list` merges all horizons into first object from the list,
            # make a copy of first horizon in list to save merge into its instance
            container = copy(to_merge[0])
            container.name = f"Merged {'/'.join([horizon.short_name for horizon in to_merge])}"
            _ = [container.adjacent_merge(horizon, inplace=True, mean_threshold=999, adjacency=999)
                 for horizon in to_merge]
            container.reset_cache()
            results[idx].append(container)
        setattr(self, dst_labels, results)
        if dst_base:
            self.add_subsets(src_subset=dst_labels, dst_base=dst_base)

    def evaluate(self, src_true, src_pred, metrics_fn, metrics_names=None, indices=None, src_labels='labels'):
        """ Apply given function to 'masks' attribute of requested labels and return merged dataframe of results.

        Parameters
        ----------
        src_true : str
            Name of `labels` subset to load true mask from.
        src_pred : str
            Name of `labels` subset to load prediction mask from.
        metrics_fn : callable or list of callable
            Metrics function(s) to calculate.
        metrics_name : str, optional
            Name of the column with metrics values in resulted dataframe.
        """
        metrics_values = self.map_labels(function='evaluate', src_labels=src_labels, indices=indices,
                                         src_true=src_true, src_pred=src_pred,
                                         metrics_fn=metrics_fn, metrics_names=metrics_names)
        return pd.concat(metrics_values.flat)
