""" A mixin with field visualizations. """
#pylint: disable=global-variable-undefined
from functools import partial
import numpy as np
from matplotlib import pyplot as plt

from .viewer import FieldViewer
from ..plotters import plot_image, show_3d


COLOR_GENERATOR = iter(['firebrick', 'darkorchid', 'sandybrown'])
NAME_TO_COLOR = {}



class VisualizationMixin:
    """ !!. """
    # Textual representation
    def __repr__(self):
        return f"""<Field `{self.displayed_name}` at {hex(id(self))}>"""

    def __str__(self):
        processed_prefix = 'un' if self.geometry.has_stats is False else ''
        labels_prefix = ':' if self.labels else ''
        msg = f'Field `{self.displayed_name}` with {processed_prefix}processed geometry{labels_prefix}\n'
        for label in self.labels:
            msg += f'    {label.name}\n'
        return msg

    # 2D along axis
    def show_slide(self, loc, width=None, axis='i', zoom_slice=None,
                   src_geometry='geometry', src_labels='labels', **kwargs):
        """ Show slide with horizon on it.

        Parameters
        ----------
        loc : int
            Number of slide to load.
        width : int
            Horizon thickness. If None given, set to 1% of seismic slide height.
        axis : int
            Number of axis to load slide along.
        zoom_slice : tuple
            Tuple of slices to apply directly to 2d images.
        """
        axis = self.geometry.parse_axis(axis)

        # Load seismic and mask
        seismic_slide = getattr(self, src_geometry).load_slide(loc=loc, axis=axis)

        src_labels = src_labels if isinstance(src_labels, (tuple, list)) else [src_labels]
        masks = []
        for src in src_labels:
            masks.append(self.make_mask(location=loc, axis=axis, src=src, width=width))
        mask = sum(masks)

        # src_labels = src_labels if isinstance(src_labels, (tuple, list)) else [src_labels]
        # masks = []
        # for src in src_labels:
        #     masks.extend(getattr(self, src).load_slide(loc=loc, axis=axis, width=width))
        # mask = sum(masks)

        seismic_slide, mask = np.squeeze(seismic_slide), np.squeeze(mask)
        xmin, xmax, ymin, ymax = 0, seismic_slide.shape[0], seismic_slide.shape[1], 0

        if zoom_slice:
            seismic_slide = seismic_slide[zoom_slice]
            mask = mask[zoom_slice]
            xmin = zoom_slice[0].start or xmin
            xmax = zoom_slice[0].stop or xmax
            ymin = zoom_slice[1].stop or ymin
            ymax = zoom_slice[1].start or ymax

        # defaults for plotting if not supplied in kwargs
        header = self.geometry.axis_names[axis]
        total = self.geometry.cube_shape[axis]

        if axis in [0, 1]:
            xlabel = self.geometry.index_headers[1 - axis]
            ylabel = 'DEPTH'
        if axis == 2:
            xlabel = self.geometry.index_headers[0]
            ylabel = self.geometry.index_headers[1]
            total = self.geometry.depth

        title = f'Field `{self.displayed_name}`\n {header} {loc} out of {total}'

        kwargs = {
            'figsize': (16, 8),
            'title_label': title,
            'title_y': 1.02,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'extent': (xmin, xmax, ymin, ymax),
            'legend': False,
            'labeltop': False,
            'labelright': False,
            'curve_width': width,
            'grid': [False, True],
            'colorbar': [True, False],
            **kwargs
        }
        return plot_image(data=[seismic_slide, mask], **kwargs)


    # 2D depth slice
    def show_points(self, src='labels', **kwargs):
        """ Plot 2D map of points. """
        map_ = np.zeros(self.spatial_shape)
        denum = np.zeros(self.spatial_shape)

        for label in getattr(self, src):
            map_[label.points[:, 0], label.points[:, 1]] += label.points[:, 2]
            denum[label.points[:, 0], label.points[:, 1]] += 1
        denum[denum == 0] = 1
        map_ = map_ / denum
        map_[map_ == 0] = np.nan

        labels_class = type(getattr(self, src)[0]).__name__
        kwargs = {
            'title_label': f'{labels_class} on {self.displayed_name}',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'cmap': 'Reds',
            **kwargs
        }
        return plot_image(map_, **kwargs)


    # 2D top-view maps
    @staticmethod
    def apply_nested(function, items):
        """ Apply `function` to each of `items`, keeping the same nestedness. Works with lists only. """
        # Not a list
        if not isinstance(items, list):
            return function(items)

        # Simple list
        if all(not isinstance(item, list) for item in items):
            return [function(item) for item in items]

        # Nested list
        result = []
        for item in items:
            item = item if isinstance(item, list) else [item]
            result.append(VisualizationMixin.apply_nested(function, item))
        return result


    @staticmethod
    def _show_to_dict(attribute):
        # Convert everything to a dictionary
        if isinstance(attribute, (str, np.ndarray)):
            return {'src': attribute}
        if isinstance(attribute, dict):
            return attribute
        raise TypeError(f'Each attribute should be either str, dict or array! Got {type(attribute)} instead.')

    @staticmethod
    def _show_extract_names(attribute_dict):
        name = attribute_dict.get('name')
        if name is None:
            src = attribute_dict['src']
            if isinstance(src, np.ndarray):
                name = 'custom'
            if isinstance(src, str):
                name = src

        attribute_dict['name'] = name
        attribute_dict['short_name'] = name.split('/')[-1]
        return attribute_dict

    @staticmethod
    def _show_add_load_defaults(attribute_dict):
        attribute_dict['fill_value'] = np.nan
        attribute_dict['dtype'] = np.float32

        short_name = attribute_dict['short_name']

        if short_name in ['fourier', 'wavelet', 'fourier_decomposition', 'wavelet_decomposition']:
            attribute_dict['n_components'] = 1
        if short_name in ['masks', 'full_binary_matrix']:
            attribute_dict['fill_value'] = 0
            attribute_dict['alpha'] = 0.7
        return attribute_dict

    @staticmethod
    def _show_load_data(attribute_dict, method, **kwargs):
        """ Manage data loading depending on load params type. """
        src = attribute_dict['src']

        # Already an array: no further actions needed
        if isinstance(src, np.ndarray):
            output = src
        else:
            # Load data with `method`
            load = {**kwargs, **attribute_dict}
            postprocess = load.pop('postprocess', lambda x: x)
            output = method(**load).squeeze()
            output = postprocess(output)

        attribute_dict['output'] = output
        return attribute_dict

    @staticmethod
    def _show_add_plot_defaults(attribute_dict):
        name = attribute_dict['name']
        short_name = attribute_dict['short_name']

        # Alphas
        if short_name in ['masks', 'full_binary_matrix']:
            attribute_dict['alpha'] = 0.7
        else:
            attribute_dict['alpha'] = 1.0

        # Cmaps
        if short_name in ['depths', 'matrix', 'full_matrix']:
            attribute_dict['cmap'] = 'Depths'
        elif short_name in  ['metric', 'metrics']:
            attribute_dict['cmap'] = 'Metric'
        elif short_name == 'quality_map':
            attribute_dict['cmap'] = 'Reds'
        elif short_name == 'full_binary_matrix':
            if name not in NAME_TO_COLOR:
                NAME_TO_COLOR[name] = next(COLOR_GENERATOR)
            attribute_dict['cmap'] = NAME_TO_COLOR[name]
        else:
            attribute_dict['cmap'] = 'ocean'

        return attribute_dict


    def show(self, attributes='snr', mode='imshow', return_figure=False, width=9, short_title=False,
             savepath=None, **kwargs):
        """ !!. """
        # pylint: disable=too-many-statements

        # To dictionaries -> add names - > add defaults
        attribute_dicts = self.apply_nested(self._show_to_dict, attributes)
        attribute_dicts = self.apply_nested(self._show_extract_names, attribute_dicts)
        attribute_dicts = self.apply_nested(self._show_add_load_defaults, attribute_dicts)

        # Actually load data
        load_method = partial(self._show_load_data, method=self.load_attribute, **kwargs)
        attribute_dicts = self.apply_nested(load_method, attribute_dicts)

        # Plot params for attributes
        attribute_dicts = self.apply_nested(self._show_add_plot_defaults, attribute_dicts)
        data = self.apply_nested(lambda dct: dct['output'], attribute_dicts)
        titles = self.apply_nested(lambda dct: dct['short_name' if short_title else 'name'], attribute_dicts)
        alphas = self.apply_nested(lambda dct: dct['alpha'], attribute_dicts)
        cmaps = self.apply_nested(lambda dct: dct['cmap'], attribute_dicts)

        if isinstance(titles, list):
            titles = [item[0] if isinstance(item, list) else item for item in titles]

        # Prepare plot defaults
        plot_defaults = {
            'suptitle_label': f'Field `{self.displayed_name}`',
            'title_label': titles,
            'tight_layout': True,
            'return_figure': True,
        }

        # Defaults for chosen mode
        if mode == 'imshow':
            x, y = self.spatial_shape
            n_subplots = len(data) if isinstance(data, list) else 1

            plot_defaults = {
                **plot_defaults,
                'figsize': (x / min(x, y) * n_subplots * 7, y / min(x, y) * 7),
                'cmap': cmaps,
                'alpha': alphas,
                'colorbar': True,
                'xlabel': self.index_headers[0],
                'ylabel': self.index_headers[1],
            }
        elif mode == 'hist':
            plot_defaults = {**plot_defaults, 'figsize': (n_subplots * 10, 5)}
        else:
            raise ValueError(f"Valid modes are 'imshow' or 'hist', but '{mode}' was given.")

        # Plot image with given params and return resulting figure
        params = {**plot_defaults, **kwargs}
        savepath = self.make_savepath(savepath, name=self.short_name) if savepath is not None else None

        figure = plot_image(data=data, mode=mode, savepath=savepath, **params)
        plt.show()
        return figure if return_figure else None


    # 2D interactive
    def viewer(self, figsize=(8, 8), **kwargs):
        """ !!. """
        return FieldViewer(field=self, figsize=figsize, **kwargs)


    # 3D interactive
    def show_3d(self, src='labels', aspect_ratio=None, zoom_slice=None,
                 n_points=100, threshold=100, n_sticks=100, n_nodes=10,
                 slides=None, margin=(0, 0, 20), colors=None, **kwargs):
        """ Interactive 3D plot for some elements of a field.
        Roughly, does the following:
            - take some faults and/or horizons
            - select `n` points to represent the horizon surface and `n_sticks` and `n_nodes` for each fault
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface
            - draw few slides of the cube if needed

        Parameters
        ----------
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
        coords = []
        simplices = []

        if zoom_slice is None:
            zoom_slice = [slice(0, s) for s in self.shape]
        else:
            zoom_slice = [
                slice(item.start or 0, item.stop or stop) for item, stop in zip(zoom_slice, self.shape)
            ]
        zoom_slice = tuple(zoom_slice)
        triangulation_kwargs = {
            'n_points': n_points,
            'threshold': threshold,
            'n_sticks': n_sticks,
            'n_nodes': n_nodes,
            'slices': zoom_slice
        }

        labels = [getattr(self, src_) if isinstance(src_, str) else [src_] for src_ in src]
        labels = sum(labels, [])

        if colors is None:
            colors = ['green' for label in labels]
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
        title = self.displayed_name

        default_aspect_ratio = (self.shape[0] / self.shape[1], 1, 1)
        aspect_ratio = [None] * 3 if aspect_ratio is None else aspect_ratio
        aspect_ratio = [item or default for item, default in zip(aspect_ratio, default_aspect_ratio)]

        axis_labels = (self.index_headers[0], self.index_headers[1], 'DEPTH')

        images = []
        if slides is not None:
            for loc, axis in slides:
                image = self.geometry.load_slide(loc, axis=axis)
                if axis == 0:
                    image = image[zoom_slice[1:]]
                elif axis == 1:
                    image = image[zoom_slice[0], zoom_slice[-1]]
                else:
                    image = image[zoom_slice[:-1]]
                images += [(image, loc, axis)]

        show_3d(coords[:, 0], coords[:, 1], coords[:, 2], simplices, title, zoom_slice, simplices_colors, margin=margin,
                aspect_ratio=aspect_ratio, axis_labels=axis_labels, images=images, **kwargs)
