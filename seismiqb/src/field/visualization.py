""" A mixin with field visualizations. """
#pylint: disable=global-variable-undefined
import numpy as np
from matplotlib import pyplot as plt

from .viewer import FieldViewer
from ..utils import AugmentedList
from ..plotters import plot_image, MatplotlibPlotter, show_3d
from ..labels.horizon_attributes import AttributesMixin



COLOR_GENERATOR = iter(MatplotlibPlotter.MASK_COLORS)
NAME_TO_COLOR = {}



class VisualizationMixin:
    """ Methods for field visualization: textual, 2d along various axis, 2d interactive, 3d. """
    # Textual representation
    def __repr__(self):
        return f"""<Field `{self.displayed_name}` at {hex(id(self))}>"""

    REPR_MAX_LEN = 100
    REPR_MAX_ROWS = 5

    def __str__(self):
        processed_prefix = 'un' if self.geometry.has_stats is False else ''
        labels_prefix = ' and labels:' if self.labels else ''
        msg = f'Field `{self.displayed_name}` with {processed_prefix}processed geometry{labels_prefix}\n'

        for label_src in self.loaded_labels:
            labels = getattr(self, label_src)
            names = [label.short_name for label in labels]

            labels_msg = ''
            line = f'    - {label_src}: ['
            while names:
                line += names.pop(0)

                if names:
                    line += ', '
                else:
                    labels_msg += line
                    break

                if len(line) > self.REPR_MAX_LEN:
                    labels_msg += line
                    line = '\n         ' + ' ' * len(label_src)

                if len(labels_msg) > self.REPR_MAX_LEN * self.REPR_MAX_ROWS:
                    break

            if names:
                labels_msg += f'\n         {" "*len(label_src)}and {len(names)} more item(s)'
            labels_msg += ']\n'
            msg += labels_msg
        return msg

    # 2D along axis
    def show_slide(self, loc, width=None, axis='i', zoom_slice=None,
                   src_geometry='geometry', src_labels='labels', indices='all', **kwargs):
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
            masks.append(self.make_mask(location=loc, axis=axis, src=src, width=width, indices=indices))
        mask = sum(masks)
        mask[mask == 0] = np.nan

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
            'title_label': title,
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
            'title_label': f'{labels_class}s on {self.displayed_name}',
            'xlabel': self.index_headers[0],
            'ylabel': self.index_headers[1],
            'cmap': ['Reds', 'black'],
            'alpha': [1.0, 0.4],
            'colorbar': True,
            **kwargs
        }
        return plot_image([map_, self.zero_traces], **kwargs)


    # 2D top-view maps
    def show(self, attributes='snr', mode='imshow', return_figure=False,
             savepath=None, bbox=False, load_kwargs=None, **plot_kwargs):
        """ Show one or more field attributes on one figure.

        Parameters
        ----------
        attributes : str, np.ndarray, dict or sequence of them
            Attributes to display.
            If str, then use `:meth:.load_attribute` to load the data. For example, `geometry/snr`, `labels:0/depths`.
            If instead of label index contains `:*`, for example, `labels:*/amplitudes`, then run this method
            for each of the objects in `labels` attribute.
            If np.ndarray, then directly used as data to display.
            If dict, then should define either string or np.ndarray (and used the same as previous types),
            as well as other parameters for `:meth:.load_attribute`.
            If sequence of them, then either should be a list to display loaded entities one over the other,
            or nested list to define separate axis and overlaying for each of them.
            For more details, refer to `:func:plot_image`.
        mode : 'imshow' or 'hist'
            Mode to display images.
        return_figure : bool
            Whether to return the figure.
        short_title : bool
            Whether to use only attribute names as titles for subplots.
        savepath : str, optional
            Path to save the figure. `**` is changed to a field base directory, `*` is changed to field base name.
        bbox : bool
            Whether crop horizon by its bounding box or not.
        load_kwargs : dict
            Loading parameters common for every requested attribute.
        plot_kwargs : dict
            Additional parameters for plot creation.

        Examples
        --------
        Simplest possible plot of a geometry-related attribute:
        >>> field.show('mean_matrix')

        Display attribute of a fan over the geometry map:
        >>> field.show(['mean_matrix', 'fans:0/masks'])

        Display attributes on separate axis:
        >>> field.show(['mean_matrix', 'horizons:0/fourier', custom_data_array], separate=True)

        Use various parameters for each of the plots:
        >>> field.show([{'src': 'labels:0/fourier', 'window': 20, 'normalize': True},
                        {'src': 'labels:0/fourier', 'window': 40, 'n_components': 3}],
                       separate=True)

        Display amplitudes and gradients for each of the horizons in a field:
        >>> field.show(['horizons:*/amplitudes', 'horizons:*/gradient'], separate=True)

        Display several attributes on multiple axes with overlays and save it near the cube:
        >>> field.show(['geometry/std_matrix', 'horizons:3/amplitudes',
                        ['horizons:3/instant_phases', 'fans:3/masks'],
                        ['horizons:3/instant_phases', predicted_mask]],
                       savepath='~/IMAGES/complex.png')
        """
        # If `*` is present, run `show` multiple times with `*` replaced to a label id
        attributes = AugmentedList(attributes)

        attribute_contains_wildcard = attributes.apply(self._wildcard_check)
        if any(attribute_contains_wildcard.flat):
            # Get len of each attribute
            get_labels_len = lambda attr: AugmentedList([len(getattr(self, item)) for item in self.loaded_labels
                                                         if '*' in attr and item in attr])
            labels_lists_lengths = attributes.apply(get_labels_len)
            if labels_lists_lengths[1:] == labels_lists_lengths[:-1]:
                raise ValueError('When using `show` with starred-expressions, length of attributes must be the same!')

            figures = []
            n_items = labels_lists_lengths.reference_object
            for label_num in range(n_items):
                #pylint: disable=cell-var-from-loop
                substitutor = lambda attr: attr.replace('*', str(label_num)) if isinstance(attr, str) else attr
                attributes_ = attributes.apply(substitutor)

                fig = self.show(attributes=attributes_, mode=mode, return_figure=True,
                                savepath=savepath, bbox=bbox, load_kwargs=load_kwargs, **plot_kwargs)
                figures.append(fig)
            return figures if return_figure else None

        # Transform load params into dicts, populate them with defaults and load data
        loaded = attributes.apply(self._show_load_data, common_params=load_kwargs)

        # Make plot defaults
        plot_defaults = loaded.apply(self._show_make_defaults).to_dict()
        plot_defaults['title'] = [item[0] if isinstance(item, list) else item for item in plot_defaults['title']]
        plot_defaults['suptitle'] = f'Field `{self.displayed_name}`'

        if bbox:
            bboxes_list = loaded.apply(lambda params: params['bbox'])
            lims_list = [np.stack([bboxes]).transpose(1, 2, 0) for bboxes in bboxes_list]
            plot_defaults['xlim'] = [(lims[0, 0].min(), lims[0, 1].max()) for lims in lims_list]
            plot_defaults['ylim'] = [(lims[1, 1].max(), lims[1, 0].min()) for lims in lims_list]

        if mode == 'imshow':
            plot_defaults['colorbar'] = True
            plot_defaults['xlabel'] = self.index_headers[0]
            plot_defaults['ylabel'] = self.index_headers[1]

        first_label_name = loaded.reference_object['label_name']
        savepath = self.make_path(savepath, name=first_label_name) if savepath is not None else None

        # Plot image with given params and return resulting figure
        plot_params = {**plot_defaults, **plot_kwargs}
        figure = plot_image(mode=mode, savepath=savepath, **plot_params)
        plt.show()

        return figure if return_figure else None

    # Auxilary methods utilized by `show`
    @staticmethod
    def _wildcard_check(attribute):
        if isinstance(attribute, str):
            result = ':*/' in attribute
        elif isinstance(attribute, dict):
            result = ':*/' in attribute['src']
        else:
            result = False
        return result

    ALIAS_TO_ATTRIBUTE = AttributesMixin.ALIAS_TO_ATTRIBUTE

    def _show_load_data(self, attribute_params, common_params):
        # Convert params to dict, if provided in a simple form (attribute name to load or ready-to-use numpy array)
        if isinstance(attribute_params, (str, np.ndarray)):
            attribute_params = {'src': attribute_params}
        elif not isinstance(attribute_params, dict):
            raise TypeError(f'Attribute should be either str, dict or array! Got {type(attribute_params)} instead.')

        # Extract attribute name and if it's an alias, convert it to the final attribute name
        src = attribute_params['src']
        attribute_name = src.split('/')[-1] if isinstance(src, str) else 'user data'
        attribute_name = self.ALIAS_TO_ATTRIBUTE.get(attribute_name, attribute_name)
        loaded = {'attribute_name': attribute_name}

        # Make load defaults
        load_defaults = {'dtype': np.float32}

        if attribute_name in ['fourier_decomposition', 'wavelet_decomposition']:
            load_defaults['n_components'] = 1

        if attribute_name in ['masks', 'full_binary_matrix']:
            load_defaults['fill_value'] = 0

        # Merge defaults with provided parameters
        load_params = {**load_defaults, **(common_params or dict()), **attribute_params}
        postprocess = load_params.pop('postprocess', lambda x: x)

        # Load
        if not isinstance(src, np.ndarray):
            data, label = self.load_attribute(_return_label=True, **load_params)
            loaded['label_name'] = label.displayed_name
            loaded['bbox'] = label.bbox[:2]
        else:
            data = src
            loaded['label_name'] = ''
            loaded['bbox'] = np.array([[0, max] for max in data.shape])

        loaded['data'] = postprocess(data.squeeze())

        return loaded

    CMAP_TO_ATTRIBUTE = {
        'Depths': ['full_matrix'],
        'Reds': ['spikes', 'quality_map'],
        'Metric': ['metric']
    }
    ATTRIBUTE_TO_CMAP = {attr: cmap for cmap, attributes in CMAP_TO_ATTRIBUTE.items()
                         for attr in attributes}

    def _show_make_defaults(self, loaded):
        params = {'data': loaded['data']}

        attribute_name = loaded['attribute_name']
        label_name = loaded['label_name']

        # Choose default cmap
        if attribute_name == 'full_binary_matrix':
            global_name = ''.join(filter(lambda x: x.isalpha(), attribute_name))
            if global_name not in NAME_TO_COLOR:
                NAME_TO_COLOR[global_name] = next(COLOR_GENERATOR)
            cmap = NAME_TO_COLOR[global_name]
        else:
            cmap = self.ATTRIBUTE_TO_CMAP.get(attribute_name, 'Basic')

        params['cmap'] = cmap

        # Choose default alpha
        if attribute_name in ['full_binary_matrix']:
            alpha = 0.7
        else:
            alpha = 1.0

        params['alpha'] = alpha

        # Make title text
        if label_name:
            title = f'`{label_name}`\n{attribute_name}'
        else:
            title = attribute_name

        params['title'] = title

        return params


    # 2D interactive
    def viewer(self, figsize=(8, 8), **kwargs):
        """ Interactive field viewer. """
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
