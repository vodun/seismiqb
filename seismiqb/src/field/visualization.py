""" A mixin with field visualizations. """
import numpy as np
from matplotlib import pyplot as plt

from ..plotters import plot_image


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

    # Graphical representation
    # 2D
    def show(self, attributes='snr', mode='imshow', return_figure=False, enlarge=False, width=9, **kwargs):
        """ !!. """
        # pylint: disable=too-many-statements
        def apply_by_scenario(action, params):
            """ Generic method that applies given action to params depending on their type. """
            if not isinstance(params, list):
                res = action(params)
            elif all(not isinstance(item, list) for item in params):
                res = [action(subplot_params) for subplot_params in params]
            else:
                res = []
                for subplot_params in params:
                    if isinstance(subplot_params, list):
                        subplot_res = [action(layer_params) for layer_params in subplot_params]
                    else:
                        subplot_res = [action(subplot_params)]
                    res.append(subplot_res)
            return res

        # Load attributes and put obtained data in a list with same nestedness as `load`
        def load_data(attributes):
            """ Manage data loading depending on load params type. """
            if isinstance(attributes, np.ndarray):
                return attributes
            if isinstance(attributes, str):
                load = {'src': attributes}
            if isinstance(attributes, dict):
                load = attributes
            postprocess = load.pop('postprocess', lambda x: x)

            load_defaults = {}
            if load['src'].split('/')[-1] in ['fourier', 'wavelet']:
                load_defaults['n_components'] = 1
            if load['src'].split('/')[-1] in ['masks', 'full_binary_matrix']:
                load_defaults['fill_value'] = 0

            load = {**load_defaults, **load}
            data = self.load_attribute(**load).squeeze()
            return postprocess(data)

        def enlarge_data(data):
            if enlarge and self.is_carcass:
                data = self.matrix_enlarge_carcass(data, width)
            return data

        data = apply_by_scenario(load_data, attributes)
        data = apply_by_scenario(enlarge_data, data)

        # Make titles
        def extract_data_name(attributes):
            if isinstance(attributes, np.ndarray):
                name = 'custom'
            elif isinstance(attributes, dict):
                name = attributes['src']
            elif isinstance(attributes, str):
                name = attributes
            return name

        names = apply_by_scenario(extract_data_name, attributes)
        n_subplots = len(data) if isinstance(data, list) else 1

        def make_titles(names):
            if any(isinstance(item, list) for item in attributes):
                return [', '.join(subplot_names) for subplot_names in names]
            return names

        defaults = {
            'title_label': make_titles(names),
            'suptitle_label': f"Field `{self.displayed_name}`",
            'colorbar': mode == 'imshow',
            'tight_layout': True,
            'return_figure': True,
        }

        # Infer defaults for `mode`: generate cmaps according to requested data, set axis labels as index headers
        default_colors = ['firebrick', 'darkorchid', 'sandybrown']
        gen_color = (color for color in default_colors)
        name_to_color = {}
        def make_cmap(name):
            attr = name.split('/')[-1]
            # attr = self.ALIAS_TO_ATTRIBUTE.get(attr, attr)

            if attr in ['matrix', 'full_matrix']:
                return 'Depths'
            if attr == 'metric':
                return 'Metric'
            if attr == 'full_binary_matrix':
                if name not in name_to_color:
                    name_to_color[name] = next(gen_color)
                return name_to_color[name]
            return 'ocean'

        def make_alpha(name):
            return 0.7 if name.split('/')[-1] == 'masks' else 1.0

        if mode == 'imshow':
            x, y = self.spatial_shape
            defaults = {
                **defaults,
                'figsize': (x / min(x, y) * n_subplots * 7, y / min(x, y) * 7),
                # 'xlim': self.bbox[0],
                # 'ylim': self.bbox[1][::-1],
                'cmap': apply_by_scenario(make_cmap, names),
                'alpha': apply_by_scenario(make_alpha, names),
                'xlabel': self.index_headers[0],
                'ylabel': self.index_headers[1],
            }
        elif mode == 'hist':
            defaults = {**defaults, 'figsize': (n_subplots * 10, 5)}
        else:
            raise ValueError(f"Valid modes are 'imshow' or 'hist', but '{mode}' was given.")

        # Merge default and given params
        params = {**defaults, **kwargs}

        # Substitute asterisks in title-like parameters with default suptitle
        for text in ['suptitle_label', 'suptitle', 'title_label', 'title', 't']:
            if text in params:
                params[text] = apply_by_scenario(lambda s: s.replace('*', defaults['suptitle_label']), params[text])

        # Substitute asterisk in `savepath` parameter with label name
        if 'savepath' in params:
            params['savepath'] = self.make_savepath(params['savepath'], name=self.short_name)

        # Plot image with given params and return resulting figure
        figure = plot_image(data=data, mode=mode, **params)
        plt.show()

        return figure if return_figure else None


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
            masks.extend(getattr(self, src).load_slide(loc=loc, axis=axis, width=width))
        mask = sum(masks)

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
