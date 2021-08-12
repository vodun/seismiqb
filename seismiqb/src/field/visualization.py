""" A mixin with field visualizations. """
#pylint: disable=global-variable-undefined
from functools import partial
import numpy as np
from matplotlib import pyplot as plt

from ..plotters import plot_image


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

    # Graphical representation: 2D along axis
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
        short_name = attribute_dict['short_name']

        if short_name in ['fourier', 'wavelet']:
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
        if short_name in ['matrix', 'full_matrix']:
            attribute_dict['cmap'] = 'Depths'
        elif short_name == 'metric':
            attribute_dict['cmap'] = 'Metric'
        elif short_name == 'full_binary_matrix':
            if name not in NAME_TO_COLOR:
                NAME_TO_COLOR[name] = next(COLOR_GENERATOR)
            attribute_dict['cmap'] = NAME_TO_COLOR[name]
        else:
            attribute_dict['cmap'] = 'ocean'

        return attribute_dict


    def show(self, attributes='snr', mode='imshow', return_figure=False, width=9, savepath=None, **kwargs):
        """ !!. """
        # pylint: disable=too-many-statements

        # To dictionaries -> add names - > add defaults
        attribute_dicts = self.apply_nested(self._show_to_dict, attributes)
        attribute_dicts = self.apply_nested(self._show_extract_names, attribute_dicts)
        attribute_dicts = self.apply_nested(self._show_add_load_defaults, attribute_dicts)

        # Actually load data
        load_method = partial(self._show_load_data, method=self.load_attribute)
        attribute_dicts = self.apply_nested(load_method, attribute_dicts)

        # Plot params for attributes
        attribute_dicts = self.apply_nested(self._show_add_plot_defaults, attribute_dicts)
        data = self.apply_nested(lambda dct: dct['output'], attribute_dicts)
        titles = self.apply_nested(lambda dct: dct['name'], attribute_dicts)
        alphas = self.apply_nested(lambda dct: dct['alpha'], attribute_dicts)
        cmaps = self.apply_nested(lambda dct: dct['cmap'], attribute_dicts)

        titles = [item[0] if isinstance(item, list) else item for item in titles]

        # Prepare plot defaults
        plot_defaults = {
            'suptitle_label': f"Field `{self.displayed_name}`",
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
        savepath = self.make_savepath(params['savepath'], name=self.short_name) if savepath is not None else None

        figure = plot_image(data=data, mode=mode, savepath=savepath, **params)
        plt.show()
        return figure if return_figure else None
