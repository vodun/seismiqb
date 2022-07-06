""" A mixin with batch visualizations. """
from collections import defaultdict

from ..plotters import plot
from ..utils import to_list, DelegatingList



class VisualizationMixin:
    """ Methods for batch components visualizations. """
    def get_component_data(self, component, idx, zoom):
        """ Get component data from batch by name and index, optionally slicing it.

        Parameters
        ----------
        component : str
            Name of batch component to retrieve.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        """
        data = getattr(self, component)[idx].squeeze()

        if zoom is not None:
            data = data[zoom]

        return data

    def get_plot_config(self, components, idx, zoom, displayed_name, augment_titles):
        """ Get batch components data for specified index and make its plot config.

        Parameters
        ----------
        components : sequence
            List of components names to plot.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        diplayed_name : str
            Field name to show in suptitle instead of default.
        augment_titles : bool
            Whether add data location string representation to titles or not.
        """
        components = DelegatingList(components)

        data = components.apply(self.get_component_data, idx=idx, zoom=zoom)
        cmap = components.apply(lambda item: 'Reds' if 'mask' in item or 'prediction' in item else 'Greys_r')

        # Extract location
        location = self.locations[idx]
        i_start, i_end = location[0].start, location[0].stop
        x_start, x_end = location[1].start, location[1].stop
        h_start, h_end = location[2].start, location[2].stop

        # Make suptitle and axis labels
        if (i_end - i_start) == 1:
            location_description = f'INLINE={i_start}', f'CROSSLINES <{x_start}:{x_end}>', f'DEPTH <{h_start}:{h_end}>'
            xlabel, ylabel = 'CROSSLINE', 'HEIGHT'
        elif (x_end - x_start) == 1:
            location_description = f'CROSSLINE={x_start}', f'INLINES <{i_start}:{i_end}>', f'DEPTH <{h_start}:{h_end}>'
            xlabel, ylabel = 'INLINE', 'HEIGHT'
        else:
            location_description = f'DEPTH={h_start}', f'INLINES <{i_start}:{i_end}>', f'CROSSLINES <{x_start}:{x_end}>'
            xlabel, ylabel = 'INLINE', 'CROSSLINE'
        suptitle = '   '.join(location_description)

        xlabel = [xlabel] * len(components)
        ylabel = [ylabel] * len(components)

        # Try to get the name of a field
        if displayed_name is None:
            batch_index = self.indices[idx]
            displayed_name = self.get(batch_index, 'fields').displayed_name
        suptitle = f'batch_idx={idx}                  `{displayed_name}`\n{suptitle}'

        # Titles for individual axis
        title = [str(item) for item in components]
        # TODO: Replace with `set_xticklabels` parametrization
        if augment_titles:
            if len(components) >= 1:
                title[0] += '\n' + location_description[0]
            if len(components) >= 2:
                title[1] += '\n' + location_description[1]
            if len(components) >= 3:
                title[2] += '\n' + location_description[2]

        plot_config = {
            'data': data,
            'cmap': cmap,
            'suptitle': suptitle,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
        }

        return plot_config

    @property
    def default_plot_components(self):
        """ Return a list of default components to plot, that are actually present in batch. """
        components = ['images', 'masks', ['images', 'masks'], 'predictions', ['images', 'predictions']]
        components = DelegatingList(components)
        component_present = lambda item: hasattr(self, item) if isinstance(item, str) \
                                         else all(hasattr(self, subitem) for subitem in item)
        components = components.filter(component_present, shallow=True)
        return components

    def plot(self, components=None, idx=0, zoom=None, displayed_name=None, augment_titles=False, **kwargs):
        """ Plot components of batch for specific index.

        Parameters
        ----------
        components : str or sequence
            Component(s) names to plot.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        diplayed_name : str
            Field name to show in suptitle instead of default.
        augment_titles : bool
            Whether add data location string representation to titles or not.
        """
        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [components]

        plot_config = self.get_plot_config(components=components, idx=idx,
                                           zoom=zoom, displayed_name=displayed_name, augment_titles=augment_titles)

        plot_config = {
            'scale': 0.8,
            'separate': not max([isinstance(item, list) for item in components]),
            **plot_config,
            **kwargs
        }

        if 'ncols' not in plot_config and 'nrows' not in plot_config:
            plot_config['ncols'] = len(components)

        return plot(**plot_config)

    def plot_roll(self, n=1, components=None, indices=None, zoom=None,
                  displayed_name=None, augment_titles=True, **kwargs):
        """ Plot `n` random batch items on one figure.

        Parameters
        ----------
        n : int
            Number of batch indices to sample. Not used, when `indices` provided.
        components : str or sequence
            Component(s) names to plot.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        diplayed_name : str
            Field name to show in suptitle instead of default.
        augment_titles : bool
            Whether add data location string representation to titles or not.
        """
        if indices is None:
            indices = self.random.choice(len(self), size=min(n, len(self)), replace=False)
        else:
            indices = to_list(indices)

        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [components]

        plot_config = defaultdict(list)
        for idx in indices:
            plot_config_idx = self.get_plot_config(components=components, idx=idx,
                                                   zoom=zoom, displayed_name=displayed_name, augment_titles=augment_titles)
            _ = plot_config_idx.pop('suptitle')

            for name, value in plot_config_idx.items():
                plot_config[name].extend(value)

        plot_config = {
            'scale': 0.8,
            'separate': not max([isinstance(item, list) for item in components]),
            **plot_config,
            **kwargs
        }

        if 'ncols' not in plot_config and 'nrows' not in plot_config:
            plot_config['ncols'] = len(components)

        return plot(**plot_config)
