""" A mixin with batch visualizations. """
from collections import defaultdict

from ..plotters import plot
from ..utils import DelegatingList, to_list



class VisualizationMixin:
    """ Methods for batch components visualizations. """
    @property
    def default_plot_components(self):
        """ Return a list of default components to plot, that are actually present in batch. """
        components = [['images'], ['masks'], ['images', 'masks'], ['predictions'], ['images', 'predictions']]
        components = [items for items in components if all(hasattr(self, item) for item in items)]
        return components

    def get_layer_config(self, component, layer_index, item_index, zoom,
                         augment_mask, augment_prediction, data_cmap, mask_cmap, mask_color):
        """ Retrieve requested component from batch, preprocess its data and infer its display parameters.

        Component data is obtained by its name and index in batch.
        If `zoom` parameter is provided, component is sliced according to it.
        A default colormap or a color is chosen for component display based on it category (data/mask/prediction).

        Component is treated as mask if it contains 'mask' in its name.
        Component is treated as prediction if it contains 'prediction' in its name and its data is in [0, 1] range.
        Else component is treated as just general data.

        General data is always displayed with `data_cmap`, if `cmap` parameter is not provided explicitly.
        Masks and predictions are in a way different from other data, since they have their own display scenarios.

        Specific scenario depends on whether component augmentation is enabled by `augment_mask`/`augment_prediction`
        and whether component is displayed on a first subplot layers or not.

        Mask/prediction on a first subplot layer is always displayed with `mask_cmap` no matter what.
        Else it is displayed with `mask_cmap` if augmentation is disabled, else it is dispalyed with `mask_color`.

        Parameters
        ----------
        component : str
            Name of component to plot.
        layer_index : int
            Index of the layer a component is displayed upon.
        item_index : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        augment_mask: bool
            If True, hide 0s in binary mask and automatically choose color for 1s.
            Doesn't affect component if it is on a first subplot layer.
        augment_prediction : bool or number from [0, 1]
            If True, mask lower then threshold in prediction component. Threshold is 0.5 if value is True.
            Doesn't affect component if it is on a first subplot layer.
        data_cmap : valid matplotlib colormap
            Colormap to use for general data components display.
        mask_cmap : valid matplotlib colormap
            Colormap to use for masks/predictions components display.
        mask_color : valid matplotib color
            Color to use for masks/predictions components display.
        """
        data = getattr(self, component)[item_index].squeeze()

        if zoom is not None:
            data = data[zoom]

        cmap = data_cmap
        mask = None
        vmin, vmax = None, None

        if 'mask' in component:
            if not augment_mask or layer_index == 0:
                augment_mask = False
                cmap = mask_cmap
                vmin, vmax = 0, 1
            else:
                cmap = mask_color
        elif 'prediction' in component and (data.min() >= 0.0 and data.max() <= 1.0):
            if augment_prediction is False or layer_index == 0:
                cmap = mask_cmap
                vmin, vmax = 0, 1
            else:
                threshold = 0.5 if augment_prediction is True else augment_prediction
                mask = f'<{threshold}'
                cmap = mask_color
                vmin, vmax = threshold, 1

        config = {
            'data': data,
            'cmap': cmap,
            'mask': mask,
            'augment_mask': augment_mask,
            'vmin': vmin,
            'vmax': vmax
        }
        return config

    def get_plot_config(self, components, item_index, zoom, add_suptitle, add_location,
                        augment_mask, augment_prediction, data_cmap, mask_cmap, mask_color):
        """ Retrieve requested components from batch, preprocess their data and infer parameters for their display.

        Component data is obtained by its name and index in batch.
        If `zoom` parameter is provided, component is sliced according to it.
        A default colormap or a color is chosen for component display based on it category (data/mask/prediction).

        Component is treated as mask if it contains 'mask' in its name.
        Component is treated as prediction if it contains 'prediction' in its name and its data is in [0, 1] range.
        Else component is treated as just general data.

        General data is always displayed with `data_cmap`, if `cmap` parameter is not provided explicitly.
        Masks and predictions are in a way different from other data, since they have their own display scenarios.

        Specific scenario depends on whether component augmentation is enabled by `augment_mask`/`augment_prediction`
        and whether component is displayed on a first subplot layers or not.

        Mask/prediction on a first subplot layer is always displayed with `mask_cmap` no matter what.
        Else it is displayed with `mask_cmap` if augmentation is disabled, else it is dispalyed with `mask_color`.

        Parameters
        ----------
        components : double nested list of str
            Names of component to plot arranged in order of their display on subplots/layers.
        item_index : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        add_suptitle : bool
            If True, display suptitle with batch item index, field name (if it exists) and location info (if requested).
        add_location : bool, str or iterable of str
            If True or 'suptitle', add location info to suptitle. If 'title', add location info to title.
            If 'ticks', replace ticks labels from relative (e.g. from 0 to 256) to absolute (e.g. from 1056 to 1312).
            If list, add location info to corresponding annotation objects ('suptitle', 'title', 'ticks').
        augment_mask: bool
            If True, hide 0s in binary mask and automatically choose color for 1s.
            Doesn't affect component if it is on a first subplot layer.
        augment_prediction : bool or number from [0, 1]
            If True, mask lower then threshold in prediction component. Threshold is 0.5 if value is True.
            Doesn't affect component if it is on a first subplot layer.
        data_cmap : valid matplotlib colormap
            Colormap to use for general data components display.
        mask_cmap : valid matplotlib colormap
            Colormap to use for masks/predictions components display.
        mask_color : valid matplotib color
            Color to use for masks/predictions components display.
        """
        # pylint: disable=too-many-statements
        # Make plot config layer-wise
        layers_indices = list(map(lambda item: list(range(len(item))), components))
        config = components.map(self.get_layer_config, layers_indices, item_index=item_index, zoom=zoom,
                                augment_mask=augment_mask, augment_prediction=augment_prediction,
                                data_cmap=data_cmap, mask_cmap=mask_cmap, mask_color=mask_color)
        config = config.to_dict()

        # Infer slide extent from its location and zoom
        location = self.locations[item_index]
        labels = ['INLINE', 'CROSSLINE', 'DEPTH']
        for x, y, z in [[0, 1, 2], [0, 2, 1], [1, 2, 0]]:
            if location[z].stop - location[z].start == 1:
                x_label = labels[x]
                x_start, x_stop = location[x].start, location[x].stop

                y_label = labels[y]
                y_start, y_stop = location[y].start, location[y].stop

                z_label = labels[z]
                z_start =  location[z].start

                if zoom is not None:
                    x_zoom, y_zoom = zoom

                    if x_zoom.start is not None:
                        x_start = x_start + x_zoom.start
                    if x_zoom.stop is not None:
                        if x_zoom.stop >= 0:
                            x_stop = x_start + x_zoom.stop
                        else:
                            x_stop = x_stop + zoom[0].stop + 1

                    if y_zoom.start is not None:
                        y_start = y_start + y_zoom.start
                    if y_zoom.stop is not None:
                        if y_zoom.stop >= 0:
                            y_stop = y_stop + y_zoom.stop
                        else:
                            y_stop = y_stop + y_zoom.stop + 1
                break
        else:
            raise ValueError("Data must be 2D or pseudo-3D.")

        # Annotate x and y axes
        config['xlabel'] = x_label
        config['ylabel'] = y_label
        if 'ticks' in add_location:
            config['extent'] = (x_start, x_stop, y_stop, y_start)

        location_info = f"{z_label}={z_start}", f"{x_label} <{x_start}:{x_stop}>'", f"{y_label} <{y_start}:{y_stop}>"

        # Construct suptitle
        if add_suptitle:
            suptitle = f'batch item #{item_index}'

            batch_index = self.indices[item_index]
            field = self.get(batch_index, 'fields')
            if hasattr(field, 'displayed_name'):
                suptitle += f' | field `{field.displayed_name}`'

            if 'suptitle' in add_location:
                suptitle += '\n' + '   '.join(location_info)
            config['suptitle'] = suptitle

        # Make titles for individual axis
        title = [', '.join(item) for item in components]
        if 'title' in add_location:
            for axis_index, axis_info in enumerate(location_info[:len(title)]):
                title[axis_index] += '\n' + axis_info
        config['title'] = title

        return config

    def plot(self, components=None, idx=0, zoom=None, add_suptitle=True, add_location='suptitle', augment_mask=True,
             augment_prediction=True, data_cmap='Greys_r', mask_cmap='gist_heat', mask_color='darkorange', **kwargs):
        """ Plot requested batch components.

        Parameters
        ----------
        components : double nested list of str
            Names of component to plot arranged in order of their display on subplots/layers.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        add_suptitle : bool
            If True, display suptitle with batch item index, field name (if it exists) and location info (if requested).
        add_location : bool, str or iterable of str
            If True or 'suptitle', add location info to suptitle. If 'title', add location info to title.
            If 'ticks', replace ticks labels from relative (e.g. from 0 to 256) to absolute (e.g. from 1056 to 1312).
            If list, add location info to corresponding annotation objects ('suptitle', 'title', 'ticks').
        augment_mask: bool
            If True, hide 0s in binary mask and automatically choose color for 1s.
            Doesn't affect component if it is on a first subplot layer.
        augment_prediction : bool or number from [0, 1]
            If True, mask lower then threshold in prediction component. Threshold is 0.5 if value is True.
            Doesn't affect component if it is on a first subplot layer.
        data_cmap : valid matplotlib colormap
            Colormap to use for general data components display.
        mask_cmap : valid matplotlib colormap
            Colormap to use for masks/predictions components display.
        mask_color : valid matplotib color
            Color to use for masks/predictions components display.
        kwargs : misc
            For `batchflow.plot`.
        """
        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [[components]]
        else:
            components = [to_list(item) for item in components]
        components = DelegatingList(components)

        if add_location is False:
            add_location = []
        elif add_location is True:
            add_location = ['suptitle']
        else:
            add_location = to_list(add_location)

        config = self.get_plot_config(components=components, item_index=idx, zoom=zoom,
                                      add_suptitle=add_suptitle, add_location=add_location,
                                      augment_mask=augment_mask, augment_prediction=augment_prediction,
                                      data_cmap=data_cmap, mask_cmap=mask_cmap, mask_color=mask_color)

        config = {
            'scale': 0.8,
            **config,
            **kwargs
        }

        if 'ncols' not in config and 'nrows' not in config:
            config['ncols'] = len(config['data'])

        return plot(**config)

    def plot_roll(self, n=1, components=None, indices=None, zoom=None,
                  add_location='title', augment_mask=True, augment_prediction=True,
                  data_cmap='Greys_r', mask_cmap='gist_heat', mask_color='darkorange', **kwargs):
        """ Plot requested components of batch items with specified indices if provided, else choose indices randomly.

        Parameters
        ----------
        n : int
            Number of batch indices to sample. Not used, when `indices` provided.
        components : double nested list of str
            Names of component to plot arranged in order of their display on subplots/layers.
        indices : int or list of int
            Indices of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        add_location : bool, str or iterable of str
            If True or 'title', add location info to title.
            If 'ticks', replace ticks labels from relative (e.g. from 0 to 256) to absolute (e.g. from 1056 to 1312).
            If list, add location info to corresponding annotation objects ('title', 'ticks').
        augment_mask: bool
            If True, hide 0s in binary mask and automatically choose color for 1s.
            Doesn't affect component if it is on a first subplot layer.
        augment_prediction : bool or number from [0, 1]
            If True, mask lower then threshold in prediction component. Threshold is 0.5 if value is True.
            Doesn't affect component if it is on a first subplot layer.
        data_cmap : valid matplotlib colormap
            Colormap to use for general data components display.
        mask_cmap : valid matplotlib colormap
            Colormap to use for masks/predictions components display.
        mask_color : valid matplotib color
            Color to use for masks/predictions components display.
        kwargs : misc
            For `batchflow.plot`.
        """
        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [[components]]
        else:
            components = [to_list(item) for item in components]
        components = DelegatingList(components)

        if add_location is False:
            add_location = []
        elif add_location is True:
            add_location = ['title']
        else:
            add_location = to_list(add_location)

        if indices is None:
            indices = self.random.choice(len(self), size=min(n, len(self)), replace=False)
        else:
            indices = to_list(indices)

        config = defaultdict(list)
        for idx in indices:
            config_idx = self.get_plot_config(components=components, item_index=idx, zoom=zoom,
                                              add_suptitle=False, add_location=add_location,
                                              augment_mask=augment_mask, augment_prediction=augment_prediction,
                                              data_cmap=data_cmap, mask_cmap=mask_cmap, mask_color=mask_color)
            for name, value in config_idx.items():
                if not isinstance(value, list):
                    value = [value] * len(components)
                config[name].extend(value)

        config = {
            'scale': 0.8,
            **config,
            **kwargs
        }

        if 'ncols' not in config and 'nrows' not in config:
            config['ncols'] = len(components)

        return plot(**config)
