""" A mixin with batch visualizations. """
from collections import defaultdict

import numpy as np
from cv2 import dilate as dilation

from ..plotters import plot
from ..utils import to_list, DelegatingList



class VisualizationMixin:
    """ Methods for batch components visualizations. """
    def get_component_data(self, component, idx, zoom, dilate, clip):
        """ Get component data from batch by name and index, optionally slice it, clip by threshold and dilate.

        Parameters
        ----------
        component : str
            Name of batch component to retrieve.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        dilate : str, sequence or dict
            Controls components data enlargment.
            Default components to dilate are ('masks', 'predictions') and default dilation kernel is [1, 1, 1].
            If a bool, controls whether default components should be dilated or not with default kernel.
            If an int, control number of times default components should be dilated with default kernel.
            If a tuple of two ints, defines a shape of kernel filled with ones to use for default components dilation.
            If str or sequence of str, must specify components name(s) which data should be dilated.
            If a dict, keys must be either int/tuple with same meaning as above or a dict of params for `cv2.dilate`.
        clip : str, sequence or dict
            Controls masking of components data lower than threshold.
            Default clip component is 'predictions' and default clip threshold is 0.5.
            If bool, toggles clipping for default components data by default threshold.
            If float, used as a threshold for clipping of default components data.
            If str or sequence of str, must specify components names which data should be clipped by default threshold.
            If a dict, keys must specify components names which data should be clipped and values should specify
            corresponding clip thresholds.
        """
        data = getattr(self, component)[idx].squeeze()

        if zoom is not None:
            data = data[zoom]

        default_kernel = np.ones((1, 3), dtype=np.uint8)
        if dilate is False:
            dilate = {}
        elif dilate is True:
            dilate = {'masks': default_kernel, 'predictions': default_kernel}
        elif isinstance(dilate, int):
            dilate = {'masks': dilate, 'predictions': dilate}
        elif isinstance(dilate, tuple):
            dilate = {'masks': dilate, 'predictions': dilate}
        elif isinstance(dilate, str):
            dilate = {dilate: default_kernel}
        elif isinstance(dilate, list):
            dilate = {component: default_kernel for component in dilate}

        if component in dilate:
            dilation_config = dilate[component]
            if isinstance(dilation_config, int):
                dilation_config = {'iterations': dilation_config, 'kernel': default_kernel}
            if isinstance(dilation_config, tuple):
                dilation_config = {'iterations': 1, 'kernel': np.ones(dilation_config, dtype=np.uint8)}
            elif isinstance(dilation_config, (str, np.ndarray)):
                dilation_config = {'iterations': 1, 'kernel': dilation_config}

            data = dilation(data, **dilation_config)

        default_threshold = 0.5
        if clip is False:
            clip = {}
        elif clip is True:
            clip = {'predictions': default_threshold}
        elif isinstance(clip, float):
            clip = {'predictions': clip}
        elif isinstance(clip, str):
            clip = {clip: default_threshold}
        elif isinstance(clip, list):
            clip = {component: default_threshold for component in clip}

        if component in clip:
            threshold = clip[component]
            data = np.ma.masked_array(data, data < threshold)

        return data

    def get_plot_config(self, components, idx, zoom, dilate, clip, displayed_name, augment_titles):
        """ Get batch components data for specified index and make its plot config.

        Parameters
        ----------
        components : sequence
            List of components names to plot.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        dilate : str, sequence or dict
            Controls components data enlargment.
            Default components to dilate are ('masks', 'predictions') and default dilation kernel is [1, 1, 1].
            If a bool, controls whether default components should be dilated or not with default kernel.
            If an int, control number of times default components should be dilated with default kernel.
            If a tuple of two ints, defines a shape of kernel filled with ones to use for default components dilation.
            If str or sequence of str, must specify components name(s) which data should be dilated.
            If a dict, keys must be either int/tuple with same meaning as above or a dict of params for `cv2.dilate`.
        clip : str, sequence or dict
            Controls masking of components data lower than threshold.
            Default clip component is 'predictions' and default clip threshold is 0.5.
            If bool, toggles clipping for default components data by default threshold.
            If float, used as a threshold for clipping of default components data.
            If str or sequence of str, must specify components names which data should be clipped by default threshold.
            If a dict, keys must specify components names which data should be clipped and values should specify
            corresponding clip thresholds.
        diplayed_name : str
            Field name to show in suptitle instead of default.
        augment_titles : bool
            Whether add data location string representation to titles or not.
        """
        components = DelegatingList(components)

        data = components.apply(self.get_component_data, idx=idx, clip=clip, dilate=dilate, zoom=zoom)
        cmap = components.apply(lambda item: 'Reds' if 'mask' in item or 'prediction' in item else 'Greys_r')

        # TODO: Remove on plotter update
        cmap = cmap.apply(lambda item: item if isinstance(item, list) else [item], shallow=True)

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

    def plot(self, components=None, idx=0, zoom=None, dilate=False, clip=False,
             displayed_name=None, augment_titles=False, **kwargs):
        """ Plot components of batch for specific index.

        Parameters
        ----------
        components : str or sequence
            Component(s) names to plot.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        dilate : str, sequence or dict
            Controls components data enlargment.
            Default components to dilate are ('masks', 'predictions') and default dilation kernel is [1, 1, 1].
            If a bool, controls whether default components should be dilated or not with default kernel.
            If an int, control number of times default components should be dilated with default kernel.
            If a tuple of two ints, defines a shape of kernel filled with ones to use for default components dilation.
            If str or sequence of str, must specify components name(s) which data should be dilated.
            If a dict, keys must be either int/tuple with same meaning as above or a dict of params for `cv2.dilate`.
        clip : str, sequence or dict
            Controls masking of components data lower than threshold.
            Default clip component is 'predictions' and default clip threshold is 0.5.
            If bool, toggles clipping for default components data by default threshold.
            If float, used as a threshold for clipping of default components data.
            If str or sequence of str, must specify components names which data should be clipped by default threshold.
            If a dict, keys must specify components names which data should be clipped and values should specify
            corresponding clip thresholds.
        diplayed_name : str
            Field name to show in suptitle instead of default.
        augment_titles : bool
            Whether add data location string representation to titles or not.

        Clipping and dilation examples
        ------------------------
        - Dilate 'masks' and 'predictions' (default components to enlarge when `dilate=True`):
        >>> batch.plot(['masks', 'predictions'], dilate=True)

        - Dilate 'predictions' 3 times (again, applied to default components only):
        >>> batch.plot(['images', 'predictions'], dilate=3)

        - Dilate 'masks' and clip 'predictions':
        >>> batch.plot(['images', 'masks', 'predictions'], dilate='masks', clip='predictions')

        - Dilate 'masks' with kernel `np.ones([1, 1, 1])` and clip 'predictions' by 0.7 threshold:
        >>> batch.plot(['masks', 'predictions'], clip={'predictions': 0.7}, dilate={'masks': (3, 1)})

        - Dilate 'predictions' with custom `kernel` 10 times:
        >>> batch.plot('predictions', dilate={'predictions': {'kernel': kernel, 'iterations': 10}})
        """
        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [components]

        plot_config = self.get_plot_config(components=components, idx=idx,
                                           zoom=zoom, dilate=dilate, clip=clip,
                                           displayed_name=displayed_name, augment_titles=augment_titles)

        plot_config = {
            'scale': 0.8,
            'ncols': len(components),
            'separate': not max([isinstance(item, list) for item in components]),
            **plot_config,
            **kwargs
        }
        return plot(**plot_config)

    def roll_plot(self, n=1, components=None, indices=None, zoom=None, dilate=False, clip=False,
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
        dilate : str, sequence or dict
            Controls components data enlargment.
            Default components to dilate are ('masks', 'predictions') and default dilation kernel is [1, 1, 1].
            If a bool, controls whether default components should be dilated or not with default kernel.
            If an int, control number of times default components should be dilated with default kernel.
            If a tuple of two ints, defines a shape of kernel filled with ones to use for default components dilation.
            If str or sequence of str, must specify components name(s) which data should be dilated.
            If a dict, keys must be either int/tuple with same meaning as above or a dict of params for `cv2.dilate`.
        clip : str, sequence or dict
            Controls masking of components data lower than threshold.
            Default clip component is 'predictions' and default clip threshold is 0.5.
            If bool, toggles clipping for default components data by default threshold.
            If float, used as a threshold for clipping of default components data.
            If str or sequence of str, must specify components names which data should be clipped by default threshold.
            If a dict, keys must specify components names which data should be clipped and values should specify
            corresponding clip thresholds.
        diplayed_name : str
            Field name to show in suptitle instead of default.
        augment_titles : bool
            Whether add data location string representation to titles or not.

        Clipping and dilation examples
        ------------------------
        - Dilate 'masks' and 'predictions' (default components to enlarge when `dilate=True`):
        >>> batch.plot(['masks', 'predictions'], dilate=True)

        - Dilate 'predictions' 3 times (again, applied to default components only):
        >>> batch.plot(['images', 'predictions'], dilate=3)

        - Dilate 'masks' and clip 'predictions':
        >>> batch.plot(['images', 'masks', 'predictions'], dilate='masks', clip='predictions')

        - Dilate 'masks' with kernel `np.ones([1, 1, 1])` and clip 'predictions' by 0.7 threshold:
        >>> batch.plot(['masks', 'predictions'], clip={'predictions': 0.7}, dilate={'masks': (3, 1)})

        - Dilate 'predictions' with custom `kernel` 10 times:
        >>> batch.plot('predictions', dilate={'predictions': {'kernel': kernel, 'iterations': 10}})
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
                                                   zoom=zoom, dilate=dilate, clip=clip,
                                                   displayed_name=displayed_name, augment_titles=augment_titles)
            _ = plot_config_idx.pop('suptitle')

            for name, value in plot_config_idx.items():
                plot_config[name].extend(value)

        plot_config = {
            'scale': 0.8,
            'ncols': len(components),
            'separate': not max([isinstance(item, list) for item in components]),
            **plot_config,
            **kwargs
        }

        return plot(**plot_config)
