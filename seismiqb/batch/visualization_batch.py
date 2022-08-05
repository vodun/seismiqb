""" A mixin with batch visualizations. """
from collections import defaultdict
import numpy as np
from scipy.fft import rfftfreq, rfft

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

    def get_plot_config(self, components, idx, zoom, displayed_name, augment_title, augment_prediction):
        """ Get batch components data for specified index and make its plot config.

        Parameters
        ----------
        components : sequence
            List of components names to plot.
        idx : int
            Index of batch component to retrieve.
        zoom : tuple of two slices
            Additional limits to show batch components in.
        displayed_name : str
            Field name to show in suptitle instead of default.
        augment_title : bool
            Whether add data location string representation to titles or not.
        augment_prediction : bool
            If True, hide values smaller than 0.5 in 'predictions' batch component.
        """
        #pylint: disable=too-many-nested-blocks
        components = DelegatingList(components)

        data = components.apply(self.get_component_data, idx=idx, zoom=zoom)
        cmap = components.apply(lambda item: 'darkorange' if 'mask' in item or 'prediction' in item else 'Greys_r')

        if augment_prediction:
            for i, component in enumerate(components):
                for j, component_ in enumerate(component):
                    if 'prediction' in component_:
                        data_ = data[i][j]
                        if data_.min() >= 0.0 and data_.max() <= 1.0:
                            data_ = np.ma.array(data_, mask=data_ < 0.5)
                            data[i][j] = data_

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
        title = [', '.join(item) for item in components]
        # TODO: Replace with `set_xticklabels` parametrization
        if augment_title:
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
        components = [['images'], ['masks'], ['images', 'masks'], ['predictions'], ['images', 'predictions']]
        components = DelegatingList(components)
        components = components.filter(lambda item: all(hasattr(self, subitem) for subitem in item), shallow=True)
        return components

    def plot(self, components=None, idx=0, zoom=None, displayed_name=None,
             augment_title=False, augment_mask=True, augment_prediction=True, **kwargs):
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
        augment_title : bool
            If True, add data location string representation to titles.
        augment_mask: bool
            If True, hide 0s in binary mask and automatically choose color for 1s.
        augment_prediction : bool
            If True, hide values smaller than 0.5 in 'predictions' batch component.
        """
        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [[components]]
        else:
            components = [to_list(item) for item in components]

        plot_config = self.get_plot_config(components=components, idx=idx, zoom=zoom, displayed_name=displayed_name,
                                           augment_title=augment_title, augment_prediction=augment_prediction)

        plot_config = {
            'scale': 0.8,
            'augment_mask': augment_mask,
            **plot_config,
            **kwargs
        }

        if 'ncols' not in plot_config and 'nrows' not in plot_config:
            plot_config['ncols'] = len(components)

        return plot(**plot_config)

    def plot_roll(self, n=1, components=None, indices=None, zoom=None, displayed_name=None,
                  augment_title=True, augment_mask=True, augment_prediction=True, **kwargs):
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
        augment_title : bool
            Whether add data location string representation to titles or not.
        augment_mask: bool
            If True, hide 0s in binary mask and automatically choose color for 1s.
        augment_prediction : bool
            If True, hide values smaller than 0.5 in 'predictions' batch component.
        """
        if indices is None:
            indices = self.random.choice(len(self), size=min(n, len(self)), replace=False)
        else:
            indices = to_list(indices)

        if components is None:
            components = self.default_plot_components
        elif isinstance(components, str):
            components = [[components]]
        else:
            components = [to_list(item) for item in components]


        plot_config = defaultdict(list)
        for idx in indices:
            plot_config_idx = self.get_plot_config(components=components, idx=idx, zoom=zoom,
                                                   displayed_name=displayed_name, augment_title=augment_title,
                                                   augment_prediction=augment_prediction)
            _ = plot_config_idx.pop('suptitle')

            for name, value in plot_config_idx.items():
                plot_config[name].extend(value)

        plot_config = {
            'scale': 0.8,
            'augment_mask': augment_mask,
            **plot_config,
            **kwargs
        }

        if 'ncols' not in plot_config and 'nrows' not in plot_config:
            plot_config['ncols'] = len(components)

        return plot(**plot_config)

    def plot_frequencies(self, indices=(0, ), src='images', trace_indices=((0, 0), (-1, -1)), axis=2,
                         sample_spacing=None, displayed_name=None, **kwargs):
        """ Show Fourier frequency spectrum of a component. X-axis of the plot corresponds to frequency
        values in Hz while y-axis stands for amplitudes of specific frequencies.

        The method selects for the analysis specific traces and batch items. Traces are 1d slices
        taken along an axis.

        Uses units of Nyquist frequency when depicting the spectrum. In this way achieves consistency with
        action `bandpass_filter`.

        Parameters
        ----------
        indices : int or sequence of int
            Takes items with these indices to demonstrate the spectrum.
        src : str
            The component, which spectrum is shown.
        trace_indices : sequence of tuples
            Uses traces with these indices.
        axis : int
            Axis along which traces are taken. By default set to 2. This value correpsonds
            to depth, which is the most natural direction to research the spectrum.
        sample_spacing : float or None
            Inverse of the sampling rate. The same argument that `scipy.fftpack.fftfreq` uses under the
            name of `d`. Specifies units of the x-axis of the spectrum plot. If set to None, `show_frequencies`
            uses Nyquist units. That is, `sample_spacing` is set to inverse of the nyquist frequency.
            In this case, x-axis units correspond to units of `lowcut`/ `highcut` agruments of
            `SeismicCropBatch.bandpass_filter`.

            NOTE: in `Seismiqb` the same thing is called `sample_rate`. Measured not in Hz but in ms.
        displayed_name : str or None
            Whenever supplied, assumes that traces are taken from field with this name.
        kwargs : dict
            Arguments for customizing plot. For instance, removing/changing labels and titles.
        """
        indices = indices if isinstance(indices, (list, tuple)) else (indices, )
        insert_index = len(trace_indices[0]) if axis == -1 else axis
        plot_data = []
        plot_label = []

        # Iterate over item-indices and traces, gather info about spectrum.
        for idx in indices:
            field = self.get(self.indices[idx], 'fields')
            sample_spacing = sample_spacing or field.sample_rate / 1000  # field.sample_rate is given in ms

            # Try to get the name of a field
            if displayed_name is None:
                displayed_name = field.displayed_name

            data = self.get(self.indices[idx], src)
            frequencies = rfftfreq(data.shape[axis], sample_spacing)

            for trace_idx in trace_indices:
                trace_idx_ = tuple(np.insert(np.array(trace_idx, dtype=np.object_), insert_index, slice(0, None)))
                amplitudes = rfft(data[trace_idx_])
                plot_data.append((frequencies, np.abs(amplitudes)))
                plot_label.append(f'FIELD: {displayed_name}   IDX: {idx}   TRACE: {trace_idx}')

        plot_params = {'title': f'Spectrum of {src}-component',
                       'label': plot_label,
                       'xlabel': 'Frequency, HZ',
                       'ylabel': 'Amplitude'}
        kwargs = {
            **plot_params,
            **kwargs
        }
        return plot(plot_data, mode='curve', **kwargs)
