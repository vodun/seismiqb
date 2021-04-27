""" Plot functions. """
from copy import copy
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, register_cmap
from matplotlib.patches import Patch
from matplotlib.colors import ColorConverter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits import axes_grid1

import plotly.figure_factory as ff
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .utils import to_list



CDICT = {
    'red': [[0.0, None, 1.0], [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 0.0, None]],
    'green': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 1.0, 1.0], [1.0, 0.5, None]],
    'blue': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, None]]
}
METRIC_CMAP = LinearSegmentedColormap('Metric', CDICT)
METRIC_CMAP.set_bad(color='black')
register_cmap(name='Metric', cmap=METRIC_CMAP)

DEPTHS_CMAP = ListedColormap(get_cmap('viridis_r')(np.linspace(0.0, 0.5, 100)))
register_cmap(name='Depths', cmap=DEPTHS_CMAP)





CDICT = {
    'red': [[0.0, None, 1.0], [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 0.0, None]],
    'green': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 1.0, 1.0], [1.0, 0.5, None]],
    'blue': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, None]]
}
METRIC_CMAP = LinearSegmentedColormap('Metric', CDICT)
METRIC_CMAP.set_bad(color='black')

DEPTHS_CMAP = ListedColormap(get_cmap('viridis_r')(np.linspace(0.0, 0.5, 100)))



def color_to_cmap(color):
    if isinstance(color, str):
        color = ColorConverter().to_rgb(color)
    return ListedColormap(color)


def add_colorbar(image, aspect=30, fraction=0.5, color='black', **kwargs):
    divider = axes_grid1.make_axes_locatable(image.axes)
    width = axes_grid1.axes_size.AxesY(image.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(fraction, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    colorbar = image.axes.figure.colorbar(image, cax=cax, **kwargs)
    colorbar.ax.yaxis.set_tick_params(color=color)


def add_legend(axis, color, label, size=20):
    handles = getattr(axis.get_legend(), 'legendHandles', [])
    new_patch = Patch(color=color, label=label)
    handles.append(new_patch)
    axis.legend(handles=handles, loc=0, prop={'size': size})


def channelize_image(image, total_channels, color=None, greyscale=False, opacity=None):
    """ Channelize an image. Can be used to make an opaque rgb or grayscale image.
    """
    # case of a partially channelized image
    if image.ndim == 3:
        if image.shape[-1] == total_channels:
            return image

        background = np.zeros((*image.shape[:-1], total_channels))
        background[:, :, :image.shape[-1]] = image

        if opacity is not None:
            background[:, :, -1] = opacity
        return background

    # case of non-channelized image
    if isinstance(color, str):
        color = ColorConverter().to_rgb(color)
    background = np.zeros((*image.shape, total_channels))
    for i, value in enumerate(color):
        background[:, :, i] = image * value

    # in case of greyscale make all 3 channels equal to supplied image
    if greyscale:
        for i in range(3):
            background[:, :, i] = image

    # add opacity if needed
    if opacity is not None:
        background[:, :, -1] = opacity * (image != 0).astype(int)

    return background


def filter_kwargs(kwargs, keys, index=slice(None), prefix=''):
    """ Filter the dict of kwargs leaving only supplied keys. """
    result = {}
    for key in keys:
        value = kwargs.get(prefix + key, kwargs.get(key))
        if isinstance(value, (tuple, list, np.ndarray)):
            result[key] = value[index]
        elif value is not None:
            result[key] = value
    return result


def plot_image(image, mode='single', backend='matplotlib', **kwargs):
    """ Overall plotter function, converting kwarg-names to match chosen backend and redirecting
    plotting task to one of the methods of backend-classes.
    """
    if backend in ('matplotlib', 'plt', 'mpl', 'm', 'mp'):
        return getattr(MatplotlibPlotter, mode)(image, **kwargs)
    if backend in ('plotly', 'go'):
        return getattr(PlotlyPlotter, mode)(image, **kwargs)
    raise ValueError('{} backend is not supported!'.format(backend))


def plot_loss(*data, title=None, **kwargs):
    """ Shorthand for loss plotting. """
    kwargs = {
        'xlabel': 'Iterations',
        'ylabel': 'Loss',
        'label': title or 'Loss graph',
        **kwargs
    }
    plot_image(data, mode='curve', backend='mpl', **kwargs)


class MatplotlibPlotter:
    """ Plotting backend for matplotlib. """
    @staticmethod
    def color_to_cmap(color):
        if isinstance(color, str):
            color = ColorConverter().to_rgb(color)
        return ListedColormap(color)

    @staticmethod
    def add_colorbar(image, aspect=30, fraction=0.5, color='black', **kwargs):
        divider = axes_grid1.make_axes_locatable(image.axes)
        width = axes_grid1.axes_size.AxesY(image.axes, aspect=1./aspect)
        pad = axes_grid1.axes_size.Fraction(fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        colorbar = image.axes.figure.colorbar(image, cax=cax, **kwargs)
        colorbar.ax.yaxis.set_tick_params(color=color)

    @staticmethod
    def add_legend(axis, color, label, size=20):
        handles = getattr(axis.get_legend(), 'legendHandles', [])
        new_patch = Patch(color=color, label=label)
        handles.append(new_patch)
        axis.legend(handles=handles, loc=0, prop={'size': size})

    @staticmethod
    def convert_kwargs(mode, kwargs):
        """ Make a dict of kwargs to match matplotlib-conventions: update keys of the dict and
        values in some cases.
        """
        # make conversion-dict for kwargs-keys
        if mode in ['single', 'rgb', 'overlap', 'histogram', 'curve', 'histogram']:
            keys_converter = {'title': 'label', 't':'label'}
        elif mode == 'separate':
            keys_converter = {'title': 't', 'label': 't'}
        elif mode == 'grid':
            keys_converter = {'title': 't'}

        keys_converter = {
            **keys_converter,
            'zmin': 'vmin', 'zmax': 'vmax',
            'xaxis': 'xlabel', 'yaxis': 'ylabel'
        }

        # make new dict updating keys and values
        converted = {}
        for key, value in kwargs.items():
            if key in keys_converter:
                new_key = keys_converter[key]
                if key in ['xaxis', 'yaxis']:
                    converted[new_key] = value.get('title_text', '')
                else:
                    converted[new_key] = value
            else:
                converted[key] = value
        return converted

    @staticmethod
    def save_and_show(fig, show=True, savepath=None, return_figure=False, pyqt=False, **kwargs):
        """ Save and show plot if needed.
        """
        if pyqt:
            return None
        save_kwargs = dict(bbox_inches='tight', pad_inches=0, dpi=100)
        save_kwargs.update(kwargs.get('save', dict()))

        # save if necessary and render
        if savepath is not None:
            fig.savefig(savepath, **save_kwargs)
        if show:
            fig.show()
        else:
            plt.close()

        if return_figure:
            return fig
        return None

    @classmethod
    def single(cls, image, **kwargs):
        """ Plot single image/heatmap using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            2d-array for plotting.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            vmin : float
                the lowest brightness-level to be rendered.
            vmax : float
                the highest brightness-level to be rendered.
            cmap : str
                colormap of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            alpha : float
                transparency-level of the rendered image
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('single', kwargs)
        # update defaults
        defaults = {'figsize': (12, 7),
                    'cmap': DEPTHS_CMAP,
                    'fontsize': 20,
                    'fraction': 0.022, 'pad': 0.07,
                    'labeltop': True, 'labelright': True, 'direction': 'inout',
                    'facecolor': 'white',
                    'label': '', 'title': '', 'xlabel': '', 'ylabel': '',
                    'order_axes': (1, 0),
                    # colorbar
                    'colorbar': True,
                    'colorbar_fraction': 0.5,
                    'colorbar_aspect': 30}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax', 'alpha', 'interpolation'])
        title_kwargs = filter_kwargs(updated, ['label', 'y', 'fontsize', 'family', 'color'], prefix='title_')
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])
        tick_params = filter_kwargs(updated, ['labeltop', 'labelright', 'labelcolor', 'direction'])
        colorbar_kwargs = filter_kwargs(updated, ['fraction', 'pad'], prefix='colorbar_')

        cm = copy(plt.get_cmap(render_kwargs['cmap']))
        cm.set_bad(color=updated.get('bad_color', updated.get('fill_color', 'white')))
        render_kwargs['cmap'] = cm

        # Create figure and axes
        if 'ax' in kwargs:
            ax = kwargs['ax']
            fig = ax.figure
        else:
            fig, ax = plt.subplots(**figure_kwargs)

        # channelize and plot the image
        img = np.transpose(image.squeeze(), axes=updated['order_axes'])
        xticks, yticks = updated.get('xticks', [0, img.shape[1]]), updated.get('yticks', [img.shape[0], 0])
        extent = [xticks[0], xticks[-1], yticks[0], yticks[-1]]

        ax_img = ax.imshow(img, extent=extent, **render_kwargs)

        # add titles and labels
        ax.set_title(**title_kwargs)
        ax.set_xlabel(**xaxis_kwargs)
        ax.set_ylabel(**yaxis_kwargs)

        if 'xticks' in updated:
            ax.set_xticks(xticks)
        if 'yticks' in updated:
            ax.set_yticks(yticks)

        if updated['colorbar']:
            cls.add_colorbar(ax_img, color=yaxis_kwargs.get('color', 'black'), **colorbar_kwargs)
        ax.set_facecolor(updated['facecolor'])
        ax.tick_params(**tick_params)

        if kwargs.get('disable_axes'):
            ax.set_axis_off()

        return cls.save_and_show(fig, **updated)


    @classmethod
    def wiggle(cls, image, **kwargs):
        """ Make wiggle plot of an image. If needed overlap the wiggle plot with a curve supplied by an
        array of heights.

        Parameters
        ----------
        image : np.ndarray or list
            either 2d-array or a list of 2d-array and a 1d-curve to plot atop the array.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            title : str
                title of the plot.
            reverse : bool
                whether to reverse the plot in y-axis. True by default. In that
                way, uses the same orientation as other modes.
            other
        """
        kwargs = cls.convert_kwargs('wiggle', kwargs)
        defaults = {'figsize': (12, 7),
                    'line_color': 'k',
                    'label': '', 'xlabel': '', 'ylabel': '', 'title': '',
                    'fontsize': 20,
                    'width_multiplier': 2,
                    'xstep': 5,
                    'points_marker': 'ro',
                    'reverse': True}

        # deal with kwargs
        updated = {**defaults, **kwargs}
        line_color, xstep, width_mul, points_marker, reverse = [updated[key] for key in (
            'line_color', 'xstep', 'width_multiplier', 'points_marker', 'reverse')]

        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        label_kwargs = filter_kwargs(updated, ['label', 'y', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        # parse image arg
        with_curve = False
        if isinstance(image, (list, tuple)):
            if len(image) > 1:
                with_curve = True
                image, heights = image[:2]

                # transform height-mask to heights if needed
                if heights.ndim == 2:
                    heights = np.where(heights)[1]
            else:
                image = image[0]

        # Create figure and axes
        if 'ax' in kwargs:
            ax = kwargs['ax']
            fig = ax.figure
        else:
            fig, ax = plt.subplots(**figure_kwargs)

        # add titles and labels
        ax.set_title(**label_kwargs)
        ax.set_xlabel(**xaxis_kwargs)
        ax.set_ylabel(**yaxis_kwargs)

        # Creating wiggle-curves and adding height-points if needed
        xlim_curr = (0, len(image))
        ylim_curr = (0, len(image[0]))
        offsets = np.arange(*xlim_curr, xstep)

        if isinstance(line_color, str):
            line_color = [line_color] * len(offsets)

        y = np.arange(*ylim_curr)
        if reverse:
            y = y[::-1]
        for ix, k in enumerate(offsets):
            x = k + width_mul * image[k, slice(*ylim_curr)] / np.std(image)
            col = line_color[ix]
            ax.plot(x, y, '{}-'.format(col))
            ax.fill_betweenx(y, k, x, where=(x > k), color=col)

            if with_curve:
                ax.plot(x[heights[ix]], heights[ix], points_marker)
            if ix == 0:
                xmin = np.min(x)
            if ix == len(offsets) - 1:
                xmax = np.max(x)

        # adjust the canvas
        xlim = updated.get('xlim', (xmin, xmax))
        ylim = updated.get('ylim', (np.min(y), np.max(y)))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return cls.save_and_show(fig, **updated)

    @classmethod
    def grid(cls, sequence, **kwargs):
        """ Make grid of plots using range of images and info about how the grid should be organized.

        Parameters
        ----------
        sequence : tuple or list of images or dicts
            sequence of either arrays or dicts with kwargs for plotting.
        kwargs : dict
            contains arguments for updating single plots-dicts. Can either contain lists or simple args.
            In case of lists, each subsequent arg is used for updating corresponding single-plot dict.

            label : str
                title of rendered image.
            vmin : float
                the lowest brightness-level to be rendered.
            vmax : float
                the highest brightness-level to be rendered.
            cmap : str
                colormap of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('grid', kwargs)
        defaults = {'figsize': (14, 12),
                    'sharex': 'col',
                    'sharey': 'row',
                    'nrows': 1,
                    'ncols': len(sequence)}

        updated = {**defaults, **kwargs}
        subplots_kwargs = filter_kwargs(updated, ['nrows', 'ncols', 'sharex', 'sharey',
                                                  'figsize', 'fig', 'ax'])
        single_defaults = filter_kwargs(updated, ['label', 'xlabel', 'ylabel', 'cmap',
                                                 'order_axes', 'facecolor', 'fontsize',
                                                 'vmin', 'vmax', 'pad'])
        title_kwargs = filter_kwargs(updated, ['t', 'y', 'fontsize', 'family', 'color'])

        # make and update the sequence of args for plotters if needed
        # make sure that each elem of single_updates is iterable
        single_defaults = {key: value if isinstance(value, (tuple, list)) else [value] * len(sequence)
                           for key, value in single_defaults.items()}

        # make final dict of kwargs for each ax
        single_kwargs = []
        for i, ax_kwargs in enumerate(sequence):
            if isinstance(ax_kwargs, dict):
                single_update = filter_kwargs(ax_kwargs, ['label', 'xlabel', 'ylabel', 'cmap',
                                                          'order_axes', 'facecolor', 'fontsize',
                                                          'vmin', 'vmax', 'pad'])
            else:
                single_update = {}
            single = {**{key: value[i] for key, value in single_defaults.items()}, **single_update}
            single_kwargs.append(single)

        # create axes and make the plots
        nrows, ncols = subplots_kwargs.get('nrows'), subplots_kwargs.get('ncols')
        if nrows is None or ncols is None:
            fig, ax = subplots_kwargs.get('fig'), subplots_kwargs.get('ax')
            if fig is None or ax is None:
                raise ValueError('Either grid params (nrows and ncols) or grid objects (fig, ax) should be supplied.')
        else:
            fig, ax = plt.subplots(**subplots_kwargs)
            ax = ax.reshape((nrows, ncols))
        for i in range(nrows):
            for j in range(ncols):
                if isinstance(sequence[i * ncols + j], dict):
                    image = sequence[i * ncols + j]['image']
                else:
                    image = sequence[i * ncols + j]
                cls.single(image, **single_kwargs[i * ncols + j], ax=ax[i, j])

        fig.suptitle(**title_kwargs)
        return cls.save_and_show(fig)

    @classmethod
    def overlap(cls, images, **kwargs):
        """ Plot several images on one canvas using matplotlib: render the first one in greyscale
        and the rest ones in 'rgb' channels, one channel for each image.
        Supports up to four images in total.

        Parameters
        ----------
        images : tuple or list
            sequence of 2d-arrays for plotting. Supports up to 4 images.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            y : float
                height of the title
            cmap : str
                colormap to render the first image in.
            vmin : float
                the lowest brightness-level to be rendered.
            vmax : float
                the highest brightness-level to be rendered.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('overlap', kwargs)
        defaults = {'figsize': (12, 7),
                    'cmap': 'gray',
                    'fontsize': 20,
                    'color': ('red', 'green', 'blue'),
                    'alpha': 1.0,
                    'label': '', 'title': '', 'xlabel': '', 'ylabel': '',
                    'order_axes': (1, 0),
                    # title
                    'title_y' : 1.1,
                    # colorbar
                    'colorbar': True,
                    'colorbar_fraction': 0.5,
                    'colorbar_aspect': 30,
                    # legend
                    'legend_size': 10}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax', 'interpolation'])
        title_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'y'], prefix='title_')
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family'])
        colorbar_kwargs = filter_kwargs(updated, ['fraction', 'aspect'])

        # Create figure and axes
        if 'ax' in kwargs:
            ax = kwargs['ax']
            fig = ax.figure
        else:
            fig, ax = plt.subplots(**figure_kwargs)

        # channelize images and put them on a canvas
        img = np.transpose(images[0].squeeze(), axes=updated['order_axes'])
        xticks, yticks = updated.get('xticks', [0, img.shape[1]]), updated.get('yticks', [img.shape[0], 0])
        extent = [xticks[0], xticks[-1], yticks[0], yticks[-1]]

        img = ax.imshow(img, extent=extent, **render_kwargs)
        if updated['colorbar']:
            cls.add_colorbar(img, color=yaxis_kwargs.get('color', 'black'), **colorbar_kwargs)
        ax.set_xlabel(**xaxis_kwargs)
        ax.set_ylabel(**yaxis_kwargs)

        if 'xticks' in updated:
            ax.set_xticks(xticks)
        if 'yticks' in updated:
            ax.set_yticks(yticks)

        for i, img in enumerate(images[1:]):
            image = np.transpose(img.squeeze(), axes=updated['order_axes'])
            render_kwargs = filter_kwargs(updated, ['color', 'alpha'], index=i)
            layer_color = render_kwargs.pop('color')
            render_kwargs['cmap'] = cls.color_to_cmap(layer_color)
            ax.imshow(image, extent=extent, **render_kwargs)
            if updated['legend']:
                legend_kwargs = filter_kwargs(updated, ['label', 'size', 'color'], index=i, prefix='legend_')
                cls.add_legend(ax, **legend_kwargs)
        ax.set_title(**title_kwargs)

        return cls.save_and_show(fig, **updated)


    @classmethod
    def hist(cls, images, **kwargs):
        """ Plot several images on one canvas using matplotlib: render the first one in greyscale
        and the rest ones in 'rgb' channels, one channel for each image.
        Supports up to four images in total.

        Parameters
        ----------
        images : tuple or list
            sequence of 2d-arrays for plotting. Supports up to 4 images.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            y : float
                height of the title
            cmap : str
                colormap to render the first image in.
            vmin : float
                the lowest brightness-level to be rendered.
            vmax : float
                the highest brightness-level to be rendered.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        defaults = {'figsize': (12, 7),
                    'fontsize': 20,
                    'color': ('red', 'green', 'blue'),
                    'alpha': 1.0,
                    'label': '', 'title': '', 'xlabel': '', 'ylabel': '',
                    # title
                    'title_y': 1.1,
                    # legend
                    'size': 20}
        updated = {**defaults, **kwargs}


        # form different groups of kwargs
        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        title_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color', 'y'], prefix='title_')
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        # Create figure and axes
        if 'ax' in kwargs:
            ax = kwargs['ax']
            fig = ax.figure
        else:
            fig, ax = plt.subplots(**figure_kwargs)

        for i, img in enumerate(images):
            render_kwargs = filter_kwargs(updated, ['bins', 'color', 'alpha'], index=i)
            ax.hist(img, **render_kwargs)
            if updated['legend']:
                legend_kwargs = filter_kwargs(updated, ['label', 'size', 'color'], index=i, prefix='legend_')
                cls.add_legend(ax, **legend_kwargs)

        ax.set_title(**title_kwargs)
        ax.set_xlabel(**xaxis_kwargs)
        ax.set_ylabel(**yaxis_kwargs)

        return cls.save_and_show(fig, **updated)


    @classmethod
    def rgb(cls, image, **kwargs):
        """ Plot one image in 'rgb' using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            3d-array containing channeled rgb-image.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            label : str
                title of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('rgb', kwargs)
        # update defaults
        defaults = {'figsize': (12, 7),
                    'fontsize': 20,
                    'labeltop': True,
                    'labelright': True,
                    'order_axes': (1, 0, 2)}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])
        tick_params = filter_kwargs(updated, ['labeltop', 'labelright'])

        # channelize and plot the image
        image = channelize_image(image, total_channels=3)
        plt.figure(**figure_kwargs)
        _ = plt.imshow(np.transpose(image.squeeze(), axes=updated['order_axes']), **render_kwargs)

        # add titles and labels
        plt.title(y=1.1, **label_kwargs)
        plt.xlabel(**xaxis_kwargs)
        plt.ylabel(**yaxis_kwargs)
        plt.tick_params(**tick_params)

        return cls.save_and_show(plt, **updated)

    @classmethod
    def separate(cls, images, **kwargs):
        """ Plot several images on a row of canvases using matplotlib.
        TODO: add grid support.

        Parameters
        ----------
        images : tuple or list
            sequence of 2d-arrays for plotting. Supports up to 4 images.
        kwargs : dict
            figsize : tuple
                tuple of two ints containing the size of the rendered image.
            t : str
                overal title of rendered image.
            label : list or tuple
                sequence of titles for each image.
            cmap : str
                colormap to render the first image in.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('separate', kwargs)
        # embedded params
        defaults = {'figsize': (6 * len(images), 7),
                    'cmap': 'gray',
                    'suptitle_fontsize': 20,
                    'suptitle_y': 0.9,
                    'title_label': '',
                    'xlabel': '',
                    'ylabel': '',
                    'order_axes': (1, 0),
                    'colorbar': True,
                    'colorbar_fraction': 0.5,
                    'colorbar_aspect': 30}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        grid = (1, len(images))
        fig, ax = plt.subplots(*grid, **figure_kwargs)
        ax = to_list(ax)

        # plot image
        for i, img in enumerate(images):
            render_kwargs = filter_kwargs(updated, ['cmap', 'vmin', 'vmax', 'interpolation'], index=i)
            cm = copy(plt.get_cmap(render_kwargs['cmap']))
            cm.set_bad(color=updated.get('bad_color', updated.get('fill_color', 'white')))
            render_kwargs['cmap'] = cm

            img = np.transpose(img.squeeze(), axes=updated['order_axes'])
            ax_img = ax[i].imshow(img, **render_kwargs)
            if filter_kwargs(updated, ['colorbar'], index=i)['colorbar']:
                colorbar_kwargs = filter_kwargs(updated, ['fraction', 'pad'], prefix='colorbar_', index=i)
                cls.add_colorbar(ax_img, **colorbar_kwargs)

            xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'], index=i)
            ax[i].set_xlabel(**xaxis_kwargs)

            yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'], index=i)
            ax[i].set_ylabel(**yaxis_kwargs)

            title_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color', 'y'],
                                         prefix='title_', index=i)
            ax[i].set_title(**title_kwargs)

        suptitle_kwargs = filter_kwargs(updated, ['t', 'y', 'fontsize', 'family', 'color'], prefix='suptitle_')
        fig.suptitle(**suptitle_kwargs)

        return cls.save_and_show(plt, **updated)

    @classmethod
    def histogram(cls, image, **kwargs):
        """ Plot histogram using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            2d-image for plotting.
        kwargs : dict
            label : str
                title of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            bins : int
                the number of bins to use.
            other
        """
        kwargs = cls.convert_kwargs('histogram', kwargs)
        # update defaults
        defaults = {'figsize': (8, 5),
                    'bins': 50,
                    'density': True,
                    'alpha': 0.75,
                    'facecolor': 'b',
                    'fontsize': 15,
                    'label': '',
                    'xlabel': 'values', 'ylabel': ''}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        figure_kwargs = filter_kwargs(updated, ['figsize', 'dpi'])
        histo_kwargs = filter_kwargs(updated, ['bins', 'density', 'alpha', 'facecolor', 'log'])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xlabel_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        ylabel_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])
        xaxis_kwargs = filter_kwargs(updated, ['xlim'])
        yaxis_kwargs = filter_kwargs(updated, ['ylim'])

        # plot the histo
        plt.figure(**figure_kwargs)
        _, _, _ = plt.hist(image.flatten(), **histo_kwargs)
        plt.xlabel(**xlabel_kwargs)
        plt.ylabel(**ylabel_kwargs)
        plt.title(**label_kwargs)
        plt.xlim(xaxis_kwargs.get('xlim'))  # these are positional ones
        plt.ylim(yaxis_kwargs.get('ylim'))

        return cls.save_and_show(plt, **updated)

    @classmethod
    def curve(cls, curve, average=True, window=10, **kwargs):
        """ Plot a curve.

        Parameters
        ----------
        curve : tuple
            a sequence containing curves for plotting along with, possibly, specification of
            plot formats. Must at least contain an array of ys, but may also be comprised of
            triples of (xs, ys, fmt) for an arbitrary number of curves.
        kwargs : dict
            label : str
                title of rendered image.
            xlabel : str
                xaxis-label.
            ylabel : str
                yaxis-label.
            fontsize : int
                fontsize of labels and titles.
            curve_labels : list or tuple
                list/tuple of curve-labels
            other
        """
        kwargs = cls.convert_kwargs('curve', kwargs)
        # defaults
        defaults = {'figsize': (8, 5),
                    'label': 'Curve plot',
                    'linecolor': 'b',
                    'xlabel': 'x',
                    'ylabel': 'y',
                    'fontsize': 15,
                    'grid': True,
                    'legend': True}
        updated = {**defaults, **kwargs}

        # form groups of kwargs
        figure_kwargs = filter_kwargs(updated, ['figsize', 'facecolor', 'dpi'])
        plot_kwargs = filter_kwargs(updated, ['alpha', 'linestyle', 'label'])
        label_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color'])
        xlabel_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        ylabel_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        plot_kwargs['color'] = updated['linecolor']

        plt.figure(**figure_kwargs)

        # Make averaged data
        if average:
            ys = curve[int(len(curve) > 1)]

            averaged, container = [], []
            running_sum = 0

            for value in ys:
                container.append(value)
                running_sum += value
                if len(container) > window:
                    popped = container.pop(0)
                    running_sum -= popped
                averaged.append(running_sum / len(container))

            if len(curve) == 1:
                avg = (averaged, )
            else:
                avg = list(curve)
                avg[1] = averaged

            avg_kwargs = {**plot_kwargs,
                          'label': f'Averaged {plot_kwargs["label"].lower()}',
                          'alpha': 1}
            plt.plot(*avg, **avg_kwargs)
            plot_kwargs['alpha'] = 0.5

        # plot the curve
        plt.plot(*curve, **plot_kwargs)

        if 'curve_labels' in updated:
            plt.legend()

        plt.xlabel(**xlabel_kwargs)
        plt.ylabel(**ylabel_kwargs)
        plt.title(**label_kwargs)
        plt.grid(updated['grid'])

        return cls.save_and_show(plt, **updated)



class PlotlyPlotter:
    """ Plotting backend for plotly.
    """
    @staticmethod
    def convert_kwargs(mode, backend, kwargs):
        """ Update kwargs-dict to match plotly-conventions: update keys of the dict and
        values in some cases.
        """
        # make conversion-dict for kwargs-keys
        keys_converter = {
            'label': 'title', 't': 'title',
            'xlabel': 'xaxis', 'ylabel': 'yaxis',
            'vmin': 'zmin', 'vmax': 'zmax',
        }

        # make new dict updating keys and values
        converted = {}
        for key, value in kwargs.items():
            if key in keys_converter:
                new_key = keys_converter[key]
                if key == 'xlabel':
                    converted[new_key] = {'title_text': value,
                                          'automargin': True,
                                          'titlefont': {'size': kwargs.get('fontsize', 30)}}
                if key == 'ylabel':
                    converted[new_key] = {'title_text': value,
                                          'titlefont': {'size': kwargs.get('fontsize', 30)},
                                          'automargin': True,
                                          'autorange': 'reversed'}
                else:
                    converted[new_key] = value
            else:
                converted[key] = value
        return converted

    @staticmethod
    def save_and_show(fig, show=True, savepath=None, **kwargs):
        """ Save and show plot if needed.
        """
        save_kwargs = kwargs.get('save', dict())

        # save if necessary and render
        if savepath is not None:
            fig.write_image(savepath, **save_kwargs)
        if show:
            fig.show()
        else:
            fig.close()

    @classmethod
    def single(cls, image, **kwargs):
        """ Plot single image/heatmap using plotly.

        Parameters
        ----------
        image : np.ndarray
            2d-array for plotting.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            zmin : float
                the lowest brightness-level to be rendered.
            zmax : float
                the highest brightness-level to be rendered.
            opacity : float
                transparency-level of the rendered image
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('single', kwargs)
        # update defaults to make total dict of kwargs
        defaults = {'reversescale': True,
                    'colorscale': 'viridis',
                    'opacity' : 1.0,
                    'max_size' : 600,
                    'order_axes': (1, 0),
                    'slice': (slice(None, None), slice(None, None))}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['reversescale', 'colorscale', 'opacity', 'showscale'])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = updated['slice']

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Heatmap(z=np.transpose(image, axes=updated['order_axes'])[slc], **render_kwargs)
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        cls.save_and_show(fig, **updated)

    @classmethod
    def overlap(cls, images, **kwargs):
        """ Plot several images on one canvas using plotly: render the first one in greyscale
        and the rest ones in opaque 'rgb' channels, one channel for each image.
        Supports up to four images in total.

        Parameters
        ----------
        images : list/tuple
            sequence of 2d-arrays for plotting. Can store up to four images.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            opacity : float
                opacity of 'rgb' channels.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('overlap', kwargs)
        # update defaults to make total dict of kwargs
        defaults = {'coloraxis_colorbar': {'title': 'amplitude'},
                    'colors': ('red', 'green', 'blue'),
                    'opacity' : 1.0,
                    'title': 'Seismic inline',
                    'max_size' : 600,
                    'order_axes': (1, 0),
                    'slice': (slice(None, None), slice(None, None))}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, ['zmin', 'zmax'])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = updated['slice']

        # calculate canvas sizes
        width, height = images[0].shape[1], images[0].shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # manually combine first image in greyscale and the rest ones colored differently
        combined = channelize_image(255 * np.transpose(images[0], axes=updated['order_axes']),
                                    total_channels=4, greyscale=True)
        for i, img in enumerate(images[1:]):
            color = updated['colors'][i]
            combined += channelize_image(255 * np.transpose(img, axes=updated['order_axes']),
                                         total_channels=4, color=color, opacity=updated['opacity'])
        plot_data = go.Image(z=combined[slc], **render_kwargs) # plot manually combined image

        # plot the figure
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        cls.save_and_show(fig, **updated)

    @classmethod
    def rgb(cls, image, **kwargs):
        """ Plot one image in 'rgb' using plotly.

        Parameters
        ----------
        image : np.ndarray
            3d-array containing channeled rgb-image.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of the rendered image.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('rgb', kwargs)
        # update defaults to make total dict of kwargs
        defaults = {'coloraxis_colorbar': {'title': 'depth'},
                    'max_size' : 600,
                    'order_axes': (1, 0, 2),
                    'slice': (slice(None, None), slice(None, None))}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['xaxis', 'yaxis', 'coloraxis_colorbar', 'title'])
        slc = updated['slice']

        # calculate canvas sizes
        width, height = image.shape[1], image.shape[0]
        coeff = updated['max_size'] / max(width, height)
        width = coeff * width
        height = coeff * height

        # plot the image and set titles
        plot_data = go.Image(z=np.transpose(image, axes=updated['order_axes'])[slc], **render_kwargs)
        fig = go.Figure(data=plot_data)
        fig.update_layout(width=width, height=height, **label_kwargs)

        cls.save_and_show(fig, **updated)

    @classmethod
    def separate(cls, images, **kwargs):
        """ Plot several images on a row of canvases using plotly.
        TODO: add grid support.

        Parameters
        ----------
        images : list/tuple
            sequence of 2d-arrays for plotting.
        kwargs : dict
            max_size : int
                maximum size of a rendered image.
            title : str
                title of rendered image.
            xaxis : dict
                controls the properties of xaxis-labels; uses plotly-format.
            yaxis : dict
                controls the properties of yaxis-labels; uses plotly-format.
            slice : tuple
                sequence of slice-objects for slicing the image to a lesser one.
            order_axes : tuple
                tuple of ints; defines the order of axes for transposition operation
                applied to the image.
            other
        """
        kwargs = cls.convert_kwargs('separate', kwargs)
        # defaults
        defaults = {'max_size' : 600,
                    'order_axes': (1, 0),
                    'slice': (slice(None, None), slice(None, None))}
        grid = (1, len(images))
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        render_kwargs = filter_kwargs(updated, [])
        label_kwargs = filter_kwargs(updated, ['title'])
        xaxis_kwargs = filter_kwargs(updated, ['xaxis'])
        yaxis_kwargs = filter_kwargs(updated, ['yaxis'])
        slc = updated['slice']

        # make sure that the images are greyscale and put them each on separate canvas
        fig = make_subplots(rows=grid[0], cols=grid[1])
        for i in range(grid[1]):
            img = channelize_image(255 * np.transpose(images[i], axes=updated['order_axes']),
                                   total_channels=4, greyscale=True, opacity=1)
            fig.add_trace(go.Image(z=img[slc], **render_kwargs), row=1, col=i + 1)
            fig.update_xaxes(row=1, col=i + 1, **xaxis_kwargs['xaxis'])
            fig.update_yaxes(row=1, col=i + 1, **yaxis_kwargs['yaxis'])
        fig.update_layout(**label_kwargs)

        cls.save_and_show(fig, **updated)

def show_3d(x, y, z, simplices, title, zoom_slice, colors=None, show_axes=True, aspect_ratio=(1, 1, 1),
            axis_labels=None, width=1200, height=1200, margin=(0, 0, 20), savepath=None,
            images=None, resize_factor=2, colorscale='Greys', **kwargs):
    """ Interactive 3D plot for some elements of cube.

    Parameters
    ----------
    x, y, z : numpy.ndarrays
        Triangle vertices.
    simplices : numpy.ndarray
        (N, 3) array where each row represent triangle. Elements of row are indices of points
        that are vertices of triangle.
    title : str
        Title of plot.
    zoom_slice : tuple of slices
        Crop from cube to show.
    colors : list or None
        List of colors for each simplex.
    show_axes : bool
        Whether to show axes and their labels.
    aspect_ratio : tuple of floats.
        Aspect ratio for each axis.
    axis_labels : tuple
        Titel for each axis.
    width, height : number
        Size of the image.
    margin : tuple of ints
        Added margin for each axis, by default, (0, 0, 20).
    savepath : str
        Path to save interactive html to.
    images : list of tuples
        Each tuple is triplet of image, location and axis to load slide from seismic cube.
    resize_factor : float
        Resize factor for seismic slides. Is needed to spedify loading and ploting of seismic slices.
    colorscale : str
        Colormap for seismic slides.
    kwargs : dict
        Other arguments of plot creation.
    """
    #pylint: disable=too-many-arguments
    # Arguments of graph creation
    kwargs = {
        'title': title,
        'colormap': [DEPTHS_CMAP(x) for x in np.linspace(0, 1, 10)],
        'edges_color': 'rgb(70, 40, 50)',
        'show_colorbar': False,
        'width': width,
        'height': height,
        'aspectratio': {'x': aspect_ratio[0], 'y': aspect_ratio[1], 'z': aspect_ratio[2]},
        **kwargs
    }
    if colors is not None:
        fig = ff.create_trisurf(x=x, y=y, z=z, color_func=colors, simplices=simplices, **kwargs)
    else:
        fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, **kwargs)
    if images is not None:
        for image, loc, axis in images:
            shape = image.shape
            image = cv2.resize(image, tuple(np.array(shape) // resize_factor))[::-1]
            grid = np.meshgrid(
                np.linspace(0, shape[0], image.shape[0]),
                np.linspace(0, shape[1], image.shape[1])
            )
            if axis == 0:
                x, y, z = loc * np.ones_like(image), grid[0].T + zoom_slice[1].start, grid[1].T + zoom_slice[2].start
            elif axis == 1:
                y, x, z = loc * np.ones_like(image), grid[0].T + zoom_slice[0].start, grid[1].T + zoom_slice[2].start
            else:
                z, x, y = loc * np.ones_like(image), grid[0].T + zoom_slice[0].start, grid[1].T + zoom_slice[1].start
            fig.add_surface(x=x, y=y, z=z, surfacecolor=np.flipud(image),
                            showscale=False, colorscale='Greys')
    # Update scene with title, labels and axes
    fig.update_layout(
        {
            'scene': {
                'xaxis': {
                    'title': axis_labels[0] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom_slice[0].stop + margin[0], zoom_slice[0].start - margin[0]]
                },
                'yaxis': {
                    'title': axis_labels[1] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom_slice[1].start + margin[1], zoom_slice[1].stop - margin[1]]
                },
                'zaxis': {
                    'title': axis_labels[2] if show_axes else '',
                    'showticklabels': show_axes,
                    'range': [zoom_slice[2].stop + margin[2], zoom_slice[2].start - margin[2]]
                },
                'camera_eye': {
                    "x": 1.25, "y": 1.5, "z": 1.5
                },
            }
        }
    )
    fig.show()

    if savepath:
        fig.write_html(savepath)
