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



METRIC_CDICT = {
    'red': [[0.0, None, 1.0], [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 0.0, None]],
    'green': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 1.0, 1.0], [1.0, 0.5, None]],
    'blue': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, None]]
}
METRIC_CMAP = LinearSegmentedColormap('Metric', METRIC_CDICT)
METRIC_CMAP.set_bad(color='black')
register_cmap(name='Metric', cmap=METRIC_CMAP)

DEPTHS_CMAP = ListedColormap(get_cmap('viridis_r')(np.linspace(0.0, 0.5, 100)))
register_cmap(name='Depths', cmap=DEPTHS_CMAP)



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
    """ Make a subdictionary of arguments with required keys.

    Parameters
    ----------
    kwargs : dict
        Arguments to filter.
    keys : sequence
        Keys to retrieve.
    index : int, optional
        Number of argument value component to get.
        If none provided, get whole argument value.
        If value is non-indexable, get it without indexing.
        Defaults to `slice(None)`, i.e. values are not indexed.
    prefix : str, optional
        If a key with prefix is in kwargs, get its value.
        If not, try to get value by the key itself.
        Defaults to `''`, i.e. no prefix used.
    """
    result = {}
    for key in keys:
        value = kwargs.get(prefix + key, kwargs.get(key))
        if value is None:
            continue
        if hasattr(value, '__getitem__') and not isinstance(value, str):
            result[key] = value[index]
        else:
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
    """ Plotting backend for matplotlib.

    Consists of supplementary and rendering methods. The latter make heavy use of the following logic:
    1. Recieve a dict of kwargs for all plotting steps.
    2. Using `filter_kwargs` split this dict into subdicts for every plotting function:
      a. First try to look for keys with specific prefix, if provided.
      b. If no key with such prefix found, look for key without a prefix.

    This trick allows one to pass arguments of the same name for different plotting steps.
    E.g. `plt.set_title` and `plt.set_xlabel` both require `fontsize` argument.
    Providing `{'fontsize': 30}` in kwargs will affect both title and x-axis labels.
    To change parameter for title only, one can provide {'title_fontsize': 30}` instead.

    To see all acceptable rendering parameters address class defaults.
    """

    # Keys to expect for different plotting functions

    # `plt.subplots`
    FIGURE_KEYS = ['figsize', 'facecolor', 'dpi']
    # `plt.plot`
    PLOT_KEYS = ['color', 'linestyle', 'marker']
    # `plt.imshow`
    IMSHOW_KEYS = ['cmap', 'vmin', 'vmax', 'alpha', 'interpolation']
    # auxiliary
    TEXT_KEYS = ['fontsize', 'family', 'color']
    # `plt.set_title`
    TITLE_KEYS = ['label', 'y'] + TEXT_KEYS
    # `plt.set_xlabel`
    XLABEL_KEYS = ['xlabel'] + TEXT_KEYS
    # `plt.set_ylabel`
    YLABEL_KEYS = ['ylabel'] + TEXT_KEYS
    # `cls.add_colorbar`
    COLORBAR_KEYS = ['fraction', 'aspect']
    # `plt.tick_params`
    TICK_KEYS = ['labeltop', 'labelright', 'labelcolor', 'direction']

    # Supplementary methods

    @staticmethod
    def color_to_cmap(color):
        """ Create a colormap of single color. """
        if isinstance(color, str):
            color = ColorConverter().to_rgb(color)
        return ListedColormap(color)

    @staticmethod
    def add_colorbar(image, aspect=30, fraction=0.5, color='black', **kwargs):
        """ Append colorbar to the image on the right. """
        divider = axes_grid1.make_axes_locatable(image.axes)
        width = axes_grid1.axes_size.AxesY(image.axes, aspect=1./aspect)
        pad = axes_grid1.axes_size.Fraction(fraction, width)
        cax = divider.append_axes("right", size=width, pad=pad)
        colorbar = image.axes.figure.colorbar(image, cax=cax, **kwargs)
        colorbar.ax.yaxis.set_tick_params(color=color)

    @staticmethod
    def add_legend(axis, color, label, size=20):
        """ Add a patch to legend. """
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
        keys_converter = {
            'zmin': 'vmin', 'zmax': 'vmax',
            'xaxis': 'xlabel', 'yaxis': 'ylabel'
        }

        if mode in ['single', 'rgb', 'overlap', 'histogram', 'curve', 'histogram']:
            keys_converter.update({'title': 'label', 't': 'label'})
        elif mode == 'separate':
            keys_converter.update({'title': 't', 'label': 't'})
        elif mode == 'grid':
            keys_converter.update({'title': 't'})

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

    @classmethod
    def make_figure(cls, updated):
        """ Create figure and axes if needed, else use provided. """
        if 'ax' in updated:
            ax = updated['ax']
            fig = ax.figure
        else:
            figure_kwargs = filter_kwargs(updated, cls.FIGURE_KEYS)
            fig, ax = plt.subplots(**figure_kwargs)
        return fig, ax

    @classmethod
    def annotate_image(cls, ax, updated, actions):
        """ Make necessary annotations. """
        if 'set_title' in actions:
            title_kwargs = filter_kwargs(updated, cls.TITLE_KEYS, prefix='title_')
            ax.set_title(**title_kwargs)
        if 'set_xlabel' in actions:
            xlabel_kwargs = filter_kwargs(updated, cls.XLABEL_KEYS, prefix='xlabel_')
            ax.set_xlabel(**xlabel_kwargs)
        if 'set_ylabel' in actions:
            ylabel_kwargs = filter_kwargs(updated, cls.YLABEL_KEYS, prefix='ylabel_')
            ax.set_ylabel(**ylabel_kwargs)
        if 'set_xticks' in actions:
            if 'xticks' in updated:
                ax.set_xticks(updated['xticks'])
        if 'set_yticks' in actions:
            if 'yticks' in updated:
                ax.set_yticks(updated['yticks'])
        if 'add_colorbar' in actions:
            colorbar_kwargs = filter_kwargs(updated, cls.COLORBAR_KEYS, prefix='colorbar_')
            cls.add_colorbar(updated['ax_img'], **colorbar_kwargs)
        if 'set_facecolor' in actions:
            if updated['facecolor']:
                ax.set_facecolor(updated['facecolor'])
        if 'tick_params' in actions:
            tick_params = filter_kwargs(updated, cls.TICK_KEYS)
            ax.tick_params(**tick_params)
        if 'disable_axes' in actions:
            if updated.get('disable_axes'):
                ax.set_axis_off()
        if 'set_xlim' in actions:
            ax.set_xlim(updated['xlim'])
        if 'set_ylim' in actions:
            ax.set_ylim(updated['ylim'])

    @staticmethod
    def save_and_show(fig, show=True, savepath=None, return_figure=False, pyqt=False, **kwargs):
        """ Save and show plot if needed. """
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

    # Rendering methods

    @classmethod
    def single(cls, image, **kwargs):
        """ Plot single image/heatmap using matplotlib.

        Parameters
        ----------
        image : np.ndarray
            2d-array for plotting.
        kwargs : dict
            order_axes : tuple of ints
                Order of image axes.
            disable_axes : bool
                Whether call `set_axis_off` or not.
            xticks : sequence
                For `plt.set_xticks`
            yticks : sequence
                For `plt.set_yticks`
            arguments for following methods:
                `plt.subplots` — with 'figure_' prefix
                `plt.imshow` — with 'imshow_' prefix
                `plt.set_title` — with 'title_' prefix
                `plt.set_xlabel`— with 'xlabel_' prefix
                `plt.set_ylabel` — with 'ylabel_' prefix
                `cls.add_colorbar` — with 'colorbar' prefix
                `plt.tick_params` — with 'tick_' prefix
                See class docs for details on prefixes usage.
                See class and method defaults for arguments names.
        """
        defaults = {# figure
                    'figsize': (12, 7),
                    # imshow
                    'cmap': DEPTHS_CMAP,
                    'facecolor': 'white',
                    # axis labels
                    'xlabel': '', 'ylabel': '',
                    # colorbar
                    'colorbar': True,
                    'colorbar_fraction': 3.0,
                    'colorbar_aspect': 30,
                    # ticks
                    'labeltop': True,
                    'labelright': True,
                    'direction': 'inout',
                    # common
                    'fontsize': 20,
                    'title': '',
                    'label': '',
                    # other
                    'order_axes': (1, 0)}

        kwargs = cls.convert_kwargs('single', kwargs)
        updated = {**defaults, **kwargs}

        # prepare figure and axis
        fig, ax = cls.make_figure(updated)

        # process imshow arguments
        imshow_kwargs = filter_kwargs(updated, cls.IMSHOW_KEYS)

        cm = copy(plt.get_cmap(imshow_kwargs['cmap']))
        cm.set_bad(color=updated.get('bad_color', updated.get('fill_color', 'white')))
        imshow_kwargs['cmap'] = cm

        img = np.transpose(image.squeeze(), axes=updated['order_axes'])
        xticks = updated.get('xticks', [0, img.shape[1]])
        yticks = updated.get('yticks', [img.shape[0], 0])
        imshow_kwargs['extent'] = [xticks[0], xticks[-1], yticks[0], yticks[-1]]

        updated['ax_img']  = ax.imshow(img, **imshow_kwargs)

        # manage title, axis labels, colorbar, ticks
        cls.annotate_image(ax, updated, ['set_title', 'set_xlabel', 'set_ylabel', 'set_xticks', 'set_yticks',
                                         'add_colorbar', 'set_facecolor', 'tick_params', 'disable_axes'])

        return cls.save_and_show(fig, **updated)


    @classmethod
    def wiggle(cls, image, **kwargs):
        """ Make wiggle plot of signals array. Optionally overlap it with a curve.

        Parameters
        ----------
        image : np.ndarray or list of np.ndarray
            If array, must be 2d.
            If list, must contain image and curve arrays.
            Curve, in turn must be either 1d array of heights or 2d array mask.
                If 1d heights, its shape must match correposnding image dimension.
                If 2d mask, its shape must match image shape.
                In both cases it is expected, that there must be `np.nan` where curve is not defined.
        kwargs : dict
            step : int, optional
                Step to take signals from the array with.
            reverse : bool, optional
                Whether reverse the plot wrt y-axis or not.
            width_multiplier : float, optional
                Scale factor for signals amplitudes.
            arguments for following methods:
                `plt.subplots` — with 'figure_' prefix
                `plt.plot` — with 'wiggle_' and 'curve_' prefixes
                `plt.set_title` — with 'title_' prefix
                `plt.set_xlabel`— with 'xlabel_' prefix
                `plt.set_ylabel` — with 'ylabel_' prefix
                `plt.set_xlim`— with 'xlim_' prefix
                `plt.set_ylim` — with 'ylim_' prefix
                See class docs for details on prefixes usage.
                See class and method defaults for arguments names.
        """
        defaults = {# general
                    'step': 15,
                    'reverse': True,
                    'width_multiplier': 1,
                    # figure
                    'figsize': (12, 7),
                    # wiggle
                    'wiggle_color': 'k',
                    'wiggle_linestyle': '-',
                    # curve
                    'curve_color': 'r',
                    'curve_marker': 'o',
                    'curve_linestyle': '',
                    # axis labels
                    'xlabel': '', 'ylabel': '',
                    # common
                    'fontsize': 20, 'label': '', 'title': ''}

        kwargs = cls.convert_kwargs('wiggle', kwargs)
        updated = {**defaults, **kwargs}

        # prepare figure and axis
        fig, ax = cls.make_figure(updated)

        # unpack image and curve data, if latter is provided
        curve = None
        if isinstance(image, (list, tuple)):
            image, *curve = image
            curve = curve[0] if curve else None

        # Creating wiggle-curves and adding height-points if needed
        offsets = np.arange(0, image.shape[0], updated['step'])

        y_order = -1 if updated['reverse'] else 1
        y_range = np.arange(0, image.shape[1])[::y_order]
        x_range = [] # accumulate traces to draw curve above them later
        for ix, k in enumerate(offsets):
            x = k + updated['width_multiplier'] * image[k] / np.std(image)
            wiggle_kwargs = filter_kwargs(updated, cls.PLOT_KEYS, prefix='wiggle_', index=ix)
            ax.plot(x, y_range, **wiggle_kwargs)
            ax.fill_betweenx(y_range, k, x, where=(x > k), color=wiggle_kwargs['color'])
            x_range.append(x)
        x_range = np.r_[x_range][:, ::y_order]

        if 'xlim' not in updated:
            updated['xlim'] = (x_range[0].min(), x_range[-1].max())
        if 'ylim' not in updated:
            updated['ylim'] = (y_range.min(), y_range.max())

        if curve is not None:
            curve = curve[offsets]
            if curve.ndim == 1:
                curve_x = (~np.isnan(curve)).nonzero()
                curve_y = curve[curve_x]
            # transform height-mask to heights if needed
            elif curve.ndim == 2:
                curve = curve[:, ::y_order]
                curve = (~np.isnan(curve)).nonzero()
                width = updated['horizon_width']
                curve_x = curve[0][width // 2::width]
                curve_y = curve[1][width // 2::width]
            curve_kwargs = filter_kwargs(updated, cls.PLOT_KEYS, prefix='curve_')
            ax.plot(x_range[curve_x, curve_y], curve_y, **curve_kwargs)

        # manage title, axis labels, colorbar, ticks
        cls.annotate_image(ax, updated, ['set_title', 'set_xlabel', 'set_ylabel', 'set_xlim', 'set_ylim'])

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
                    'size': 20,
                    'order_axes': (1, 0),
                    # title
                    'title_y' : 1.1,
                    # colorbar
                    'colorbar': True,
                    'colorbar_fraction': 0.5,
                    'colorbar_aspect': 30,
                    # legend
                    'legend': True,
                    'legend_size': 10}
        updated = {**defaults, **kwargs}

        # form different groups of kwargs
        figure_kwargs = filter_kwargs(updated, cls.FIGURE_KEYS)
        imshow_kwargs = filter_kwargs(updated, cls.IMSHOW_KEYS)
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

        img = ax.imshow(img, extent=extent, **imshow_kwargs)
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
            imshow_kwargs = filter_kwargs(updated, ['color', 'alpha'], index=i)
            layer_color = imshow_kwargs.pop('color')
            imshow_kwargs['cmap'] = cls.color_to_cmap(layer_color)
            ax.imshow(image, extent=extent, **imshow_kwargs)
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
                    'size': 20,
                    # title
                    'title_y': 1.1,
                    # legend
                    'legend': True,
                    'legend_size': 10}
        updated = {**defaults, **kwargs}


        # form different groups of kwargs
        title_kwargs = filter_kwargs(updated, ['label', 'fontsize', 'family', 'color', 'y'], prefix='title_')
        xaxis_kwargs = filter_kwargs(updated, ['xlabel', 'fontsize', 'family', 'color'])
        yaxis_kwargs = filter_kwargs(updated, ['ylabel', 'fontsize', 'family', 'color'])

        # Create figure and axes
        fig, ax = cls.make_figure(updated)

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
        figure_kwargs = filter_kwargs(updated, cls.FIGURE_KEYS)
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
        figure_kwargs = filter_kwargs(updated, cls.FIGURE_KEYS)
        grid = (1, len(images))
        fig, ax = plt.subplots(*grid, **figure_kwargs)
        ax = to_list(ax)

        # plot image
        for i, img in enumerate(images):
            imshow_kwargs = filter_kwargs(updated, cls.IMSHOW_KEYS, index=i)
            cm = copy(plt.get_cmap(imshow_kwargs['cmap']))
            cm.set_bad(color=updated.get('bad_color', updated.get('fill_color', 'white')))
            imshow_kwargs['cmap'] = cm

            img = np.transpose(img.squeeze(), axes=updated['order_axes'])
            ax_img = ax[i].imshow(img, **imshow_kwargs)
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
        figure_kwargs = filter_kwargs(updated, cls.FIGURE_KEYS)
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
        figure_kwargs = filter_kwargs(updated, cls.FIGURE_KEYS)
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
