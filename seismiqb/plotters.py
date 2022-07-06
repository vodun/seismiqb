""" Plot functions. """
# pylint: disable=too-many-statements
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.graph_objects as go

from batchflow import plot as batchflow_plot
from matplotlib.cm import get_cmap, register_cmap
from matplotlib.colors import ColorConverter, ListedColormap, LinearSegmentedColormap


# Predefined colormaps

SEISMIC2_CDICT = {
    'red': [[0.0, None, 0.0], [0.25, 0.5, 0.5], [0.5, 1., 1], [0.75, 0.75, 0.75], [1.0, 1, None]],
    'green': [[0.0, None, 0.0], [0.25, 0.5, 0.5], [0.5, 1., 1], [0.75, 0.25, 0.25], [1.0, 0., None]],
    'blue': [[0.0, None, 1.0], [0.25, 0.5, 0.5], [0.5, 1., 1], [0.75, 0., 0.0], [1.0, 0.0, None]],
}
SEISMIC2_CMAP = LinearSegmentedColormap('Seismic2', SEISMIC2_CDICT)
register_cmap(name='Seismic2', cmap=SEISMIC2_CMAP)

METRIC_CDICT = {
    'red': [[0.0, None, 1.0], [0.33, 1.0, 1.0], [0.66, 1.0, 1.0], [1.0, 0.0, None]],
    'green': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 1.0, 1.0], [1.0, 0.5, None]],
    'blue': [[0.0, None, 0.0], [0.33, 0.0, 0.0], [0.66, 0.0, 0.0], [1.0, 0.0, None]]
}
METRIC_CMAP = LinearSegmentedColormap('Metric', METRIC_CDICT)
METRIC_CMAP.set_bad(color='black')
register_cmap(name='Metric', cmap=METRIC_CMAP)

BASIC_CMAP = ListedColormap(get_cmap('ocean')(np.linspace(0.0, 0.9, 100)))
register_cmap(name='Basic', cmap=BASIC_CMAP)

DEPTHS_CMAP = ListedColormap(get_cmap('viridis_r')(np.linspace(0.0, 0.5, 100)))
register_cmap(name='Depths', cmap=DEPTHS_CMAP)

SAMPLER_CMAP = ListedColormap([ColorConverter().to_rgb('blue'),
                                ColorConverter().to_rgb('red'),
                                ColorConverter().to_rgb('purple')])
register_cmap(name='Sampler', cmap=SAMPLER_CMAP)


class plot(batchflow_plot):
    """ Wrapper over original `plot` with custom defaults. """
    def __init__(self, *args, **kwargs):
        kwargs = {'transpose': (1, 0, 2), **kwargs}
        super().__init__(*args, **kwargs)


def show_3d(x, y, z, simplices, title, zoom_slice, colors=None, show_axes=True, aspect_ratio=(1, 1, 1),
            axis_labels=None, width=1200, height=1200, margin=(0, 0, 20), savepath=None,
            images=None, bounds=False, resize_factor=2, colorscale='Greys', show=True, camera=None, **kwargs):
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
    bounds : bool or int
        Whether to draw bounds on slides. If int, width of the border.
    resize_factor : float
        Resize factor for seismic slides. Is needed to spedify loading and ploting of seismic slices.
    colorscale : str
        Colormap for seismic slides.
    show : bool
        Whether to show figure.
    camera : dict
        Parameters for initial camera view.
    kwargs : dict
        Other arguments of plot creation.
    """
    #pylint: disable=too-many-arguments
    # Arguments of graph creation
    kwargs = {
        'title': title,
        'colormap': [plt.get_cmap('Depths')(x) for x in np.linspace(0, 1, 10)],
        'edges_color': 'rgb(70, 40, 50)',
        'show_colorbar': False,
        'width': width,
        'height': height,
        'aspectratio': {'x': aspect_ratio[0], 'y': aspect_ratio[1], 'z': aspect_ratio[2]},
        **kwargs
    }
    if isinstance(colorscale, str) and colorscale in plt.colormaps():
        cmap = get_cmap(colorscale)
        levels = np.arange(0, 256, 1) / 255
        colorscale = [
            (level, f'rgb({r * 255}, {g * 255}, {b * 255})')
            for (r, g, b, _), level in zip(cmap(levels), levels)
        ]

    if simplices is not None:
        if colors is not None:
            fig = ff.create_trisurf(x=x, y=y, z=z, color_func=colors, simplices=simplices, **kwargs)
        else:
            fig = ff.create_trisurf(x=x, y=y, z=z, simplices=simplices, **kwargs)
    else:
        fig = go.Figure()
    if images is not None:
        for image, loc, axis in images:
            shape = image.shape
            image = cv2.resize(image, tuple(np.array(shape) // resize_factor))[::-1]
            if bounds:
                bounds = int(bounds)
                fill = image.max()
                image[:bounds, :] = fill
                image[-bounds:, :] = fill
                image[:, :bounds] = fill
                image[:, -bounds:] = fill

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
                            showscale=False, colorscale=colorscale)
    # Update scene with title, labels and axes
    fig.update_layout(
        {
            'width': kwargs['width'],
            'height': kwargs['height'],
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
                'camera': camera or {'eye': {"x": 1.25, "y": 1.5, "z": 1.5}},
            }
        }
    )
    if show:
        fig.show()

    if isinstance(savepath, str):
        ext = os.path.splitext(savepath)[1][1:]
        if ext == 'html':
            fig.write_html(savepath)
        elif ext in ['png', 'jpg', 'jpeg', 'pdf']:
            fig.write_image(savepath, format=ext)
