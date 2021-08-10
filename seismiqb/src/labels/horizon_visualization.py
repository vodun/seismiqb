""" Mixin for horizon visualization. """
from textwrap import dedent

import numpy as np
from matplotlib import pyplot as plt

from scipy.spatial import Delaunay

from ..plotters import plot_image, show_3d
from ..utils import filter_simplices, make_savepath



class VisualizationMixin:
    """ Methods for textual and visual representation of a horizon. """
    def __repr__(self):
        return f"""<horizon {self.name} for {self.geometry.displayed_name} at {hex(id(self))}>"""

    def __str__(self):
        msg = f"""
        Horizon {self.name} for {self.geometry.displayed_name} loaded from {self.format}
        Ilines range:      {self.i_min} to {self.i_max}
        Xlines range:      {self.x_min} to {self.x_max}
        Depth range:       {self.h_min} to {self.h_max}
        Depth mean:        {self.h_mean:.6}
        Depth std:         {self.h_std:.6}

        Length:            {len(self)}
        Perimeter:         {self.perimeter}
        Coverage:          {self.coverage:3.5}
        Solidity:          {self.solidity:3.5}
        Num of holes:      {self.number_of_holes}
        """

        if self.is_carcass:
            msg += f"""
        Unique ilines:     {self.carcass_ilines}
        Unique xlines:     {self.carcass_xlines}
        """
        return dedent(msg)


    # 2D
    def show(self, attributes='depths', mode='imshow', return_figure=False, enlarge=True, width=9, **kwargs):
        """ Display facies attributes with predefined defaults.

        Loads requested data, constructs default parameters wrt to that data and delegates plot to `plot_image`.

        Parameters
        ----------
        attributes : str or dict or np.ndarray, or a list of objects of those types
            Defines attributes to display.
            If str, a name of attribute to load. Address `Facies.METHOD_TO_ATTRIBUTE` values for details.
            If dict, arguments for `Facies.load_attribute` and optional callable param under 'postprocess` key.
            If np.ndarray, must be 2d and match facies full matrix shape.
            If list, defines several data object to display. For details about nestedness address `plot_image` docs.
        mode : 'imshow' or 'hist'
            Mode to display images in.
        return_figure : bool
            Whether return resulted figure or not.
        kwargs : for `plot_image`

        Examples
        --------
        Display depth attribute:
        >>> facies.show()
        Display several attributes one over another:
        >>> facies.show(['amplitudes', 'channels/masks'])
        Display several attributes separately:
        >>> facies.show(['amplitudes', 'instant_amplitudes'], separate=True)
        Display several attributes in mixed manner:
        >>> facies.show(['amplitudes', ['amplitudes', 'channels/masks']])
        Display attribute with additional postprocessing:
        >>> facies.show({'src': 'amplitudes', 'fill_value': 0, 'normalize': 'min-max'})

        Notes
        -----
        Asterisks in title-like and 'savepath' parameters are replaced by label displayed name.
        """
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
            load_defaults = {'dtype': np.float32, 'fill_value': np.nan}
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
            'suptitle_label': f"`{self.short_name}` of cube `{self.geometry.displayed_name}`",
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
            attr = self.ALIAS_TO_ATTRIBUTE.get(attr, attr)

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
            x, y = self.matrix.shape
            defaults = {
                **defaults,
                'figsize': (x / min(x, y) * n_subplots * 7, y / min(x, y) * 7),
                'xlim': self.bbox[0],
                'ylim': self.bbox[1][::-1],
                'cmap': apply_by_scenario(make_cmap, names),
                'alpha': apply_by_scenario(make_alpha, names),
                'xlabel': self.geometry.index_headers[0],
                'ylabel': self.geometry.index_headers[1],
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
            params['savepath'] = make_savepath(params['savepath'], self.short_name, '.png')

        # Plot image with given params and return resulting figure
        figure = plot_image(data=data, mode=mode, **params)
        plt.show()

        return figure if return_figure else None

    def show_slide(self, loc, width=None, axis='i', zoom_slice=None, **kwargs):
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
        # Make `locations` for slide loading
        axis = self.geometry.parse_axis(axis)

        # Load seismic and mask
        seismic_slide = self.geometry.load_slide(loc=loc, axis=axis)
        mask = self.load_slide(loc=loc, axis=axis, width=width)
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

        title = f'Horizon `{self.name}` on cube `{self.geometry.displayed_name}`\n {header} {loc} out of {total}'

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

    # 3D
    def show_3d(self, n_points=100, threshold=100., z_ratio=1., zoom_slice=None, show_axes=True,
                width=1200, height=1200, margin=(0, 0, 100), savepath=None, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        n_points : int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        threshold : int
            Threshold to remove triangles with bigger height differences in vertices.
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom_slice : tuple of slices
            Crop from cube to show.
        show_axes : bool
            Whether to show axes and their labels.
        width, height : int
            Size of the image.
        margin : int
            Added margin from below and above along height axis.
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Horizon `{self.short_name}` on `{self.geometry.displayed_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.geometry.index_headers[0], self.geometry.index_headers[1], 'DEPTH')
        if zoom_slice is None:
            zoom_slice = [slice(0, i) for i in self.geometry.cube_shape]
        zoom_slice[-1] = slice(self.h_min, self.h_max)

        x, y, z, simplices = self.make_triangulation(n_points, threshold, zoom_slice)

        show_3d(x, y, z, simplices, title, zoom_slice, None, show_axes, aspect_ratio,
                axis_labels, width, height, margin, savepath, **kwargs)

    def make_triangulation(self, n_points, threshold, slices, **kwargs):
        """ Create triangultaion of horizon.

        Parameters
        ----------
        n_points: int
            Number of points for horizon surface creation.
            The more, the better the image is and the slower it is displayed.
        slices : tuple
            Region to process.

        Returns
        -------
        x, y, z, simplices
            `x`, `y` and `z` are np.ndarrays of triangle vertices, `simplices` is (N, 3) array where each row
            represent triangle. Elements of row are indices of points that are vertices of triangle.
        """
        _ = kwargs
        weights_matrix = self.full_matrix.astype(np.float32)

        grad_i = np.diff(weights_matrix, axis=0, prepend=0)
        grad_x = np.diff(weights_matrix, axis=1, prepend=0)
        weights_matrix = (grad_i + grad_x) / 2
        weights_matrix[np.abs(weights_matrix) > 100] = np.nan

        idx = np.stack(np.nonzero(self.full_matrix > 0), axis=0)
        mask_1 = (idx <= np.array([slices[0].stop, slices[1].stop]).reshape(2, 1)).all(axis=0)
        mask_2 = (idx >= np.array([slices[0].start, slices[1].start]).reshape(2, 1)).all(axis=0)
        mask = np.logical_and(mask_1, mask_2)
        idx = idx[:, mask]

        probs = np.abs(weights_matrix[idx[0], idx[1]].flatten())
        probs[np.isnan(probs)] = np.nanmax(probs)
        indices = np.random.choice(len(probs), size=n_points, p=probs / probs.sum())

        # Convert to meshgrid
        ilines = self.points[mask, 0][indices]
        xlines = self.points[mask, 1][indices]
        ilines, xlines = np.meshgrid(ilines, xlines)
        ilines = ilines.flatten()
        xlines = xlines.flatten()

        # Remove from grid points with no horizon in it
        heights = self.full_matrix[ilines, xlines]
        mask = (heights != self.FILL_VALUE)
        x = ilines[mask]
        y = xlines[mask]
        z = heights[mask]

        # Triangulate points and remove some of the triangles
        tri = Delaunay(np.vstack([x, y]).T)
        simplices = filter_simplices(simplices=tri.simplices, points=tri.points,
                                     matrix=self.full_matrix, threshold=threshold)
        return x, y, z, simplices
