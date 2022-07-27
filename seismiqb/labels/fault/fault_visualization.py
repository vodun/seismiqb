import numpy as np

from ..mixins import VisualizationMixin
from ...plotters import show_3d
from ...utils import make_slices

class FaultVisualizationMixin(VisualizationMixin):
    def __repr__(self):
        return f"""<Fault `{self.name}` for `{self.field.displayed_name}` at {hex(id(self))}>"""

    def show_slide(self, loc, **kwargs):
        cmap = kwargs.get('cmap', ['Greys_r', 'red'])
        width = kwargs.get('width', 5)

        kwargs = {**kwargs, 'cmap': cmap, 'width': width}
        super().show_slide(loc, **kwargs)

    def show(self, axis=0, centering=True, zoom=None, **kwargs):
        if centering and zoom is None:
            zoom = [slice(max(0, self.bbox[i][0]-20), min(self.bbox[i][1]+20, self.field.shape[i])) for i in range(3) if i != axis]

        return self.show_slide(loc=int(np.mean(self.bbox[axis])), zoom=zoom, axis=axis, **kwargs)

    def show_3d(self, sticks_step=None, stick_nodes_step=None, z_ratio=1., colors='green',
                zoom=None, margin=20, **kwargs):
        """ Interactive 3D plot. Roughly, does the following:
            - select `n` points to represent the horizon surface
            - triangulate those points
            - remove some of the triangles on conditions
            - use Plotly to draw the tri-surface

        Parameters
        ----------
        sticks_step : int
            Number of slides between sticks.
        stick_nodes_step : int
            Distance between stick nodes
        z_ratio : int
            Aspect ratio between height axis and spatial ones.
        zoom : tuple of slices or None.
            Crop from cube to show. If None, the whole cube volume will be shown.
        show_axes : bool
            Whether to show axes and their labels, by default True
        width, height : int
            Size of the image, by default 1200, 1200
        margin : int
            Added margin from below and above along height axis, by default 20
        savepath : str
            Path to save interactive html to.
        kwargs : dict
            Other arguments of plot creation.
        """
        title = f'Fault `{self.name}` on `{self.field.displayed_name}`'
        aspect_ratio = (self.i_length / self.x_length, 1, z_ratio)
        axis_labels = (self.field.index_headers[0], self.field.index_headers[1], 'DEPTH')

        zoom = make_slices(zoom, self.field.shape)

        margin = [margin] * 3 if isinstance(margin, int) else margin
        x, y, z, simplices = self.make_triangulation(zoom, sticks_step, stick_nodes_step)
        if isinstance(colors, str):
            colors = [colors for _ in simplices]

        show_3d(x, y, z, simplices, title=title, zoom=zoom, aspect_ratio=aspect_ratio,
                axis_labels=axis_labels, margin=margin, colors=colors, **kwargs)

    # cache?
    def make_triangulation(self, slices=None, sticks_step=None, stick_nodes_step=None, *args, **kwargs):
        """ Return triangulation of the fault. It will created if needed. """
        if sticks_step is not None or stick_nodes_step is not None:
            fake_fault = Fault({'points': self.points}, field=self.field)
            fake_fault.points_to_sticks(slices, sticks_step or 10, stick_nodes_step or 10)
            return fake_fault.make_triangulation(slices, *args, **kwargs)

        if len(self.simplices) > 0:
            return self.nodes[:, 0], self.nodes[:, 1], self.nodes[:, 2], self.simplices
        fake_fault = get_fake_one_stick_fault(self)
        return fake_fault.make_triangulation(slices, sticks_step, stick_nodes_step, *args, **kwargs)
