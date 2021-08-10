""" A mixin with field visualizations. """


class VisualizationMixin:
    """ !!. """

    def __repr__(self):
        return f"""<Field `{self.displayed_name}` at {hex(id(self))}>"""

    def __str__(self):
        processed_prefix = 'un' if self.geometry.has_stats is False else ''
        labels_prefix = ':' if self.labels else ''
        msg = f'Field `{self.displayed_name}` with {processed_prefix}processed geometry{labels_prefix}\n'
        for label in self.labels:
            msg += f'    {label.name}\n'
        return msg
