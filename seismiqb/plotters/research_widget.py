import ipywidgets as widgets
from IPython.display import display, Image
import os
import re

class WidgetForResearch:
    def __init__(self, cubes, research_name, df_with_research_results, feature, repetition, width=700, height=700):
        self.idx = 0
        self.research_name = research_name
        self.df_with_idx = df_with_research_results.set_index('id')
        self.cubes = cubes

        # Gather images from the directory, parse research dataframe and create dict,
        # where key is a feature (e.g. 'blovasz/0/001_YETYPUR/((0,_1000),_(500,_501),_(100,_1100))_1') and value is its path
        self.df_ids = self.get_ids_from_df(df_with_research_results, repetition)
        self.image_paths = self.gather_images(self.cubes, self.df_ids)
        self.features = self.find_feature_in_df(feature)
        self.dict_of_images_and_paths = dict(zip(self.features, self.image_paths))

        # Make widgets
        self.image_widget = widgets.Image(format='png',
                                          width=width, height=height,
                                          value=open(self.image_paths[self.idx], 'rb').read())
        self.next_button = widgets.Button(description='Next')
        self.previous_button = widgets.Button(description='Prev')
        self.dropdown = widgets.Dropdown(options=self.features, value=self.features[self.idx])
        self.title = widgets.Label(value=self.features[self.idx])

        hbox = widgets.HBox([self.previous_button, self.next_button])
        vbox = widgets.VBox([hbox, self.image_widget, self.title])
        result_box = widgets.HBox([vbox, self.dropdown])

        # Setup widgets
        self.next_button.on_click(self.on_button_click)
        self.dropdown.observe(self.dropdown_callback, names='value')
        self.previous_button.on_click(self.prev_button_click)

        display(result_box)

    def find_id_in_string(self, image_path):

        pattern = r"\b[\dA-Fa-f]+_[\dA-Fa-f]+_[\dA-Fa-f]+\b"
        output = re.findall(pattern, image_path)[0]

        return output

    def gather_images(self, cubes, df_ids):

        image_paths = []
        for cube_path in cubes:
            for root, dirs, files in os.walk(f'{self.research_name}/experiments'):
                if root.endswith('inference') and self.find_id_in_string(root) in df_ids:
                    img = os.path.join(root, cube_path)
                    image_paths.append(img)

        return image_paths

    def find_feature_in_df(self, feature):

        features = []
        for cube in self.cubes:
            for id_ in self.df_ids:
                feature_ = '/'.join([f"{self.df_with_idx.loc[id_, feature]}", f"{self.df_with_idx.loc[id_, 'repetition']}", cube])
                features.append(feature_)

        return features

    def get_ids_from_df(self, df, rep):

        return list(df[df['repetition'] == rep]['id'])

    def on_button_click(self, b):

        self.idx += 1
        if self.idx == len(self.image_paths):
            self.idx = 0
        self.image_widget.value = open(self.image_paths[self.idx], 'rb').read()

        new_title = self.features[self.idx]
        self.title.value = new_title

    def prev_button_click(self, b):

        self.idx -= 1
        self.image_widget.value = open(self.image_paths[self.idx], 'rb').read()

        new_title = self.features[self.idx]
        self.title.value = new_title

    def dropdown_callback(self, change):

        file_path = self.dict_of_images_and_paths[change.new]
        self.idx = self.image_paths.index(file_path)
        self.image_widget.value = open(file_path, 'rb').read()

        new_title = self.features[self.idx]
        self.title.value = new_title