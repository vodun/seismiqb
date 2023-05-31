from IPython.display import display, Image

import ipywidgets as widgets
import os
import re

class GatherImagesFromResearch:
    """ Auxiliary class parsing research dataframe and gathering images corresponding to the research experiments """
    def __init__(self, cubes, research_name, df_with_research_results, feature, repetition):
        self.research_name = research_name
        self.df_with_idx = df_with_research_results.set_index('id')
        self.cubes = cubes

        self.df_ids = self.get_ids_from_df(df_with_research_results, repetition)
        self.image_paths = self.gather_images(self.cubes, self.df_ids)
        self.meta_info = self.find_feature_in_df(feature)
        self.images_with_paths = dict(zip(self.meta_info, self.image_paths))

    def find_id_in_string(self, image_path):
        """ Find id pattern in a file name of the research experiment """
        pattern = r"\b[\dA-Fa-f]+_[\dA-Fa-f]+_[\dA-Fa-f]+\b"
        output = re.findall(pattern, image_path)[0]

        return output

    def gather_images(self, cubes, df_ids):
        """ Get images from the directory and put them into list """
        image_paths = []
        for cube_path in cubes:
            for root, dirs, files in os.walk(f'{self.research_name}/experiments'):
                if root.endswith('inference') and self.find_id_in_string(root) in df_ids:
                    img = os.path.join(root, cube_path)
                    image_paths.append(img)

        return image_paths

    def find_feature_in_df(self, feature):
        """ Form the metainformation about each image """
        meta_info = []
        for cube in self.cubes:
            for id_ in self.df_ids:
                meta_info_ = '/'.join([f"{self.df_with_idx.loc[id_, feature]}", f"{self.df_with_idx.loc[id_, 'repetition']}", cube])
                meta_info.append(meta_info_)

        return meta_info

    def get_ids_from_df(self, df, rep):
        """ Get ids from research dataframe corresponding to arbitrary research repetition """
        return list(df[df['repetition'] == rep]['id'])

    @property
    def get_images_and_paths(self):
        """ Return list of image_paths, list with their meta info, and dict with them """
        return self.image_paths, self.meta_info, self.images_with_paths

class ScrollingImagesWidget:
    """ Interactive widget for scrolling images """
    def __init__(self, image_paths, dropdown_list, images_with_paths, width=700, height=700):
        self.idx = 0
        self.image_paths = image_paths
        self.dropdown_list = dropdown_list
        self.images_with_paths = images_with_paths

        # Make widgets
        self.image_widget = widgets.Image(format='png',
                                          width=width, height=height,
                                          value=open(self.image_paths[self.idx], 'rb').read())
        self.next_button = widgets.Button(description='Next')
        self.previous_button = widgets.Button(description='Prev')
        self.dropdown = widgets.Dropdown(options=self.dropdown_list, value=self.dropdown_list[self.idx])
        self.title = widgets.Label(value=self.dropdown_list[self.idx])

        hbox = widgets.HBox([self.previous_button, self.next_button, self.dropdown])
        vbox = widgets.VBox([hbox, self.image_widget, self.title])
        result_box = vbox

        # Setup widgets
        self.next_button.on_click(self.next_button_click)
        self.dropdown.observe(self.dropdown_callback, names='value')
        self.previous_button.on_click(self.prev_button_click)

        display(result_box)

    def next_button_click(self, b):
        """ Change path to image if the `next` button is clicked """
        self.idx += 1
        if self.idx == len(self.image_paths):
            self.idx = 0
        self.redraw(self.idx, self.image_paths[self.idx])

    def prev_button_click(self, b):
        """ Change path to image if the `prev` button is clicked """
        self.idx -= 1
        self.redraw(self.idx, self.image_paths[self.idx])

    def dropdown_callback(self, change):
        """ Change path to image while changing the value in dropdown """
        file_path = self.images_with_paths[change.new]
        self.idx = self.image_paths.index(file_path)
        self.redraw(self.idx, file_path)

    def redraw(self, idx, file_path):
        """ Reopen the image and change the title/value of the widgets """
        self.image_widget.value = open(file_path, 'rb').read()
        new_title = self.dropdown_list[idx]
        self.title.value = new_title
        self.dropdown.value = self.dropdown_list[idx]