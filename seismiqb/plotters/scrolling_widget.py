import ipywidgets as widgets
import os
import re
import glob

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from IPython.display import display

import pandas as pd

from batchflow.research.results import ResearchResults

class GatherImagesFromResearch:
    """ Auxiliary class gathering images corresponding to the research experiments, making
    the metainformation about each image and preparing figures from the images
    """
    def __init__(self, cubes, research_name, repetition):
        self.repetition = repetition

        # Open research dataframe and find ids corresponding to the repetition
        df_path = glob.glob(f'./{research_name}/*.csv')[0]
        research_df = pd.read_csv(df_path)
        research_df = research_df.set_index('id')
        df_ids = list(research_df[research_df['repetition'] == int(repetition)].index)

        # Find the research feature in the research configs
        res = ResearchResults(research_name)
        res.load_configs()
        feature = list(list(res.configs.values())[0].keys())[0]

        # Make metainformation from the research dataframe
        self.meta_info = self.make_metainfo(feature, cubes, df_ids, research_df)

        # Gather images from the research files
        image_paths = self.gather_images(cubes, research_name)
        self.figures = self.make_figures(image_paths)

    def find_id_in_string(self, image_path):
        """ Find id pattern in a file name of the research experiment corresponding to the given repetition """
        pattern = r"\b[\dA-Fa-f]+_[\dA-Fa-f]+_[\dA-Fa-f]+\b"
        output = re.findall(pattern, image_path)[0]
        return output[2] == self.repetition

    def gather_images(self, cubes, research_name):
        """ Get images from the directory and put them into list """
        image_paths = []
        for cube_path in cubes:
            images = glob.glob(pathname=f'./{research_name}/**/inference/{cube_path}', recursive=True)
            image_paths.extend(images)
        image_paths = list(filter(self.find_id_in_string, image_paths))
        return image_paths

    def make_figures(self, image_paths):
        """ Iterate over image paths and make the list with figures from the images """
        figures = []
        for i, file in enumerate(image_paths):
            fig, ax = plt.subplots(figsize=(20, 20))
            handles = mpatches.Patch(color='salmon', label=self.meta_info[i])
            ax.imshow(Image.open(file))
            ax.axis('off')
            ax.legend(handles=[handles], bbox_to_anchor=(0.0, 0.0), loc='upper left')
            figures.append(fig)
            plt.close()

        return figures

    def make_metainfo(self, feature, cubes, df_ids, df):
        """ Form the metainformation about each image """
        meta_info = []
        for cube in cubes:
            for id_ in df_ids:
                meta_info_ = '/'.join([f"{df.loc[id_, feature]}", f"{self.repetition}", cube])
                meta_info.append(meta_info_)

        return meta_info

    @property
    def get_figures(self):
        """ Return list of figures and list with their meta info """
        return self.figures, self.meta_info

class ScrollingImagesWidget:
    """ Interactive widget for scrolling images """
    def __init__(self, image_figures, dropdown_list):
        self.idx = 0
        self.image_figures = image_figures
        self.dropdown_list = dropdown_list
        self.figures_with_info = dict(zip(dropdown_list, image_figures))

        # Make widgets
        self.out = widgets.Output()
        with self.out:
            display(self.image_figures[self.idx])
        self.next_button = widgets.Button(description='Next')
        self.prev_button = widgets.Button(description='Prev')
        self.dropdown = widgets.Dropdown(options=dropdown_list, value=dropdown_list[self.idx],
                                    layout=widgets.Layout(width='400px', height='30px', left='43%'))

        # Setup widgets
        self.next_button.on_click(self.next_button_click)
        self.prev_button.on_click(self.prev_button_click)
        self.dropdown.observe(self.dropdown_callback, names='value')

        # Make the box
        hbox = widgets.HBox([self.prev_button, self.next_button, self.dropdown])
        vbox = widgets.VBox([hbox, self.out])
        display(vbox)

    def next_button_click(self, button):
        """ Change path to image if the `next` button is clicked """
        self.idx = (self.idx + 1) % len(self.image_figures)
        self.update(self.idx, self.image_figures[self.idx])

    def prev_button_click(self, button):
        """ Change path to image if the `prev` button is clicked """
        self.idx -= 1
        self.update(self.idx, self.image_figures[self.idx])

    def dropdown_callback(self, change):
        """ Change path to image while changing the value in dropdown """
        figure = self.figures_with_info[change.new]
        self.idx = self.dropdown_list.index(change.new)
        self.update(self.idx, figure)

    def update(self, idx, figure):
        """ Clear the current figure and display the new one with new value in dropdown """
        with self.out:
            self.out.clear_output(wait=True)
            display(figure)
        self.dropdown.value = self.dropdown_list[idx]