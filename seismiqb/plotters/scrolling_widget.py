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
    def __init__(self, research_name, repetition, combine=False):
        self.repetition = repetition

        # Find cube paths 
        subdir = os.listdir(f'./{research_name}/experiments')[0]
        if not combine:
            cube_paths = glob.glob(f'./{research_name}/experiments/{subdir}/inference/*/[!combine]*.png', recursive=True)
        else:
            cube_paths = glob.glob(f'./{research_name}/experiments/{subdir}/inference/*/*.png', recursive=True)
        cubes = list(map(self.find_cube_path, cube_paths))

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
        self.image_paths = self.gather_images(cubes, research_name)

    def find_cube_path(self, string):
        """ Find cube path in a filename """
        pattern = r'inference(.*)'
        match = re.search(pattern, string)
        return match.group(1)[1:]

    def find_repetition_in_id(self, id):
        """ Find repetition in id """
        pattern = r'_(.*?)_'
        matches = re.findall(pattern, id)
        return matches[0]

    def find_id_in_path(self, image_path):
        """ Find id pattern in a file name of the research experiment corresponding to the given repetition """
        pattern = r"\b[\dA-Fa-f]+_[\dA-Fa-f]+_[\dA-Fa-f]+\b"
        output = re.findall(pattern, image_path)[0]
        repetition = self.find_repetition_in_id(output)
        return repetition == self.repetition

    def gather_images(self, cubes, research_name):
        """ Get images from the directory and put them into list """
        image_paths = []
        for cube_path in cubes:
            images = glob.glob(pathname=f'./{research_name}/**/inference/{cube_path}', recursive=True)
            image_paths.extend(images)
        image_paths = list(filter(self.find_id_in_path, image_paths))
        return image_paths

    def make_metainfo(self, feature, cubes, df_ids, df):
        """ Form the metainformation about each image """
        meta_info = []
        for cube in cubes:
            for id_ in df_ids:
                meta_info_ = '/'.join([f"{df.loc[id_, feature]}", f"{self.repetition}", cube])
                meta_info.append(meta_info_)
        return meta_info

    @property
    def get_paths(self):
        """ Return list of paths and list with their meta info """
        return self.image_paths, self.meta_info

class ScrollingImagesWidget:
    """ Interactive widget for scrolling images """
    def __init__(self, paths, names, figsize=(20, 20)):
        self.idx = 0
        self.dropdown_list = names
        self.figures = self.make_figures(paths, figsize)
        self.figures_with_info = dict(zip(names, self.figures))

        # Make widgets
        self.out = widgets.Output()
        with self.out:
            display(self.figures[self.idx])
        self.next_button = widgets.Button(description='Next', layout=widgets.Layout(width='20%'))
        self.prev_button = widgets.Button(description='Prev', layout=widgets.Layout(width='20%'))
        self.dropdown = widgets.Dropdown(options=names, value=names[self.idx],
                                    layout=widgets.Layout(width='400px', height='30px', left='30%'))

        # Setup widgets
        self.next_button.on_click(self.next_button_click)
        self.prev_button.on_click(self.prev_button_click)
        self.dropdown.observe(self.dropdown_callback, names='value')

        # Make the box
        hbox = widgets.HBox([self.prev_button, self.next_button, self.dropdown])
        vbox = widgets.VBox([hbox, self.out])
        display(vbox)

    def make_figures(self, paths, figsize):
        """ Iterate over image paths and make the list with figures from the images """
        figures = []
        for i, file in enumerate(paths):
            fig, ax = plt.subplots(figsize=figsize)
            handles = mpatches.Patch(color='salmon', label=self.dropdown_list[i])
            ax.imshow(Image.open(file))
            ax.axis('off')
            ax.legend(handles=[handles], bbox_to_anchor=(0.0, 0.0), loc='upper left')
            figures.append(fig)
            plt.close()

        return figures

    def next_button_click(self, button):
        """ Change path to image if the `next` button is clicked """
        self.idx = (self.idx + 1) % len(self.figures)
        self.refresh(self.idx, self.figures[self.idx])

    def prev_button_click(self, button):
        """ Change path to image if the `prev` button is clicked """
        self.idx -= 1
        self.refresh(self.idx, self.figures[self.idx])

    def dropdown_callback(self, change):
        """ Change path to image while changing the value in dropdown """
        figure = self.figures_with_info[change.new]
        self.idx = self.dropdown_list.index(change.new)
        self.refresh(self.idx, figure)

    def refresh(self, idx, figure):
        """ Clear the current figure and display the new one with new value in dropdown """
        with self.out:
            self.out.clear_output(wait=True)
            display(figure)
        self.dropdown.value = self.dropdown_list[idx]