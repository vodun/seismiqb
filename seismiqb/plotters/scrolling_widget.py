import ipywidgets as widgets
import os
import glob

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from IPython.display import display

import pandas as pd

from batchflow.research.results import ResearchResults

def gather_images_from_research(research_name, cubes,
                                cube_paths='./{research_name}/experiments/{experiment_id}/inference/{cube}',
                                repetition=None):
    """ Auxiliary function gathering images corresponding to the research experiments and making
    the metainformation about each image
    """
    # Make research dataframe
    results = ResearchResults(name=research_name)
    research_df = results.to_df(use_alias=True, remove_auxilary=False)
    research_df.repetition = pd.to_numeric(research_df.repetition)
    df_with_idx = research_df.set_index('id')
    feature = research_df.columns[1]

    # Iterate over repetetitions and cubes, and gather images from the research files
    # with making metainformation about them
    images = []
    meta_info = []
    repetitions = [repetition] if repetition is not None else range(research_df.repetition.max() + 1)
    for repetition in repetitions:
        for cube in cubes:
            for _, row in research_df[research_df.repetition == repetition].iterrows():
                experiment_id = row['id']
                experiment_images = cube_paths.format(research_name=research_name,
                                                      experiment_id=experiment_id,
                                                      cube=cube)
                experiment_images = glob.glob(experiment_images, recursive=True)
                if experiment_images:
                    images.extend(experiment_images)
                    meta_info_ = '/'.join([f"{df_with_idx.loc[experiment_id, feature]}",
                                           f"{repetition}", cube])
                    meta_info.append(meta_info_)

    return images, meta_info

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
        self.next_button = widgets.Button(description='Next', layout=widgets.Layout(width='18%'))
        self.prev_button = widgets.Button(description='Prev', layout=widgets.Layout(width='18%'))
        dropdown_layout = widgets.Layout(width='28%', height='30px', left='33%', align_items='center')
        self.dropdown = widgets.Dropdown(options=names, value=names[self.idx],
                                         layout=dropdown_layout)

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
        """ Change a figure if the `next` button is clicked """
        self.idx = (self.idx + 1) % len(self.figures)
        self.refresh(self.idx, self.figures[self.idx])

    def prev_button_click(self, button):
        """ Change a figure if the `prev` button is clicked """
        self.idx -= 1
        self.refresh(self.idx, self.figures[self.idx])

    def dropdown_callback(self, change):
        """ Switch a figure while changing the value in dropdown """
        figure = self.figures_with_info[change.new]
        self.idx = self.dropdown_list.index(change.new)
        self.refresh(self.idx, figure)

    def refresh(self, idx, figure):
        """ Clear the current figure and display the new one with new value in dropdown """
        with self.out:
            self.out.clear_output(wait=True)
            display(figure)
        self.dropdown.value = self.dropdown_list[idx]