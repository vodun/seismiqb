<div align="center">

# seismiQB

<a href="#installation">Installation</a> •
<a href="#getting-started">Getting Started</a> •
<a href="#citing-seismicpro">Citation</a>


[![License](https://img.shields.io/github/license/analysiscenter/batchflow.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Python](https://img.shields.io/badge/python-3.8-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10-green.svg)](https://pytorch.org)
[![Status](https://github.com/gazprom-neft/seismiqb/actions/workflows/status.yml/badge.svg?branch=master&event=push)](https://github.com/gazprom-neft/seismiqb/actions/workflows/status.yml)
[![Test installation](https://github.com/gazprom-neft/seismiqb/actions/workflows/test-install.yml/badge.svg?branch=master&event=push)](https://github.com/gazprom-neft/seismiqb/actions/workflows/test-install.yml)
</div>

---


**seismiQB** is a framework for research and deployment of deep learning models on post-stack seismic data. It covers all the main stages of model development and its production usage, and the main features are:

* Convert `SEG-Y` to a compressed and quantized data formats, that take 4x less disk space and load slices up to 40 times faster
* Work (e.g. load, make segmentation masks, QC) with a number of labels: horizons, faults, facies (both 2d and 3d)
* Prepare data for the model inputs, e.g. patches (2d or 3d) of seismic data and corresponding segmentation masks
* Augment model data with a number of geological and geometrical transforms, as well as traditional ML augmentations
* Define even the most complex neural networks with just a few lines of code, or re-use existing `PyTorch` models
* Optimize train/inference stages with ready-to-use primitives to meet the demands of modern field development 
* Export predicted entities (horizon, faults, facies, arrays) to convenient formats (CHARISMA, FAULT_STICKS) for validation by geophysicists


## Installation
**seismiQB** is compatible with Python 3.8+ and well tested on Ubuntu 20.04.

Installation of **seismiQB** as a Python package should be as easy as running one of the commands below, depending on your environment:

    # pipenv
    pipenv install git+https://github.com/gazprom-neft/seismiqb.git#egg=seismiqb

    # pip / pip3
    pip3 install git+https://github.com/gazprom-neft/seismiqb.git

    # developer version (add `--depth 1` if needed)
    git clone https://github.com/gazprom-neft/seismiqb.git


## Getting started

After installation just import **seismiQB** into your code. A quick demo of our primitives and methods:
```python
import seismiqb

field = Field('/path/to/cube.sgy')                                    # Initialize field with SEG-Y
field.load_labels('path/to/horizons/*.char', labels_class='horizon')  # Add labeling

# Labels
field.horizons.interpolate()                                          # Fill in small holes
field.horizons.smooth_out()                                           # Smooth out and remove spikes
field.horizons.evaluate()                                             # Compute a quality control metric

# Visualizations
field.geometry.print()                                                # Display key stats about SEG-Y
field.show_slide(index=100, axis=1)                                   # Show 100-th crossline
field.show('horizons:0/metric')                                       # Show QC metric for one horizon

```

Be sure to check out our [tutorials](tutorials) to get more info about the **seismiQB** primitives and usage.



## Citing

Please cite **seismiQB** in your publications if it helps your research.

    Khudorozhkov R., Koryagin A., Tsimfer S., Mylzenova D. SeismiQB library for seismic interpretation with deep learning. 2019.

```
@misc{seismiQB_2019,
  author       = {R. Khudorozhkov and A. Koryagin and S. Tsimfer and D. Mylzenova},
  title        = {SeismiQB library for seismic interpretation with deep learning},
  year         = 2019
}
```
