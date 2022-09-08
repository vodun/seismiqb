""" A class for working with seismic data. """
from .base import Geometry
from .segyio_loader import SegyioLoader, SafeSegyioLoader
from .memmap_loader import MemmapLoader
from .segy import GeometrySEGY
