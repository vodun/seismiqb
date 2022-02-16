"""Init file.
Also disables the OMP warnings, which are produced by Numba or Tensorflow and can't be disabled otherwise.
The change of env variable should be before any imports, relying on it, so we place it on top.
"""
#pylint: disable=wrong-import-position
import os
os.environ['KMP_WARNINGS'] = '0'

from . import batchflow
from .src import * # pylint: disable=wildcard-import

__version__ = '0.1.0'
