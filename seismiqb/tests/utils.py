""" Tests helper functions."""
import os
import shutil

def remove_paths(paths=None):
    """ Safely remove directories and files.

    Parameters
    ----------
    paths: str or list of str
        A list of paths to directories and files to remove.
    """
    paths = paths or []
    if isinstance(paths, str):
        paths = [paths]

    for path in paths:
        if os.path.exists(path):
            if os.path.isdir(path):
                # This try-except block is required for Windows tests
                try:
                    shutil.rmtree(path)
                except OSError as e:
                    print(f"Can't delete the directory {path} : {e.strerror}")
            else:
                os.remove(path)
