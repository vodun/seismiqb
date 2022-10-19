""" seismiQB is a framework for deep learning research on 3d-cubes of seismic data. """

from setuptools import setup, find_packages
import re

with open('__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='seismiQB',
    packages=find_packages(exclude=['tutorials', 'datasets', 'models']),
    version=version,
    url='https://github.com/gazprom-neft/seismiqb',
    license='Apache 2.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='An ML framework for research on volumetric seismic data',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        # General Python libraries
        'dill>=0.3.1.1',
        'psutil>=5.6.7',
        'requests>=2.22.0',
        'pytest>=5.3.1',

        # Numerical
        'numpy>=1.16.0',
        'bottleneck>=1.3',
        'numexpr>=2.7',
        'numba>=0.43.0',
        'scipy>=1.3.3',
        'scikit-learn>=0.21.3',
        'scikit_image>=0.16.2',
        'connected-components-3d>=3.10.2',

        # Data manipulation
        'pandas>=1.0.0',
        'dask[dataframe]>=2.8.1',
        'feather_format>=0.4.0',
        'blosc>=1.8.1',
        'segyio>=1.8.3',
        'h5py>=2.10.0',
        'h5pickle>=0.2.0',
        'hdf5plugin>=3.3.0',

        # Working with images
        'opencv_python>=4.1.2.30',
        'matplotlib>=3.0.2',
        'plotly>=4.3.0',
        'seaborn>=0.9.0',
        'Pillow>=8.0.1',

        # Jupyter and introspection
        'tqdm>=4.50.0',
        'nbconvert>=5.6.1',
        'ipython>=7.10.0',
        'ipywidgets>=7.0',
        'nvidia_smi>=0.1.3',
        'nvidia-ml-py3>=7.3',

        # Our libraries
        'batchflow==0.7.5',
        'py-nbtools>=0.9.5',
    ],
    extras_require={
        'nn': [
            'torch>=1.7.0',
            'torchvision>=0.1.3',
            'cupy>=8.1.0',
        ],
        'cupy': [
            'cupy>=8.1.0'
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
)
