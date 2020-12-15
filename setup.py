""" seismiQB is a framework for deep learning research on 3d-cubes of seismic data. """

from setuptools import setup, find_packages
import re

with open('seismiqb/__init__.py', 'r') as f:
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', f.read(), re.MULTILINE).group(1)

setup(
    name='seismiQB',
    packages=find_packages(exclude=['tutorials', 'datasets', 'models']),
    version=version,
    url='https://github.com/gazprom-neft/seismiqb',
    license='CC BY-NC-SA 4.0',
    author='Gazprom Neft DS team',
    author_email='rhudor@gmail.com',
    description='A framework for research on volumetric seismic data',
    long_description='',
    zip_safe=False,
    platforms='any',
    install_requires=[
        'numpy>=1.16.0',
        'scipy>=1.3.3',
        'pandas>=0.24.0',
        'scikit-learn==0.21.3',
        'matplotlib>=3.0.2',
        'dill>=0.2.7.1',
        'tqdm==4.30.0',
        'segyio>=1.8.3',
        'scikit-image>=0.13.1',
        'numba>=0.43.0'
    ],
    extras_require={
        'torch': ['torch>=1.3.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering',
    ],
)