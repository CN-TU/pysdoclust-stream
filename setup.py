#!/usr/bin/env python3

import glob
import os
import pathlib

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension

parent_path = pathlib.Path(__file__).parent

try:
    import numpy as np
except:
    raise ImportError('Numpy is required for building this package.', name='numpy')

def get_numpy_include():
    """Handle NumPy 1.x vs 2.0+ include paths"""
    include_path = np.get_include()
    
    # Check if arrayobject.h exists in the detected path
    arrayobject_h = os.path.join(include_path, 'numpy', 'arrayobject.h')
    if os.path.exists(arrayobject_h):
        return include_path
    
    # Fallback for NumPy 2.0+ if not found in standard location
    legacy_path = include_path.replace('_core/include', 'core/include')
    if os.path.exists(os.path.join(legacy_path, 'numpy', 'arrayobject.h')):
        return legacy_path
    
    # Final fallback to numpy/core/include (older versions)
    return os.path.join(os.path.dirname(np.__file__), 'core', 'include')

# numpy_path = os.path.dirname(numpy.__file__)
# numpy_include = numpy_path + '/core/include'
# numpy_include = numpy.get_include().replace('_core/include', 'core/include')  # Backward compat

CPP_SOURCES = [
    'swig/clustering_wrapper.cpp',
    'swig/SDOstreamclust_wrap.cxx'
]

SDOstreamclust_cpp = Extension(
    'SDOstreamclust.swig._SDOstreamclust',
    CPP_SOURCES,
    include_dirs=[
        'cpp',
        get_numpy_include(),  # Use the dynamic path resolver
        'contrib/boost/include'
    ],
    # include_dirs=['cpp', numpy_include, 'contrib/boost/include'],
    extra_compile_args=['-g0']
)

setup(
    name='pysdoclust-stream',
    version='0.3.1',
    license='LGPL-3.0',
    description='SDOstreamclust is an algorithm for clustering data streams',
    author='Simon Konzett',
    author_email='konzett.simon@gmail.com',
    url='https://github.com/CN-TU/pysdoclust-stream',
    packages=['SDOstreamclust', 'SDOstreamclust.swig'],
    package_dir={'SDOstreamclust': 'python', 'SDOstreamclust.swig': 'swig'},
    ext_modules = [ SDOstreamclust_cpp ],
    # install_requires=['numpy'],
    # python_requires='>=3.5',
    install_requires=['numpy>=1.16.0'],  # Minimum version with stable ABI
    python_requires='>=3.7',
    setup_requires=['numpy>=1.16.0']  # Ensure numpy is available during setup
)
