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
    import numpy
except:
    raise ImportError('Numpy is required for building this package.', name='numpy')

numpy_path = os.path.dirname(numpy.__file__)
numpy_include = numpy_path + '/core/include'

CPP_SOURCES = [
    'swig/clustering_wrapper.cpp',
    'swig/SDOclustream_wrap.cxx'
]

SDOclustream_cpp = Extension(
    'SDOclustream.swig._SDOclustream',
    CPP_SOURCES,
    include_dirs=['cpp', numpy_include, 'contrib/boost/include'],
    extra_compile_args=['-g0']
)

setup(
    name='SDOclustream',
    version='0.1',
    license='LGPL-3.0',
    description='SDOclustream is an algorithm for clustering data streams',
    author='Simon Konzett',
    author_email='konzett.simon@gmail.com',
    # url='https://github.com/CN-TU/dSalmon',
    # project_urls={
    #     'Source': 'https://github.com/CN-TU/dSalmon',
    #     'Documentation': 'https://dSalmon.readthedocs.io',
    #     'Tracker': 'https://github.com/CN-TU/dSalmon/issues'
    # },
    # long_description=(parent_path / 'README.rst').read_text(encoding='utf-8'),
    # long_description_content_type='text/x-rst',
    # classifiers=[
    #     'Development Status :: 4 - Beta',
    #     'Intended Audience :: Developers',
    #     'Intended Audience :: Science/Research',
    #     'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
    #     'Programming Language :: C++',
    #     'Programming Language :: Python :: 3',
    #     'Programming Language :: Python :: Implementation :: CPython',
    #     'Topic :: Scientific/Engineering'
    # ],
    packages=['SDOclustream', 'SDOclustream.swig'],
    package_dir={'SDOclustream': 'python', 'SDOclustream.swig': 'swig'},
    ext_modules = [ SDOclustream_cpp ],
    install_requires=['numpy'],
    python_requires='>=3.5'
)
