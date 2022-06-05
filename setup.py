#!/usr/bin/env python
import os
from setuptools import setup

version = os.environ.get('VERSION', '0.0.1')

setup(
    name='macorp.forecast',
    description='MaCorp Forecast toolkit.',
    version=version,
    author='Marzie Saghayi',
    author_email='',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: MacOS X',
        'Environment :: Win32 (MS Windows)',
        'Environment :: X11 Applications',
        'Intended Audience :: Developers',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    install_requires=[
        'numpy>=1.19',
        'pandas>=1.2',
        'click>=1.7',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.1.3',
        'holidays',
    ],
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'macorp.forecast = macorp.forecast.cli:main',
        ]
    }
)
