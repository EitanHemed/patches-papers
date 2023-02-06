import glob
import os
import pathlib
import shutil

import setuptools

with open("../README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setuptools.setup(
    name="po_utils",
    version="0.0.1",
    author="Eitan Hemed",
    author_email="eitan.hemed@gmail.com",
    description="Utilities for analysis of the project",
    long_description='',
    long_description_content_type="text/markdown",
    url="https://github.com/EitanHemed/patches-papers",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPL-3",
        "Operating System :: OS Independent",
    ],
    packages=['po_utils'],
    python_requires="==3.9.12",
    install_requires=['robusta-stats==0.0.4', 'matplotlib', 'scipy',
                      'seaborn'],
)
