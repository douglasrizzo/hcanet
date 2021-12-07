#!/usr/bin/env python3

from os import path

from setuptools import find_packages, setup

from hcanet import __version__

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
   long_description = f.read()

setup(
   classifiers=[
      # How mature is this project? Common values are
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      "Development Status :: 3 - Alpha",
      "Intended Audience :: Science/Research",
      "Natural Language :: English",
      "Programming Language :: Python :: 3.5",
      "Programming Language :: Python :: 3.6",
      "Programming Language :: Python :: 3.7",
      "Programming Language :: Python :: 3.8",
      "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)", ],
   name="hcanet",
   version=__version__,
   description="Neural Network for Heterogeneous Communicative Agents",
   long_description=long_description,
   long_description_content_type="text/markdown",
   author="Douglas De Rizzo Meneghetti",
   author_email="douglasrizzom@gmail.com",
   packages=find_packages(exclude=["contrib", "docs", "tests*"]),
   install_requires=[
      "numpy",
      "GPUtil",
      "sortedcontainers",
      "tqdm",
      "torch",
      "torch-geometric>=1.5.0",
      "pysc2 @ git+https://github.com/douglasrizzo/pysc2.git@smac-view",
      "smac @ git+https://github.com/douglasrizzo/smac.git@patch-1",
      "wandb"],
   extras_require=dict(
      dev=["isort", "black", "yapf", "pylama", "mypy"],
      testing=[
         "nose",
         "nose-cov",
         "coverage",
         "coveralls",
         "python-coveralls",
         "flake8", ],
      docs=[
         "Sphinx",
         "numpydoc",
         "sphinx_autodoc_annotation",
         "sphinx_bootstrap_theme",
         "bibtex-pygments-lexer", ],
      upload=["twine", "build"],
   ),
   license="LGPLv3",
)
