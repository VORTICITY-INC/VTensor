# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import sys
import warnings
from sphinx.deprecation import RemovedInSphinx90Warning

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())
warnings.filterwarnings("ignore", category=RemovedInSphinx90Warning)
project = 'VTensor'
copyright = '2024, Vorticity Inc'
author = 'Sheng-Yang Tsui'
release = 'v1.0.0'


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
            'breathe',
            'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.doctest',
              'sphinx.ext.extlinks',
              'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax',
              'sphinx.ext.napoleon',
              'sphinx_copybutton']

breathe_projects = {
    "vtensor": "../build/xml"
}

breathe_default_project = "vtensor"



templates_path = ['_templates']
exclude_patterns = []


pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_logo = '../image/logo_black.png'