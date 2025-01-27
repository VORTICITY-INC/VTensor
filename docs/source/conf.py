# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import pathlib
import sys

sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().as_posix())

project = 'VTensor'
copyright = '2024, Vorticity Inc'
author = 'Sheng-Yang Tsui'
release = 'v1.1.0'


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
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/VORTICITY-INC/VTensor",
            "icon": "fab fa-github-square",
        },
        {
            "name": "Vorticity",
            "url": "https://vorticity.xyz/",
            "icon": "fas fa-globe",
        },
        
    ],
}
