# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'nn4n'
copyright = '2025, Zhaoze Wang'
author = 'Zhaoze Wang'
release = '1.2.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',        # For processing docstrings with automodule
    'sphinx.ext.napoleon',       # For Google/NumPy-style docstrings
    'sphinx.ext.viewcode',       # To add links to source code
    'sphinx.ext.mathjax',        # To render math equations (if needed)
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']
html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_depth": 2,
}
html_favicon = "_static/favicon.ico"