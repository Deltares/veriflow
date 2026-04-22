import os
import sys

from veriflow.constants import VERSION

sys.path.insert(0, os.path.abspath("../../src"))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "veriflow"
copyright = "2026, Deltares"
author = "Jurian Beunk"

release = VERSION

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google/NumPy docstrings
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "myst_parser",  # Markdown support
    "nbsphinx",
]

myst_enable_extensions = ["colon_fence"]

templates_path = ["_templates"]

exclude_patterns = []

# Links to external documentation pages
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
    "scores": ("https://scores.readthedocs.io/en/stable/", None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_title = "Verification "
html_theme_options = {
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],
    # "switcher": {
    #     "json_url": "_static/versions.json",  # see below
    #     "version_match": "current",
    # },
    "navigation_depth": 3,
}

# Autodoc
autosummary_generate = True
autosummary_ignore_module_all = False
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "exclude-members": "model_config",
    "members": True,
    "undoc-members": True,  # show StrEnum members from constants
}


# nbshpinx
nb_execution_mode = "off"  # or "auto"
nbsphinx_execute = "auto"  # options: 'auto', 'always', 'never'
nbsphinx_kernel_name = "python3"  # kernel to use for notebook execution
nbsphinx_timeout = 600  # seconds per notebook
