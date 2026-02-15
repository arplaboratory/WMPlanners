"""Sphinx configuration for cjepa documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external"))

project = "cjepa"
author = "Long Quang"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "autoapi.extension",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

# -- autoapi configuration ---------------------------------------------------
autoapi_dirs = ["../src", "../external/vjepa2"]

autoapi_type = "python"
autoapi_root = "technical_reference"
autoapi_add_toctree_entry = False
autoapi_keep_files = True

# Control what gets documented
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "special-members",
    "imported-members",
]

bibtex_bibfiles = ["refs.bib"]

nb_execution_mode = "off"

html_theme = "sphinx_book_theme"
html_title = project

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]
