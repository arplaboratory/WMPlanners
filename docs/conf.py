"""Sphinx configuration for cjepa documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

project = "cjepa"
author = "Long Quang"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
]

autosummary_generate = True

nb_execution_mode = "off"

html_theme = "sphinx_book_theme"
html_title = project

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]
