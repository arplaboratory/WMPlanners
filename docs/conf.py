"""Sphinx configuration for ds-ga-3001 documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
# sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
# sys.path.insert(0, os.path.join(PROJECT_ROOT, "external"))

project = "ds-ga-3001"
author = "Long Quang"
copyright = f"{datetime.now():%Y}, {author}"
release = "1.0.0"

extensions = [
    "autoapi.extension",
    "myst_nb",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
]

autosummary_generate = False

# -- autoapi configuration ---------------------------------------------------
autoapi_dirs = ["../src", "../external/vjepa2"]

autoapi_type = "python"
autoapi_root = "technical_reference"
autoapi_add_toctree_entry = False
autoapi_keep_files = True
autoapi_python_use_implicit_namespaces = True

# Control what gets documented
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    # "show-module-summary",
    "special-members",
    # "imported-members",
]

def skip_submodules(app, what, name, obj, skip, options):
    # Identifies the specific duplicated attributes in your mdp configs
    # and tells Sphinx to skip them if they are being processed a second time.
    if name.endswith(".class_type") or name.endswith(".scale") or name.endswith(".offset"):
        return True
    return skip

def setup(app):
    app.connect("autoapi-skip-member", skip_submodules)

suppress_warnings = [
    "autoapi.python_import_resolution",
    "docutils",
    "toc.not_included",
    "ref.ref",
    "ref.python"
]

bibtex_bibfiles = ["refs.bib"]

nb_execution_mode = "off"

# html_theme = "sphinx_book_theme"
html_title = project

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
}

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]