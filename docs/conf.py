"""Sphinx configuration for the World Model Planners guides."""

from datetime import datetime

project = "World Model Planners"
author = "Long Quang"
copyright = f"{datetime.now():%Y}, {author}"
release = "1.0.0"

extensions = [
    "myst_parser",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown"}
myst_fence_as_directive = ["mermaid"]
master_doc = "index"
bibtex_bibfiles = ["refs.bib"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
html_title = project
