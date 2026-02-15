"""Sphinx configuration for cjepa documentation."""

from __future__ import annotations

import os
import sys
from datetime import datetime
from unittest.mock import MagicMock

PROJECT_ROOT = os.path.abspath(os.path.join(__file__, "../.."))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "external"))

project = "cjepa"
author = "Long Quang"
copyright = f"{datetime.now():%Y}, {author}"
release = "0.1.0"

autodoc_mock_imports = [
    "pypylon",
    "gymnasium",
    "cv2",
]

class SmartMock(MagicMock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__path__ = []
    def __getattr__(self, name): return SmartMock()
    def __call__(self, *args, **kwargs): return SmartMock()
    def __iter__(self): return iter([SmartMock(), SmartMock(), SmartMock()])

mock_obj = SmartMock()
sys.modules["isaaclab"] = mock_obj
sys.modules["isaaclab.envs"] = mock_obj

# class SmartMock(MagicMock):
#     """Allows Mock() | None"""
#     def __getattr__(self, name):
#         if name.startswith('_'): # Protect internal Python/Mock attributes
#             raise AttributeError(name)
#         return SmartMock()

#     def __iter__(self):
#         return iter([SmartMock(), SmartMock(), SmartMock()])

#     @property
#     def __path__(self):
#         return []

# mock_modules = ["torch", "isaaclab", "isaaclab_tasks", "pxr", "omni"]
# for mod_name in mock_modules:
#     sys.modules[mod_name] = SmartMock()

# class SmartMock(MagicMock):
#     """Allows Mock() | None"""
#     def __or__(self, other):
#         return self
#     def __ror__(self, other):
#         return self

# mock_modules = ["isaaclab", "isaaclab_tasks", "torch", "pxr", "omni"]
# for mod_name in mock_modules:
#     mock_obj = SmartMock()
#     mock_obj.__path__ = []
#     sys.modules[mod_name] = mock_obj

# extra_mocks = [
#     "isaaclab_tasks.utils",
#     "isaaclab.utils",
#     "torch.nn",
#     "torch.optim"
# ]
# for mod_name in extra_mocks:
#     sys.modules[mod_name] = SmartMock()

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["refs.bib"]
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
