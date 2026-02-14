import os

WHEELED_ASSETS_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATA_DIR = os.path.abspath(
    os.path.join(WHEELED_ASSETS_EXT_DIR, "..", "..", "..", "..", "..", "WheeledLab", "source", "wheeledlab_assets", "data")
)
WHEELED_ASSETS_DATA_DIR = os.environ.get("RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR", DEFAULT_DATA_DIR)

from .mushr import *
from .f1tenth import *
