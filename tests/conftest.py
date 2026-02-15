import sys
from unittest.mock import MagicMock

# Trick Python into thinking these exist
MOCK_MODULES = [
    "isaaclab",
    "isaaclab_tasks",
    "isaaclab_tasks.utils",
    "pxr",
    "omni",
    "omni.isaac.core",
    "torch",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
