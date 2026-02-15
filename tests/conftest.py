import sys
from unittest.mock import MagicMock

def mock_package(name):
    """Recursively mocks a package path."""
    components = name.split('.')
    for i in range(len(components)):
        path = '.'.join(components[:i+1])
        if path not in sys.modules:
            mock_obj = MagicMock()
            mock_obj.__path__ = []
            sys.modules[path] = mock_obj

# Trick Python into thinking these exist
MOCK_MODULES = [
    "isaaclab",
    "issaclab.sim",
    "isaaclab.sim.robots",
    "issaclab.assets",
    "isaaclab.assets.robots",
    "isaaclab.assets.robots.unitree",
    "isaaclab.manager",
    "isaaclab.envs",
    "isaaclab.envs.mdp",
    "isaaclab.utils",
    "isaaclab.utils.data",
    "isaaclab.utils.visualization",
    "isaaclab.utils.math",
    "issaclab.utils.noise",
    "isaaclab_tasks",
    "isaaclab_tasks.utils",
    "isaaclab.terrains",
    "isaaclab.scenes",
    "isaaclab.sensors",
    "isaaclab.markers",
    "isaaclab.markers.config",
    "pxr",
    "omni",
    "omni.isaac.core",
    "gymnasium",
    "torch",
]

for mod_name in MOCK_MODULES:
    sys.modules[mod_name] = MagicMock()
