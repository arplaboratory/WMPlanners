# World Model Planners

[![Tests](https://github.com/lquan9/wm_planner/actions/workflows/tests.yml/badge.svg)](https://github.com/lquan9/wm_planner/actions/workflows/tests.yml)
[![Docs](https://github.com/lquan9/wm_planner/actions/workflows/docs.yml/badge.svg)](https://github.com/lquan9/wm_planner/actions/workflows/docs.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

World-model data processing, training, and planning for recorded robot observations. The project includes CNN, DINOv2, V-JEPA2, JEPA-lite, and DinoWM model paths, plus a ROS 2 planner bridge. External libraries and sources live under `external/`.

Check out the [project homepage](https://lquan9.github.io/wm_planner/). Documentation lives in [`docs/`](docs/index.md):

- [Getting started](docs/getting_started.md)
- [Data pipeline](docs/data_pipeline.md)
- [World models](docs/world_model.md)
- [ROS 2 wrapper](docs/ros2.md)
- [Training and evaluation](docs/training.md)

## Quick Start

### Prerequisites

1. [Git LFS](https://git-lfs.com/) for tracked binary assets.
2. Python 3.12 with virtual environment support (the system Python for ROS 2 Jazzy).
3. [uv](https://docs.astral.sh/uv/getting-started/installation/) for the project setup script.

<details>
<summary>Install prerequisites on Ubuntu</summary>

```bash
sudo apt update
sudo apt install git-lfs python3.12 python3.12-venv python3.12-dev
git lfs install
```

Install uv using its [installation instructions](https://docs.astral.sh/uv/getting-started/installation/) or:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

### Set up the local workspace

```bash
mkdir -p wm_ws/src
cd wm_ws
git clone git@github.com:lquan9/wm_planner.git src/wm_planner
export WM_WS="$PWD"
export WM_VENV="$WM_WS/src/wm_planner/.venvs/wm"
export WM_PYTHON=/usr/bin/python3.12
source /opt/ros/jazzy/setup.bash
git -C src/wm_planner submodule update --init --recursive
bash src/wm_planner/scripts/python_setup.sh
source "$WM_VENV/bin/activate"
```

The setup script creates a Python 3.12 environment with access to Jazzy's system ROS packages, then installs the Python project and its DINOv2 and V-JEPA2 external packages. See [Getting started](docs/getting_started.md) for the local ROS 2 wrapper build, and the other guides above for data preparation and training.

## License

MIT. See [LICENSE](LICENSE).