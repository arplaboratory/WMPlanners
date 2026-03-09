# DS-GA 3001
This repo supports the DS-GA 3001 - Embodied Learning and Vision Course Project

| Authors              | Email                |
|-------------------|----------------------|
| **Long Quang**    | lq2146@nyu.edu       |
| **Akash Guna** | ar9799@nyu.edu     |
| **Deepanshu Mody** | dm6262@nyu.edu      |

<sup>
Authors are with New York University.
</sup>

[![Tests](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/tests.yml/badge.svg)](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/tests.yml)
![Pylint](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/lquan9/d09cfb9f249c9995b9fa1943d90a1d71/raw/ds3001-pylint.json)
[![Docs](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/docs.yml/badge.svg)](https://github.com/lquan9/NYU_DS_GA_3001/actions/workflows/docs.yml)
[![Python Version](https://img.shields.io/badge/python-%203.11-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Code](#) | [Paper](#)

`src/cjepa` is a lightweight integration layer for adapting continual learning to JEPA.
<br>
`external/vjepa2` official [V-JEPA 2](https://github.com/facebookresearch/vjepa2) repository.
<br>
`src/rloclab` package provides an Isaac Lab reinforcement learning workflow for Unitree Go2 locomotion and wheeled robots.

## Inspiration and preliminary works

The `rloclab` package builds on and cites the following repositories as upstream references and preliminary implementations:

- **isaac-sim/IsaacLab**: core simulator/task framework and APIs that power `rloclab` [Link](https://github.com/isaac-sim/IsaacLab).  
- **machines-in-motion/rob6323_go2_project**: original course starter workflow and task framing for Go2 in Isaac Lab [Link](https://github.com/machines-in-motion/rob6323_go2_project).
  
- **wheeledlab/wheeledlab**: reference for wheeled drifting task structure and asset conventions integrated in this package [Link](https://github.com/wheeledlab/wheeledlab).

# Quick Start
## Pre-requisites
1. [Git LFS](https://git-lfs.github.com/)
2. [Python Virtual Env](https://docs.python.org/3/tutorial/venv.html)
3. [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html)
4. [UV](https://docs.astral.sh/uv/getting-started/installation/) (optional)

<details>
<summary><strong>Click for instructions on installing prerequisites</strong></summary>

---
**1. Git-LFS**

We rely on GIT Large File Storage for handling binary files that are required for tracking assets.

```bash
sudo apt install git-lfs
git lfs install
```

**2. Python3 Virtual Environment**
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**3. IsaacLab** (Installed within Python environment)

**4. UV**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
---
</details>

### Setup Python workspace
> The bundled V-JEPA 2 project in `external/vjepa2` targets Linux environments like
> Ubuntu 24.04 and commonly requires CUDA + multi-GPU for full-scale training.

```bash
python3.11 -m venv .ds3001_env
source .ds3001_env/bin/activate
```
or (using **uv**)
```bash
uv venv .ds3001 --python 3.11
source .ds3001/bin/activate
```

### Install dependencies
```bash
pip install pip --upgrade
# V-JEPA2 runtime dependencies
pip install -r external/vjepa2/requirements.txt

# Project dependencies
pip install -e .[dev,docs]
```
<sup>
If using <b>zsh</b>, you may need to run: <code>setopt nonomatch</code> to set glob behavior properly.
</sup>

### Run
```bash
dino --config configs/config.yaml
```
Run the example notebook by launching Jupyter Lab and opening
``docs/notebooks/getting_started.md``.  Build the documentation locally with:

```bash
sphinx-build -b html docs docs/_build/html
```

Execute the tests:

```bash
pytest
```

#### Headless
```bash
TBD
```


## Extra documentation

See `docs/getting_started.md` for a concise step-by-step.

## Resources
- [Isaac Lab documentation](https://isaac-sim.github.io/IsaacLab/main/source/setup/ecosystem.html) — Everything you need to know about IsaacLab, and more!
- [Isaac Lab ANYmal C environment](https://github.com/isaac-sim/IsaacLab/tree/main/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c) — This targets ANYmal C (not Unitree Go2), so use it as a reference and adapt robot config, assets, and reward to Go2.
- [DMO (IsaacGym) Go2 walking project page](https://machines-in-motion.github.io/DMO/) • [Go2 walking environment used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/tasks/go2_terrain.py) • [Config file used by the authors](https://github.com/Jogima-cyber/IsaacGymEnvs/blob/e351da69e05e0433e746cef0537b50924fd9fdbf/isaacgymenvs/cfg/task/Go2Terrain.yaml) — Look at the function `compute_reward_CaT` (beware that some reward terms have a weight of 0 and thus are deactivated, check weights in the config file); this implementation includes strong reward shaping, domain randomization, and training disturbances for robust sim‑to‑real, but it is written for legacy IsaacGym and the challenge is to re-implement it in Isaac Lab.
- **API References**:
    - [ArticulationData (`robot.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.assets.html#isaaclab.assets.ArticulationData) — Contains `root_pos_w`, `joint_pos`, `projected_gravity_b`, etc.
    - [ContactSensorData (`_contact_sensor.data`)](https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sensors.html#isaaclab.sensors.ContactSensorData) — Contains `net_forces_w` (contact forces).
---

## License

This project is licensed under the MIT license. See [LICENSE](LICENSE).
