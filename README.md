# DS_GA_3001
This repo supports the DS_GA_3001 - Embodied Learning and Vision Course Project

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
[![Python Version](https://img.shields.io/badge/python-%203.10-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Code](#) | [Paper](#)

`src/cjepa` is a lightweight integration layer for adapting continual learning to JEPA.
<br>
`external/vjepa2` official [V-JEPA 2](https://github.com/facebookresearch/vjepa2) repository.
<br>
`src/rloclab` package provides an Isaac Lab reinforcement learning workflow for Unitree Go2 locomotion and wheeled robots.

- V-JEPA 2 experiments scaffolding for two-phase continual learning:
  - **Task 1**: self-supervised representation learning on broad video data.
  - **Task 2**: post-training/action-conditioned adaptation on robot data.

- Utilities to build V-JEPA 2 CSV datasets (`data/0001.mp4 label`).
- A small CLI (`cjepa`) to generate a reproducible local training
  command for `external/vjepa2/app/main.py`.



## Inspiration and preliminary works

The `rloclab` package builds on and cites the following repositories as upstream references and preliminary implementations:

- **machines-in-motion/rob6323_go2_project**: original course starter workflow and task framing for Go2 in Isaac Lab.  
  https://github.com/machines-in-motion/rob6323_go2_project
- **ReghunathRahul/rob6323_go2_project**: forked implementation used as a practical baseline for reward shaping and robustness experiments before this rebrand.  
  https://github.com/ReghunathRahul/rob6323_go2_project
- **isaac-sim/IsaacLab**: core simulator/task framework and APIs that power `rloclab`.  
  https://github.com/isaac-sim/IsaacLab
- **wheeledlab/wheeledlab**: reference for wheeled drifting task structure and asset conventions integrated in this package.  
  https://github.com/wheeledlab/wheeledlab

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

## Features

- **Modern packaging** – PEP 517/518/621 ``pyproject.toml`` using Hatchling.
- **Editable installs** – ``pip install -e .[dev,docs]`` for iterative notebook
  development.
- **Documentation-first** – Sphinx configured with MyST-NB and a notebooks
  gallery ready for Read the Docs or local builds.
- **Testing** – Pytest suite to validate core functionality.

# Quick Start
## Pre-requisites
1. [Git LFS](https://git-lfs.github.com/)
2. [Python Virtual Env](https://docs.python.org/3/tutorial/venv.html)
3. [IsaacLab](https://isaac-sim.github.io/IsaacLab/main/index.html)

<details>
<summary><strong>Click for instructions on installing prerequisites</strong></summary>

---
**1. Git-LFS**

We rely on GIT Large File Storage for handling binary files that are required for building and using Phoenix. E.g., pre-compiled system dependencies that aren't available from public servers and trained neural network weights.

```bash
#To install Git-LFS
sudo apt install git-lfs
git lfs install
```

**2. Python3 Virtual Environment**
```bash
sudo apt update
sudo apt install python3.12-venv
```
---
</details>

### Setup Python workspace
> The bundled V-JEPA 2 project in `external/vjepa2` targets Linux environments like
> Ubuntu 24.04 and commonly requires CUDA + multi-GPU for full-scale training.

```bash
python -m venv .venv
source .venv/bin/activate
```
### Install dependencies
```bash
pip install -e .[dev,docs]

# V-JEPA2 runtime dependencies
pip install -r external/vjepa2/requirements.txt
```

### Run
```bash
TBD
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

## Data preparation (CSV format)

V-JEPA 2 dataset manifests are plain CSV-like text files where each row is:

```text
/path/to/video.mp4 <label>
```

Labels are ignored in self-supervised pre-training, but the column is still
required by the loader contract.

You can generate these manifests with `cjepa.write_dataset_csv`.

## CL mapping to V-JEPA 2

When adapting V-JEPA 2 as a world model, keep the following module boundaries:

- **Encoder (ViT context/target)**: latent state extraction; often frozen or
  regularized in CL to reduce catastrophic forgetting.
- **Predictor**: latent dynamics model; primary target for adaptation/expansion.
- **V-JEPA 2-AC predictor**: action-conditioned dynamics for control/robotics.

A practical CL setup is phase-based:

1. Train on general videos (Task 1).
2. Adapt to robot interaction videos (Task 2) while applying replay,
   regularization (EWC/MAS), or predictor-only expansion.

Generate a launch command:

```bash
cjepa \
  --config configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0 \
  --root-path datasets/task1 \
  --train-csv datasets/task1/train.csv \
  --val-csv datasets/task1/val.csv \
  --task-name task1_pretrain
```

Then run it from the V-JEPA 2 directory:

```bash
cd src/vjepa2
python -m app.main \
  --fname configs/train/vitl16/pretrain-256px-16f.yaml \
  --devices cuda:0 \
  --root_path ../../datasets/task1 \
  --train_csv ../../datasets/task1/train.csv \
  --val_csv ../../datasets/task1/val.csv \
  --tag task1_pretrain
```

For a task-2 adaptation pass, reuse the same command pattern with robot-specific
CSVs and a new `--tag`, then incorporate your CL method in the trainer/loss path
inside `src/vjepa2/app/vjepa/train.py` (or AC equivalent).

### Task-specfic training
Launch PPO training and automated evaluation from Greene. You may override `TASK_NAME` or pass extra flags to `scripts/rsl_rl/train.py` via `train.sh`.
```bash
cd ~/rloclab
TASK_NAME=Template-Rloclab-Go2-Direct-v0 ./train.sh --experiment_name go2_flat_direct --logger wandb
```

For other tasks
```bash
cd ~/rloclab
TASK_NAME=Template-Rloclab-Go2-<Direct|Backflip|RobustLocomotion|Bipedal>-v0 ./train.sh --experiment_name go2_<flat_direct|backflip|robust_localmotion|bipedal> --max_iterations 1000
```

### Wheeled robot RL training (WheeledLab-style drifting)

This repo now includes a wheeled stack ported from WheeledLab (Ackermann actions, wheeled asset configs, drift MDP, and PPO configs) and registers two new IsaacLab tasks:

- `Template-Rloclab-Mushr-DriftRL-v0`
- `Template-Rloclab-F1Tenth-DriftRL-v0`

Run these with the same HPC workflow:

```bash
cd ~/rloclab
TASK_NAME=Template-Rloclab-Mushr-DriftRL-v0 ./train.sh --experiment_name mushr_drift
TASK_NAME=Template-Rloclab-F1Tenth-DriftRL-v0 ./train.sh --experiment_name f1tenth_drift
```

The wheeled environments expect WheeledLab robot USD assets (`mushr_nano(.usd)`, `mushr_nano_v2.usd`, `f1tenth.usd`).

- Set `RLOCLAB_WHEELEDLAB_ASSETS_DATA_DIR` to the directory containing `Robots/` from WheeledLab assets.
- If unset, the code defaults to `src/WheeledLab/source/wheeledlab_assets/data` inside this mono-repo layout.

## Running tests
Submit the provided automated checks to Burst:
```bash
cd ~/rloclab
./tests.sh
```

## Extra documentation

See `docs/getting_started.md` for a concise step-by-step.

## Visualize with TensorBoard

You can inspect training metrics (reward curves, loss values, episode lengths) using TensorBoard. This requires installing it on your local machine.

1.  **Install TensorBoard:**
    On your local computer (do NOT run this on Greene), install the package:
    ```
    pip install tensorboard
    ```

2.  **Launch the Server:**
    Navigate to the folder where you downloaded your logs and start the server:
    ```
    # Assuming you are in the directory containing the 'logs' folder
    tensorboard --logdir ./logs
    ```

3.  **View Metrics:**
    Open your browser to the URL shown (usually `http://localhost:6006/`).

## Debugging on Burst

Burst storage is accessible only from a job running on burst, not from the burst login node. The provided scripts do not automatically synchronize error logs back to your home directory on Greene. However, you will need access to these logs to debug failed jobs. These error logs differ from the logs in the previous section.

The suggested way to inspect these logs is via the Open OnDemand web interface:

1.  Navigate to [https://ood-burst-001.hpc.nyu.edu](https://ood-burst-001.hpc.nyu.edu).
2.  Select **Files** > **Home Directory** from the top menu.
3.  You will see a list of files, including your `.err` log files.
4.  Click on any `.err` file to view its content directly in the browser.

> **Important:** Do not modify anything inside the `rloclab` folder on burst storage. This directory is managed by the job scripts, and manual changes may cause synchronization issues or job failures.
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
