# Reinforcement Learning in Isaac Lab

## Model-free approach

### Links
1.  **Project Webpage:** [https://machines-in-motion.github.io/RL_class_go2_project/](https://machines-in-motion.github.io/RL_class_go2_project/)
2.  **Project Tutorial:** [https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md](https://github.com/machines-in-motion/rob6323_go2_project/blob/master/tutorial/tutorial.md)

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