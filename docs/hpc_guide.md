(hpc-guide)=
# HPC Guide

This guide covers running the world model training pipeline on NYU HPC (Torch cluster).

## Clusters and Access

- **Torch**: Primary login host and storage.
- **Burst**: Higher-performance storage and compute, accessible via `burst` login node.
- **Open OnDemand**: Web interface for file management and interactive jobs.
  - [Torch OOD](https://ood.hpc.nyu.edu)
  - [Burst OOD](https://ood-burst-001.hpc.nyu.edu)

## Slurm Configuration

SLURM templates and scripts live in `scripts/hpc/slurms/`.

### Singularity / Apptainer Images

| Image | Purpose |
|---|---|
| `/scratch/lq2146/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif` | World model training (PyTorch + CUDA) |
| `/scratch/$USER/isaac-sim_ds3001.sif` | Isaac Sim / Isaac Lab (RL training, simulation) |

### Resource Allocation

| Job Type | Partition | GPUs | Purpose |
|---|---|---|---|
| WM Training (`train_wm`) | `c24m170-a100-2` | 2 × A100 | DDP multi-GPU training |
| RL Training (`wheeled_train`) | `c24m170-a100-2` | 2 × A100 | Isaac Lab RL experiments |
| Install / Setup | `g2-standard-12` | 1 | Environment bootstrapping |
| Interactive / Bash | `g2-standard-12` | 1 | Debugging, data inspection, Isaac Sim livestream |

### Key Parameters

- **Account**: `ds_ga_3001_003-2026sp`
- **Venv Path**: `/scratch/$USER/world_model_cache/venv` (persistent across jobs)
- **Isaac Lab Path**: `/scratch/$USER/IsaacLab`

All containers are read-only; a persistent venv on `/scratch` provides mutable package state.

## First-Time Setup

```bash
git clone --recursive <repo-url> ~/ds_ga_3001
cd ~/ds_ga_3001

sbatch scripts/hpc/slurms/ds_3001_install.slurm
cat >> ~/.envs << 'EOF'

export HF_TOKEN=hf_...
export WANDB_API_KEY=...
EOF
```

## Pipeline Client

The `pipeline-py-client` is the primary interface for all HPC operations.

```bash
python scripts/hpc/pipeline-py-client
```

### Key Menu Options

| Option | Description |
|---|---|
| **slurm** → **Generate from template** | Generate a SLURM script from a template with pre-filled defaults and submit it |
| **download-dataset** | Download a HuggingFace dataset (e.g., `lquan9/collect1`) |
| **install** | Queue the environment setup script |
| **interactive** | Launch an interactive container session |
| **monitor** | Watch `squeue` in real-time |

### Template-Based SLURM Generation

Instead of hand-editing SLURM files, use the pipeline client to generate them:

1. Select **slurm** → **Generate from template (ds_3001)**
2. Choose experiment type (e.g., `train_wm`)
3. Confirm or adjust pre-filled defaults (partition, GPUs, dataset, backbone, etc.)
4. Submit the generated script directly via `sbatch`

Templates live in `scripts/hpc/slurms/ds_3001_*_template.slurm`.

## World Model Training with DDP

All WM training experiments use **DistributedDataParallel (DDP)** across 2 A100 GPUs via `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 \
  scripts/world_model/train_world_model.py --ddp \
  --dataset lquan9/collect1 \
  --backbone cnn \
  --epochs 20 --batch_size 64 \
  --wandb
```

Key DDP behaviors:
- Each GPU gets a unique data shard via `DistributedSampler`
- Only rank 0 logs to WANDB, prints progress, and saves checkpoints
- Checkpoints are saved as unwrapped `state_dict` (loadable without DDP)

## Isaac Lab RL Training

RL experiments (e.g., wheeled drift, Go2 locomotion) use Isaac Sim inside the `isaac-sim_ds3001.sif` container:

```bash
sbatch scripts/hpc/slurms/ds_3001_wheeled_train.slurm \
  --task="Template-Rloclab-Mushr-DriftRL-v0" \
  --agent="rsl_rl_cfg_entry_point" \
  --headless
```

### Isaac Sim Livestream

For interactive visualization, use the `ds_3001_isaac_example.slurm` script and forward port 8211:

```bash
sbatch scripts/hpc/slurms/ds_3001_isaac_example.slurm

ssh -N -L 8211:<compute-node>:8211 $USER@dtn.torch.hpc.nyu.edu
# http://localhost:8211/streaming/webrtc-demo/
```

## Storage and Data

- **Project Root**: `~/ds_ga_3001`
- **WM Persistent Cache**: `/scratch/$USER/world_model_cache/` (venv, pip cache, HF cache)
- **WM Training Logs**: `/scratch/$USER/world_model/logs/<job_id>/`
- **Isaac Lab Logs**: `/scratch/$USER/isaaclab/logs/<job_id>/`
- **Isaac Lab Caches**: `/scratch/$USER/singularity-cache/` (kit, ov, pip)
- **Datasets**: Downloaded via `pipeline-py-client` or `huggingface_hub`

## Debugging

```bash
cat <experiment_name>_<job_id>.out
cat <experiment_name>_<job_id>.err

squeue -u $USER
```
