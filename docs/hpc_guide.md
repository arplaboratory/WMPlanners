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

### Available Partitions

| Partition | GPU | VRAM | SBATCH `--gres` | Notes |
|---|---|---|---|---|
| `interactive` / `n2c48m24` | None (CPU only) | — | — | Jupyter, data prep |
| `n1s8-t4-1` | 1x T4 | 16 GB | `gpu:t4:1` | Light inference, testing |
| `g2-standard-12` | 1x L4 | 24 GB | `gpu:l4:1` | Install, debugging, small training |
| `g2-standard-24` | 2x L4 | 2x 24 GB | `gpu:l4:2` | Medium training |
| `g2-standard-48` | 4x L4 | 4x 24 GB | `gpu:l4:4` | Multi-GPU training |
| `c12m85-a100-1` | 1x A100 | 40 GB | `gpu:a100:1` | Single-GPU training (recommended) |
| `c24m170-a100-2` | 2x A100 | 2x 40 GB | `gpu:a100:2` | DDP multi-GPU training |

### Resource Allocation

| Job Type | Partition | GPUs | Purpose |
|---|---|---|---|
| WM Training (`train_wm`) | `c12m85-a100-1` or `c24m170-a100-2` | 1–2 × A100 | Single or DDP multi-GPU training |
| RL Training (`wheeled_train`) | `c24m170-a100-2` | 2 × A100 | Isaac Lab RL experiments |
| Install / Setup | `g2-standard-12` | 1x L4 | Environment bootstrapping |
| Interactive / Bash | `g2-standard-12` | 1x L4 | Debugging, data inspection, Isaac Sim livestream |

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

(resuming-training)=
## Resuming Training

The training script (`train_world_model.py`) supports automatic checkpoint resumption. Every epoch, the trainer saves:
- `epoch_NNN.pt` — per-epoch checkpoint
- `latest.pt` — always points to the most recent epoch

Each checkpoint contains the model weights, optimizer state, epoch number, and model config.

### Auto-Resume (recommended)

If `latest.pt` exists in `--out_dir`, training resumes automatically — no extra flags needed. Just resubmit the same SLURM script:

```bash
# Original run saved checkpoints to /scratch/$USER/wm_experiments/cnn_collect1/
# Job got preempted at epoch 12. Resubmit the same script:
sbatch experiments/slurm_cnn_collect1.sh
# Training picks up from epoch 13 automatically.
```

### Explicit Resume

To resume from a specific checkpoint (e.g., a different run's weights):

```bash
python scripts/world_model/train_world_model.py \
  --dataset lquan9/collect1 \
  --backbone cnn \
  --epochs 40 \
  --out_dir /scratch/$USER/wm_experiments/cnn_collect1_v2 \
  --resume /scratch/$USER/wm_experiments/cnn_collect1/epoch_015.pt
```

### Extending a Completed Run

To train for more epochs beyond the original `--epochs`, increase the value and resubmit. Auto-resume loads the last checkpoint and continues:

```bash
# Original run: --epochs 20 (completed)
# Edit the SLURM script to --epochs 40, then:
sbatch experiments/slurm_cnn_collect1.sh
# Resumes from epoch 20, trains to epoch 40.
```

### Using the Singularity Overlay Setup

If you set up training using the Singularity overlay approach (see `experiments/`), the workflow is the same. The SLURM scripts use:
- **Overlay**: `/scratch/$USER/wm_env.ext3` (conda env with all deps)
- **SIF**: `/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif`
- **Output**: `/scratch/$USER/wm_experiments/<experiment_name>/`

To resume, just resubmit — the `--out_dir` already contains `latest.pt`:

```bash
cd ~/NYU_DS_GA_3001/experiments
sbatch slurm_cnn_collect1.sh          # auto-resumes from latest.pt
```

### What Gets Restored

| Component | Restored? | Notes |
|---|---|---|
| Model weights | Yes | Loaded into raw model (works with or without DDP) |
| Optimizer state | Yes | Learning rate schedule, momentum buffers preserved |
| Epoch counter | Yes | Training continues from `epoch + 1` |
| WANDB run | No | Creates a new WANDB run (use same `--experiment_name` to group) |
| Dataset | No | Re-downloaded/loaded from scratch (cached by HuggingFace) |

### Troubleshooting Resume

- **"No checkpoint found"**: Verify `latest.pt` exists in your `--out_dir`:
  ```bash
  ls /scratch/$USER/wm_experiments/<experiment>/latest.pt
  ```
- **Preempted job**: SLURM jobs on `--requeue` partitions may be killed and restarted. Auto-resume handles this seamlessly as long as the last epoch checkpoint was fully written.
- **Mismatched architecture**: If you change `--backbone` or `--dynamics_type` between runs, you cannot resume from an old checkpoint. Start fresh with a new `--out_dir`.

## Quick Start: Resuming After a Fresh Login

If you've already completed first-time setup and just need to get back to running experiments after logging into HPC, follow these steps:

### 1. SSH into the cluster

```bash
ssh dm6262@greene.hpc.nyu.edu
# or
ssh dm6262@burst.hpc.nyu.edu
```

### 2. Navigate to the project

```bash
cd ~/NYU_DS_GA_3001
git pull   # get latest code if needed
```

### 3. Check existing jobs and results

```bash
# See if any jobs are running
squeue -u $USER

# Check results from previous runs
cat /scratch/$USER/wm_experiments/*/eval_metrics.json

# Check logs from a specific job
ls experiments/wm_*.out
cat experiments/wm_cnn_collect1_<job_id>.out
```

### 4. Submit new experiments

Everything is already installed in the overlay — no setup needed. Just submit:

```bash
cd ~/NYU_DS_GA_3001/experiments

# Submit individual experiments
sbatch slurm_cnn_collect1.sh
sbatch slurm_dinov2_collect1.sh

# Or submit all 4 comparison experiments at once
bash run_all.sh
```

If a previous run was interrupted, resubmitting the same script auto-resumes from the last checkpoint (see [Resuming Training](#resuming-training)).

### 5. (Optional) Interactive debugging session

To get a shell inside the container with GPU access:

```bash
srun --account=ds_ga_3001_003-2026sp --partition=c12m85-a100-1 \
     --gres=gpu:a100:1 --cpus-per-task=4 --mem=16GB --time=01:00:00 --pty bash

# Once on the compute node:
singularity exec --nv \
  --overlay /scratch/$USER/wm_env.ext3:ro \
  /share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif /bin/bash

# Inside the container:
source /ext3/env.sh
export PYTHONPATH=~/NYU_DS_GA_3001/src:$PYTHONPATH
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 6. (If needed) Reinstall or update the overlay environment

Only needed if dependencies changed (e.g., new packages in `pyproject.toml`):

```bash
# Get an interactive GPU node first
srun --account=ds_ga_3001_003-2026sp --partition=g2-standard-12 \
     --gres=gpu:l4:1 --cpus-per-task=4 --mem=16GB --time=01:00:00 --pty bash

# Run the setup script (idempotent — skips what's already installed)
bash ~/NYU_DS_GA_3001/experiments/setup_env.sh
```

### Checklist

| What | Where | Persistent? |
|---|---|---|
| Code | `~/NYU_DS_GA_3001` | Yes (home dir) |
| Conda env + deps | `/scratch/$USER/wm_env.ext3` | Yes (overlay on scratch) |
| SIF image | `/share/apps/images/cuda12.6.2-cudnn9.5.0-devel-ubuntu24.04.1.sif` | Yes (shared, read-only) |
| Checkpoints | `/scratch/$USER/wm_experiments/<experiment>/` | Yes (scratch) |
| HF dataset cache | Node-local `/tmp/hf_cache` | No (re-downloaded each job) |
| WANDB / HF tokens | `~/.envs` | Yes (home dir) |

Nothing needs to be reinstalled between logins — `scratch` and `home` persist across sessions.

## Debugging

```bash
cat <experiment_name>_<job_id>.out
cat <experiment_name>_<job_id>.err

squeue -u $USER
```
