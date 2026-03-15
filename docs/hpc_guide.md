# HPC Guide

This guide provides technical details for running the `cjepa` project on NYU HPC (Torch and Burst clusters).

## Clusters and Access

- **Torch**: Primary login host and storage.
- **Burst**: Higher-performance storage and compute, accessible via `burst` login node or jobs.
- **Open OnDemand**: Web interface for file management and interactive jobs.
  - [Torch OOD](https://ood.hpc.nyu.edu)
  - [Burst OOD](https://ood-burst-001.hpc.nyu.edu)

## Slurm Configuration

The default Slurm configurations are located in `scripts/hpc/slurms/`.

### Key Parameters

- **Account**: `ds_ga_3001_003-2026sp` (or `rob_gy6323-2025fa` for some legacy tasks).
- **Partition**: `n2c48m24` (general compute) or `gpu` for training.
- **Venv Path**: `/scratch/$USER/venvs/ds_ga_3001` or `.venv` in the project root.

### Singularity Images

We use pre-built Singularity/Apptainer images for Isaac Sim and Isaac Lab:
- `/scratch/$USER/isaac-sim_5.1.0.sif`
- `/scratch/$USER/isaac-lab-base.sif`

## HPC Pipeline Client

The `pipeline-py-client` (aliased as `hpc`) is the recommended way to interact with the cluster.

### Setup

```bash
cd scripts/hpc
./pipeline-py-client install
```

### Common Commands

- `hpc ui`: Launch the interactive menu.
- `hpc install`: Setup the remote environment.
- `hpc pretrain`: Launch a pretraining job.
- `hpc rloclab-train`: Launch a Reinforcement Learning job.

## Storage and Data

- **Project Root**: Mirror your local repository to `${HOME}/ds_ga_3001`.
- **Experiments**: Checkpoint and logs are typically stored in `/scratch/$USER/isaaclab/logs/` or as specified in your config.
- **Datasets**: Download datasets to `/scratch/$USER/data` for fast I/O during training.

## Debugging

Always check your Slurm output and error files:
- `slurm_%j.out`
- `slurm_%j.err`

Use the `tail` command or Open OnDemand to inspect logs in real-time.
