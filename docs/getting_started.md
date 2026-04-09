# Getting Started

## Install and verify

### Local Development

We recommend using the centralized setup script to ensure environment parity:

```bash
bash scripts/python_setup.sh
source .venv/bin/activate
```

This will handle `uv` installation, submodule initialization, and dependency resolution automatically.

### HPC Environment

On the cluster, the environment is bootstrapped via SLURM:

```bash
# One-time setup: create persistent venv on /scratch
sbatch scripts/hpc/slurms/ds_3001_install.slurm

# Set up credentials
cat >> ~/.envs << 'EOF'
export HF_TOKEN=hf_...
export WANDB_API_KEY=...
EOF
```

All subsequent jobs automatically activate the venv and refresh the project code.

## Pipeline Client

The `pipeline-py-client` is the main entry point for HPC operations:

```bash
python scripts/hpc/pipeline-py-client
```

Key capabilities:
- **Generate & submit SLURM jobs** from templates (pre-filled with project defaults)
- **Download HuggingFace datasets** (e.g., `lquan9/collect1`)
- **Monitor running jobs** in real-time

## Next Steps

- For training details, see [World Model Training](world_model_training.md)
- For HPC specifics, see [HPC Guide](hpc_guide.md)
- For building a SIF, see [Building SIF](building_sif.md)