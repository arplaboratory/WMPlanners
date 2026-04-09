# HPC pipeline client

`pipeline-py-client` is a terminal-friendly curses UI / CLI for running
our worlmodel workflows from a remote HPC login node.

## Installation

The client is a single executable script located in this folder. Make it
executable (if it is not already) and call it directly. The default behavior is
to launch the dialog-based UI.

```bash
chmod +x pipeline-py-client
```

Install the command on HPC as an executable in your `PATH`. Note: If using burst or on an interactive compute node, you may not have access to the subsequent bash files. Therefore, you can create your own environments file to source (e.g., .envs).

```bash
mkdir -p ~/.local/bin
install -m 755 pipeline-py-client ~/.local/bin/hpc

# TODO: or zshrc
echo "export PATH=\"$HOME/.local/bin:$PATH\"" >> ~/.bashrc
echo "export PROJECT_ROOT=\"<path to project>\"" >> ~/.bashrc
source ~/.bashrc # or source <custom envs> (e.g., .envs)
```

Note `dialog` might not be available on burst! See (Quick Start)[../../README.md#quick-start] for workaround. 

After that, run:

```bash
hpc --help
```

> The UI requires the `dialog` binary (often preinstalled on HPC login nodes).
> If missing, install via your package manager (e.g., `apt-get install dialog`).
> All commands support `--dry-run` to print actions without executing them

## Interactive UI

Running `hpc` opens a menu with

- **Install environment on HPC**: pick from discovered `*install*.slurm` scripts, then queue `scripts/install.sh` to submit the selected installer on the login host.
- **Download datasets**: choose datasets via checkboxes, enter the dataset root
  and splits, and build the `python -m jepa.datasets.download` commands.
- **Launch pretraining**: choose a config, datasets, and extra arguments, then
  run `python -m jepa.train.pretrain`. The flow can also generate and submit
  a Slurm script using the template under `slurms/`.
- **Generate Slurm**: wrap any arbitrary command in the Slurm template and
  optionally submit with `sbatch`.
- **Help**: open the README.md file inside a dialog textbox.

## Commands

### Install environment on HPC

```bash
hpc install
```

Queues `scripts/install.sh`, which submits `scripts/hpc/install.slurm` on the selected login host.

You can override host/project path when needed:

```bash
hpc install --host burst --remote-project-dir ~/my-repo
```

### Download datasets (WIP)

```bash
hpc download \
  --dataset-root /path/to/datasets \
  --datasets cifar100 imagenet \
  --splits train test
```

Executes `python -m jepa.datasets.download` once per dataset with the
provided splits. The dataset root must exist before running.

### Launch pretraining and optionally generate Slurm

```bash (WIP)
hpc pretrain \
  --config configs/hf224_config.json \
  --datasets common_pool unsplash \
  --slurm-path my_pretrain.slurm \
  --job-name jepa-pretrain \
  --gpus gpu:rtx8000:4 \
  --ext3-path /scratch/overlay.ext3 \
  --sif-path /scratch/env.sif \
  --submit \
  -- --batch-size 256 --epochs 100
```

This builds `python -m jepa.train.pretrain` with the provided datasets and
extra arguments. When `--slurm-path` is provided a job file is generated from
`slurms/sbatch_pretrain.slurm`; `--submit` will call `sbatch` for you.

### Launch Isaac Lab / rloclab RL training

```bash
hpc rloclab-train \
  --task Template-Rloclab-Go2-Direct-v0 \
  --max-iterations 10000 \
  --num-envs 4096 \
  --headless \
  --slurm-path go2_direct_train.slurm \
  --submit
```

For wheeled tasks, set task names such as `Isaac-RLlab-F1Tenth-Drift-v0` or
`Isaac-RLlab-MuSHR-Drift-v0`. For quadruped tasks, use Go2 templates such as
`Template-Rloclab-Go2-Direct-v0` / `Template-Rloclab-Go2-Bipedal-v0` /
`Template-Rloclab-Go2-Backflip-v0`.

This queues `scripts/train.sh`, which submits `scripts/hpc/train.slurm` on the selected host
with the selected task and forwarded training arguments.

You can override host/project path when needed:

```bash
hpc rloclab-train --host burst --remote-project-dir ~/my-repo --task Template-Rloclab-Go2-Direct-v0
```

### Generate a Slurm script for any command

```bash
hpc slurm \
  --output download.slurm \
  --job-name jepa-downloads \
  --ext3-path /scratch/overlay.ext3 \
  --sif-path /scratch/env.sif \
  -- -- python -m jepa.datasets.download --dataset-root /path --dataset-name cifar100
```

Accepts the command to run after `--` and renders the submission script using
the same defaults as the pretraining helper.

### Inspect the bundled spec

```bash
hpc show-spec
```

Prints the `spec` document shipped with this directory for quick reference in a
remote shell.

## Notes

- Commands are executed exactly as printed; use `--dry-run` to rehearse them
  before launching on the cluster.
- The Slurm generator replaces the placeholders in `slurms/sbatch_pretrain.slurm`
  (job name, resources, overlay paths, and command body).
- Dataset curation helper scripts are in `scripts/hpc/helpers/` and can be used
  standalone in addition to the pipeline client commands.
- If `sbatch` is not on your PATH, submission will fail with a clear error but
  the script will still be generated.

For a deeper dive into HPC and configuration knobs, see
[HPC Guide](hpc-guide).
