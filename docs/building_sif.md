(building-sif)=
# Building the Custom SIF Locally

This guide documents how to manually reproduce the custom Isaac Sim Singularity image
(`isaac-sim_ds3001.sif`). 

## Prerequisites

- Singularity installed (≥ 3.8 recommended) with fakeroot configured:
  your username must have entries in `/etc/subuid` and `/etc/subgid`.
- The base Isaac Sim SIF (`isaac-sim_5.1.0.sif`). See
  [Obtaining the Base SIF](#obtaining-the-base-sif) below for how to pull it from NVIDIA NGC.
- Sufficient disk space: expect **> 100 GB** for the base image, sandbox, and final SIF.
- The project repository cloned locally.
- IsaacLab cloned (step 2 below).

(obtaining-the-base-sif)=
## Obtaining the Base SIF

The base image must be pulled from the [NVIDIA NGC](https://ngc.nvidia.com) container
registry, which requires a free NGC account and API key.

1. Generate an API key at **ngc.nvidia.com → Account → Setup → Generate API Key**.

2. Authenticate Singularity against the registry (the username is the literal string
   `$oauthtoken`, not a shell variable):

   ```bash
   singularity remote login --username '$oauthtoken' --password <YOUR_NGC_API_KEY> docker://nvcr.io
   ```

3. Pull the image:

   ```bash
   singularity pull isaac-sim_5.1.0.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0
   ```

   This produces `isaac-sim_5.1.0.sif` directly in the current directory.
   Move it to `$HOME` or whichever path you set for `SIF_IMAGE` in Step 1.

:::{note}
Alternatively, if you already have the image on HPC scratch, transfer it with `rsync`:

```bash
rsync -aP <netid>@login.torch.hpc.nyu.edu:/scratch/<netid>/isaac-sim_5.1.0.sif "$HOME/"
```
:::

## Step 1 — Set Local Variables

Open a terminal and export paths that match your local layout:

```bash
export SIF_IMAGE="$HOME/isaac-sim_5.1.0.sif"
export BUILT_SIF="$HOME/isaac-sim_ds3001.sif"
export SANDBOX_DIR="$HOME/isaac_container"
export ISAACLAB_DIR="$HOME/IsaacLab"
export PROJECT_ROOT="/path/to/ds_ga_3001"   # adjust to your clone location
export SCRATCH_TMP="$HOME/tmp"
mkdir -p "$SCRATCH_TMP"
```

## Step 2 — Clone IsaacLab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
```

## Step 3 — Build the Sandbox

Extract the base SIF into a writable sandbox directory:

```bash
singularity build --sandbox "$SANDBOX_DIR" "$SIF_IMAGE"
```

This may take several minutes depending on image size.

## Step 4 — Install Packages into the Sandbox

Run the project install script inside the writable sandbox.  This replicates the
`singularity exec --writable` block from `install.slurm`:

```bash
singularity exec --writable --no-home \
  -B "$SCRATCH_TMP":/tmp:rw \
  -B "$ISAACLAB_DIR":/workspace/isaaclab:rw \
  -B "$PROJECT_ROOT":/workspace/run:rw \
  "$SANDBOX_DIR" bash -c '
set -euo pipefail
export PYTHON_EXE="/isaac-sim/kit/python/bin/python3"
export PROJECT_ROOT="/workspace/run"
export UV_LINK_MODE=copy

$PYTHON_EXE -m pip install --no-cache-dir uv

# Install the project and Isaac Lab sub-packages
$PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "$PROJECT_ROOT"

for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target="/workspace/isaaclab/source/${pkg}"
  [ -d "$target" ] && $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" \
    --no-cache-dir -e "$target" || true
done

bash "$PROJECT_ROOT/scripts/install_externals.sh"
'
```

:::{note}
If you need to patch `external/dino/requirements.txt` (removing pinned
`xformers==0.0.18` and forcing `torch>=2.0`), do so before running the block above,
or include the `sed` commands from `install.slurm` inside the heredoc.
:::

## Step 5 — Convert Sandbox to SIF (with fakeroot)

Unlike the HPC environment, fakeroot should work on a properly configured local machine:

```bash
export TMPDIR="$SCRATCH_TMP"
export SINGULARITY_TMPDIR="$SCRATCH_TMP"

singularity build --fakeroot "$BUILT_SIF" "$SANDBOX_DIR"
```

If `--fakeroot` fails, verify your subuid/subgid mappings:

```bash
grep "$(whoami)" /etc/subuid /etc/subgid
```

Each file should contain a line like `<username>:100000:65536`.

## Step 6 — Verify the Image

```bash
singularity exec --nv --no-home \
  -B "$ISAACLAB_DIR":/workspace/isaaclab:rw \
  -B "$PROJECT_ROOT":/workspace/run:rw \
  "$BUILT_SIF" /isaac-sim/kit/python/bin/python3 -c \
  "import torch, isaaclab; print('[verify] torch', torch.__version__, '| isaaclab', isaaclab.__file__)"
```

A successful run prints something like:

```
[verify] torch 2.x.x+cu12x | isaaclab /workspace/isaaclab/source/isaaclab/isaaclab/__init__.py
```

## Step 7 — Clean Up the Sandbox

Once the SIF is verified, remove the sandbox to reclaim inodes and disk space:

```bash
rm -rf "$SANDBOX_DIR"
```

## Transferring the SIF to HPC

After building locally you can transfer the SIF to HPC scratch with `rsync`:

```bash
rsync -aP "$BUILT_SIF" <netid>@greene.hpc.nyu.edu:/scratch/<netid>/isaac-sim_ds3001.sif
```

The SLURM run scripts expect the image at `/scratch/$USER/isaac-sim_ds3001.sif`.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `permission denied` on `/isaac-sim/...` inside container | `--containall` flag active | Use `--no-home` instead |
| `fakeroot: no valid mapping entry` | Missing subuid/subgid | Add entries to `/etc/subuid` and `/etc/subgid` and run `singularity config fakeroot --add <user>` |
| Build fails with `/tmp` full | Node-local `/tmp` too small | Point `TMPDIR` and `SINGULARITY_TMPDIR` to a larger scratch directory |
| SIF skipped on re-run | Old SIF still present | Delete `/scratch/$USER/isaac-sim_ds3001.sif` before resubmitting |