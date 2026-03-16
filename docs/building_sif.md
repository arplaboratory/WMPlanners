# Building the Custom SIF Locally

This guide documents how to manually reproduce the custom Isaac Sim Singularity image
(`isaac-sim_ds3001.sif`). 

## Installing Singularity on Ubuntu

<details>
<summary><strong>Click for full Singularity installation instructions</strong></summary>

---

The recommended distribution is **SingularityCE** (Community Edition), maintained by
Sylabs.  The steps below install it from the official Sylabs APT repository and
configure fakeroot so you can build SIF images without root.

### Install Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libseccomp-dev \
  pkg-config \
  uidmap \
  squashfs-tools \
  fuse2fs \
  fuse-overlayfs \
  fakeroot \
  cryptsetup \
  curl wget git \
  libfuse-dev \
  libfuse3-dev \
  libglib2.0-dev \
  zlib1g-dev \
  libsubid-dev
```

:::{note}
`libfuse-dev` and `libfuse3-dev` provide the FUSE headers required to build
`squashfuse` during the SingularityCE build.  Without them, `./mconfig` exits
with an error and no `Makefile` is generated, causing the subsequent `make` calls
to fail with *"No targets specified"*.
:::

### Install Go (required to build SingularityCE from source)

SingularityCE is written in Go.  Check the [SingularityCE release page](https://github.com/sylabs/singularity/releases)
for the minimum required Go version, then install it:

```bash
export GOVERSION=1.25.6   # adjust to match required version
wget "https://dl.google.com/go/go${GOVERSION}.linux-amd64.tar.gz"
sudo tar -C /usr/local -xzf "go${GOVERSION}.linux-amd64.tar.gz"
echo 'export PATH="/usr/local/go/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
go version
```

### Build and Install SingularityCE

```bash
export SINGULARITY_VERSION=4.4.0   # check releases for latest stable
wget "https://github.com/sylabs/singularity/releases/download/v${SINGULARITY_VERSION}/singularity-ce-${SINGULARITY_VERSION}.tar.gz"
tar -xzf "singularity-ce-${SINGULARITY_VERSION}.tar.gz"
cd "singularity-ce-${SINGULARITY_VERSION}"

./mconfig
# mconfig must exit 0 and print "Makefile written to builddir/Makefile" before continuing.
# If it reports "libsubid headers are required", either:
#   (a) install the missing package:  sudo apt-get install -y libsubid-dev  (recommended)
#   (b) disable the feature:          ./mconfig --without-libsubid
# Without libsubid, Singularity still works but won't honor NSS subid configuration.
make -C builddir
sudo make -C builddir install
```

Verify:

```bash
singularity --version
```

### Configure Fakeroot

Fakeroot lets you build SIF images as a non-root user by mapping your UID/GID inside
the container to a sub-range of IDs.

```bash
# Add subuid/subgid mappings for your user (run as root or with sudo)
sudo singularity config fakeroot --add "$USER"

# Confirm entries were written
grep "$USER" /etc/subuid /etc/subgid
```

Each file should contain a line like:

```
<username>:100000:65536
```

If `singularity config fakeroot` is unavailable on your version, add the entries
manually:

```bash
echo "${USER}:100000:65536" | sudo tee -a /etc/subuid
echo "${USER}:100000:65536" | sudo tee -a /etc/subgid
```

:::{note}
If you prefer **Apptainer** (the Linux Foundation fork), the commands are identical —
just replace `singularity` with `apptainer` throughout this guide.  Install via:
`sudo apt-get install -y apptainer` on Ubuntu 22.04+ or from the
[Apptainer releases page](https://github.com/apptainer/apptainer/releases).
:::

</details>

## Prerequisites

1. [Singularity CE](https://github.com/sylabs/singularity/releases) (≥ 3.8) with fakeroot configured
2. The base Isaac Sim SIF (`isaac-sim_5.1.0.sif`) — see [Obtaining the Base SIF](#obtaining-the-base-sif)
3. **> 100 GB** free disk space (base image + sandbox + final SIF)
4. This repository cloned locally
5. IsaacLab cloned (Step 2 below)

<details>
<summary><strong>Click for instructions on installing prerequisites</strong></summary>

---

See [Installing Singularity on Ubuntu](#installing-singularity) above for the full
Singularity installation and fakeroot setup walkthrough.

</details>

(obtaining-the-base-sif)=
## Obtaining the Base SIF

### Pull the image:

   ```bash
   singularity pull isaac-sim_5.1.0.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0
   ```

   This produces `isaac-sim_5.1.0.sif` directly in the current directory.
   Move it to `$HOME` or whichever path you set for `SIF_IMAGE` in Step 1.

## Step 1 — Set Local Variables

Open a terminal and export paths that match your local layout:

```bash
export SIF_IMAGE="$HOME/isaac-sim_5.1.0.sif"
export BUILT_SIF="$HOME/isaac-sim_ds3001.sif"
export SANDBOX_DIR="$HOME/isaac_container"
export ISAACLAB_DIR="$HOME/IsaacLab"
export PROJECT_ROOT="/path/to/ds_ga_3001"   # adjust to local clone
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