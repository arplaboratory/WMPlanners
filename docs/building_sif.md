# Building the Custom SIF Locally

This guide documents how to manually reproduce the custom Isaac Sim Apptainer image
(`isaac-sim_ds3001.sif`).

## Installing Apptainer on Ubuntu

<details>
<summary><strong>Click to expand installation instructions</strong></summary>

## Prerequisites

1. [Apptainer](https://github.com/apptainer/apptainer/releases) (≥ 1.0) with fakeroot configured — see [Installing Apptainer on Ubuntu](#installing-apptainer-on-ubuntu) above
2. The base Isaac Sim SIF (`isaac-sim_5.1.0.sif`) — see [Obtaining the Base SIF](#obtaining-the-base-sif)
3. **> 100 GB** free disk space (base image + sandbox + final SIF)
4. This repository cloned locally
5. IsaacLab cloned (Step 2 below)

The recommended distribution for HPC and cluster workflows is **Apptainer** (the
Linux Foundation fork of Singularity). The steps below cover installing Apptainer
and configuring fakeroot so you can build SIF images without root. If you still
prefer SingularityCE, most commands shown here are interchangeable by replacing
`apptainer` with `singularity`.

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

`libfuse-dev` and `libfuse3-dev` provide the FUSE headers required to build
`squashfuse` during the Apptainer source build.  Without them, `./mconfig` exits
with an error and no `Makefile` is generated, causing the subsequent `make` calls
to fail with *"No targets specified"*.

### Install Go (required to build Apptainer from source)

Apptainer is written in Go.  Check the [Apptainer release page](https://github.com/apptainer/apptainer/releases)
for the minimum required Go version, then install it:

```bash
export GOVERSION=1.25.6   # adjust to match required version
wget "https://dl.google.com/go/go${GOVERSION}.linux-amd64.tar.gz"
sudo tar -C /usr/local -xzf "go${GOVERSION}.linux-amd64.tar.gz"
echo 'export PATH="/usr/local/go/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
go version
```

### Build and Install Apptainer

```bash
# Clone and check out the latest stable tag
git clone https://github.com/apptainer/apptainer.git
cd apptainer
git fetch --tags
git checkout "$(git describe --tags --abbrev=0)"

./mconfig --with-suid
make -C builddir
sudo make -C builddir install
```

Verify:

```bash
apptainer --version
```

### Configure Fakeroot

Fakeroot lets you build SIF images as a non-root user by mapping your UID/GID
inside the container to a sub-range of IDs. Apptainer uses the standard
`/etc/subuid` and `/etc/subgid` mappings.

```bash
# Add subuid/subgid mappings for your user (run as root or with sudo)
echo "${USER}:100000:65536" | sudo tee -a /etc/subuid
echo "${USER}:100000:65536" | sudo tee -a /etc/subgid

# Confirm entries were written
grep "$USER" /etc/subuid /etc/subgid
```

Each file should contain a line like:

```
<username>:100000:65536
```

</details>

## Obtaining the Base SIF

### Pull the image

Apptainer stages the squashfs build in `$TMPDIR`, which defaults to `/tmp` — a
**tmpfs** (RAM-backed) filesystem with a size limit typically equal to half your RAM.
For the ~20 GB Isaac Sim image this will exhaust `/tmp` regardless of how much disk
space you have.  Always redirect temp directories to real disk before pulling:

```bash
export TMPDIR="$HOME/tmp"
export APPTAINER_TMPDIR="$HOME/tmp"
export APPTAINER_CACHEDIR="$HOME/apptainer-cache"
unset SINGULARITY_CACHEDIR
mkdir -p "$TMPDIR" "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"
apptainer pull isaac-sim_5.1.0.sif docker://nvcr.io/nvidia/isaac-sim:5.1.0
```

This produces `isaac-sim_5.1.0.sif` in the current directory.
Move it to `$HOME` or whichever path you set for `SIF_IMAGE` in Step 1.

## Step 1 — Set Local Variables

Open a terminal and export paths that match your local layout:

```bash
export SIF_IMAGE="$HOME/isaac-sim_5.1.0.sif"
export BUILT_SIF="$HOME/isaac-sim_ds3001.sif"
export SANDBOX_DIR="$HOME/isaac_container"
export ISAACLAB_DIR="$HOME/IsaacLab"
export SCRATCH_TMP="$HOME/tmp"
export TMPDIR="$SCRATCH_TMP"
export APPTAINER_TMPDIR="$SCRATCH_TMP"
export APPTAINER_CACHEDIR="$HOME/apptainer-cache"
unset SINGULARITY_CACHEDIR
mkdir -p "$SCRATCH_TMP" "$APPTAINER_CACHEDIR"
export PROJECT_ROOT="/path/to/ds_ga_3001"   # adjust to local clone
```

## Step 2 — Clone IsaacLab

```bash
git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR"
```

## Step 3 — Build the Sandbox

Extract the base SIF into a writable sandbox directory:

```bash
apptainer build --sandbox "$SANDBOX_DIR" "$SIF_IMAGE"
```

This may take several minutes depending on image size.

## Step 4 — Install Packages into the Sandbox

With `--writable`, Apptainer cannot auto-create bind mount destinations. Create
them inside the sandbox first:

```bash
mkdir -p "$SANDBOX_DIR/workspace/isaaclab" \
         "$SANDBOX_DIR/workspace/run" \
         "$SANDBOX_DIR/opt/venvs"
```

Patching external submodules

```bash
sed -i '/xformers==0.0.18/d' "$PROJECT_ROOT/external/dino/requirements.txt"
sed -i 's/torch==2.0.0/torch>=2.0/g' "$PROJECT_ROOT/external/dino/requirements.txt"
sed -i 's/torchvision==0.15.0/torchvision>=0.15/g' "$PROJECT_ROOT/external/dino/requirements.txt"
sed -i '/--extra-index-url/d' "$PROJECT_ROOT/external/dino/requirements.txt"
sed -i '/cuml-cu11/d' "$PROJECT_ROOT/external/dino/requirements.txt"
```

Install core packages into sandbox (kit Python, system-level)

```bash
apptainer exec --writable \
  -B "$SCRATCH_TMP":/tmp:rw \
  -B "$ISAACLAB_DIR":/workspace/isaaclab:rw \
  -B "$PROJECT_ROOT":/workspace/run:rw \
  "$SANDBOX_DIR" bash -c '
set -euo pipefail
PYKIT="/isaac-sim/kit/python/bin/python3"
export TMPDIR=/tmp UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy
export PATH="/isaac-sim/kit/python/bin:$PATH"

$PYKIT -m pip install --no-cache-dir uv

for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target="/workspace/isaaclab/source/${pkg}"
  [ -d "$target" ] && $PYKIT -m uv pip install --python "$PYKIT" --no-cache-dir -e "$target" || true
done
'
```

Create venvs and install project + externals

```bash
chmod +x "$PROJECT_ROOT/scripts/create_venvs_in_sandbox.sh"

"$PROJECT_ROOT/scripts/create_venvs_in_sandbox.sh" \
  --sandbox "$SANDBOX_DIR" \
  --project "$PROJECT_ROOT" \
  --fakeroot \
  --install-runtime \
  --install-train
  ```

Deprecated: 

```bash
apptainer exec --writable --no-home \
  -B "$SCRATCH_TMP":/tmp:rw \
  -B "$ISAACLAB_DIR":/workspace/isaaclab:rw \
  -B "$PROJECT_ROOT":/workspace/run:rw \
  "$SANDBOX_DIR" bash -c '
set -euo pipefail
PYTHON_EXE="/isaac-sim/kit/python/bin/python3"

export TMPDIR=/tmp
export UV_CACHE_DIR=/tmp/uv-cache
export UV_LINK_MODE=copy
export PATH="/isaac-sim/kit/python/bin:$PATH"

$PYTHON_EXE -m pip install --no-cache-dir uv

$PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "/workspace/run[dev]" || \
  $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "/workspace/run" || \
  # Fallback: if uv-based install fails, try plain pip editable install
  $PYTHON_EXE -m pip install --no-cache-dir -e "/workspace/run" || \
  echo "Warning: failed to install /workspace/run; continuing"

for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target="/workspace/isaaclab/source/${pkg}"
  [ -d "$target" ] && $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "$target" || true
done

for d in /workspace/run/external/*; do
  if [ -d "$d" ]; then
    if [ -f "$d/pyproject.toml" ] || [ -f "$d/setup.py" ] || [ -f "$d/setup.cfg" ]; then
      echo "Installing external: $d"
      (cd "$d" && $PYTHON_EXE -m pip install --no-cache-dir -e .) || echo "Install failed for $d (continuing)"
    else
      echo "Skipping (not a Python package): $d"
    fi
  fi
done
'
```

## Running Tests Inside the Sandbox (bind caches)

When running `pytest` or other workflows that write large caches or temporary files,
the sandbox overlay can be very small. Bind a host scratch directory into the
container for `/tmp` and redirect caches so tests don't run out of space.

Example (run from your host shell):

```bash
apptainer exec --nv --containall \
  -B "$SCRATCH_TMP":/tmp:rw \
  -B "$CACHE_ROOT/kit-data":/isaac-sim/kit/data:rw \
  -B "$CACHE_ROOT/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$CACHE_ROOT/cache/pip":/root/.cache/pip:rw \
  -B "$ISAACLAB_DIR":/workspace/isaaclab:ro \
  -B "$PROJECT_ROOT":/workspace/run:rw \
  "$SANDBOX_DIR" bash -c '
export TMPDIR=/tmp
export XDG_CACHE_HOME=/tmp/.cache
export TORCH_HOME=/tmp/.cache/torch
export HF_HOME=/tmp/.cache/huggingface
mkdir -p /tmp/.cache /tmp/.cache/torch
cd /workspace/run
/isaac-sim/kit/python/bin/python3 -m pytest -q -vv
'
```

This pattern mirrors what the SLURM job scripts do on HPC: bind a node-local
scratch and point `TMPDIR`, `XDG_CACHE_HOME`, and `TORCH_HOME` there so packages
and models can be downloaded into fast, large local storage instead of the
container overlay.

```

## Step 5 — Convert Sandbox to SIF

```bash
export TMPDIR="$SCRATCH_TMP"
export APPTAINER_TMPDIR="$SCRATCH_TMP"

apptainer build "$BUILT_SIF" "$SANDBOX_DIR"
```

## Step 6 — Verify the Image

Set up node-local cache directories (mirrors what the SLURM scripts do on the cluster):

```bash
USER_HOME="/root"
CACHE_ROOT="$SCRATCH_TMP/docker-isaac-sim"
mkdir -p \
  "$CACHE_ROOT/kit-data" \
  "$CACHE_ROOT/cache/kit" \
  "$CACHE_ROOT/cache/ov" \
  "$CACHE_ROOT/cache/pip" \
  "$CACHE_ROOT/cache/glcache" \
  "$CACHE_ROOT/cache/computecache" \
  "$CACHE_ROOT/logs" \
  "$CACHE_ROOT/data"
```

Import Test

```bash
apptainer exec --nv --containall \
  -B "$CACHE_ROOT/kit-data":/isaac-sim/kit/data:rw \
  -B "$CACHE_ROOT/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$CACHE_ROOT/cache/ov":${USER_HOME}/.cache/ov:rw \
  -B "$CACHE_ROOT/cache/pip":${USER_HOME}/.cache/pip:rw \
  -B "$CACHE_ROOT/cache/glcache":${USER_HOME}/.cache/nvidia/GLCache:rw \
  -B "$CACHE_ROOT/cache/computecache":${USER_HOME}/.nv/ComputeCache:rw \
  -B "$CACHE_ROOT/logs":${USER_HOME}/.nvidia-omniverse/logs:rw \
  -B "$CACHE_ROOT/data":${USER_HOME}/.local/share/ov/data:rw \
  -B "$ISAACLAB_DIR":/workspace/isaaclab:rw \
  -B "$PROJECT_ROOT":/workspace/run:rw \
  "$BUILT_SIF" /isaac-sim/kit/python/bin/python3 -c \
  "import torch, isaaclab; print('[verify] torch', torch.__version__, '| isaaclab', isaaclab.__file__)"
```
`--containall` blocks all automatic mounts (home dir, CWD, default bind paths), making this test representative of the HPC SLURM environment. Every path the container needs must be explicitly bound.

A successful run prints something like:

Running w/ WebRTC

```bash
apptainer exec --nv --containall \
  -B "$CACHE_ROOT/kit-data":/isaac-sim/kit/data:rw \
  -B "$CACHE_ROOT/cache/kit":/isaac-sim/kit/cache:rw \
  -B "$CACHE_ROOT/cache/ov":${USER_HOME}/.cache/ov:rw \
  -B "$CACHE_ROOT/cache/pip":${USER_HOME}/.cache/pip:rw \
  -B "$CACHE_ROOT/cache/glcache":${USER_HOME}/.cache/nvidia/GLCache:rw \
  -B "$CACHE_ROOT/cache/computecache":${USER_HOME}/.nv/ComputeCache:rw \
  -B "$CACHE_ROOT/logs":${USER_HOME}/.nvidia-omniverse/logs:rw \
  -B "$CACHE_ROOT/data":${USER_HOME}/.local/share/ov/data:rw \
  "$BUILT_SIF" /isaac-sim/isaac-sim.streaming.sh --allow-root
```

Download the WebRTC client from [here](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/download.html). 

```
[verify] torch 2.x.x+cu12x | isaaclab /workspace/isaaclab/source/isaaclab/isaaclab/__init__.py
```

## Step 7 — Clean Up the Sandbox

Once the SIF is verified, remove the sandbox to reclaim inodes and disk space:

```bash
rm -rf "$SANDBOX_DIR"
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ERROR: Failed to create container process: Operation not permitted` | Apptainer built without `--with-suid`; user-namespace extraction blocked by kernel | Rebuild Apptainer with `./mconfig --with-suid && make -C builddir && sudo make -C builddir install` |
| `ERROR: Failed to create mount namespace: mount namespace requires privileges` | `--fakeroot` flag used on a suid-install; the two privilege modes conflict | Remove `--fakeroot` — with `starter-suid` installed, run `apptainer build "$BUILT_SIF" "$SANDBOX_DIR"` with no extra flags |
| `WARNING: SINGULARITY_CACHEDIR and APPTAINER_CACHEDIR have different values` | Old `SINGULARITY_CACHEDIR` env var still set from a previous shell session | Add `unset SINGULARITY_CACHEDIR` before `apptainer` commands, or remove it from `~/.bashrc` / `~/.zshrc` |
| `destination /workspace/... doesn't exist in container` | `--writable` can't auto-create bind mount targets | Run `mkdir -p "$SANDBOX_DIR/workspace/isaaclab" "$SANDBOX_DIR/workspace/run"` before `apptainer exec` |
| `permission denied` on `/isaac-sim/...` inside container | `--containall` not set, default bind points interfering | Use `--containall` and explicitly bind every path the container needs via `-B` |
| `fakeroot: no valid mapping entry` | Missing subuid/subgid | Add entries to `/etc/subuid` and `/etc/subgid` (see Configure Fakeroot section above) |
| `No space left on device` during `pull` or `build` | `/tmp` is a size-limited tmpfs (RAM-backed), not your disk | Set `TMPDIR`, `APPTAINER_TMPDIR`, and `APPTAINER_CACHEDIR` to a path on real disk (e.g. `$HOME/tmp`) |
| Build fails with `/tmp` full | Node-local `/tmp` too small | Point `TMPDIR` and `APPTAINER_TMPDIR` to a larger scratch directory |
| SIF skipped on re-run | Old SIF still present | Delete `/scratch/$USER/isaac-sim_ds3001.sif` before resubmitting |