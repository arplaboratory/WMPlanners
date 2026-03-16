#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=20GB
#SBATCH --time=01:00:00
#SBATCH --job-name=install
#SBATCH --mail-type=END
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --output=../%x_%j.out
#SBATCH --error=../%x_%j.err

set -euo pipefail

cd /scratch/$USER

if [ -d IsaacLab/.git ]; then
  echo "[NODE] IsaacLab exists, updating safely..."
  cd IsaacLab
  if ! git diff --quiet || ! git diff --cached --quiet; then
    STASH_FILE="../isaaclab_stash_$(date +%Y%m%d_%H%M%S).patch"
    echo "[NODE] Local changes detected, stashing and saving to $STASH_FILE"
    git stash push -u -m "Auto-stash before update on $(date)"
    git stash show -p > "$STASH_FILE" || echo "[WARN] No diff to save in stash."
  fi
  # Fetch and fast-forward only
  git fetch origin
  git merge --ff-only origin/$(git rev-parse --abbrev-ref HEAD)
  cd ..
else
  echo "[NODE] Cloning IsaacLab..."
  git clone https://github.com/isaac-sim/IsaacLab.git
fi

CONTAINER_PATH="/scratch/$USER/isaac_container"
SIF_IMAGE="/scratch/$USER/isaac-sim_5.1.0.sif"
BUILT_SIF="/scratch/$USER/isaac-sim_ds3001.sif"  # final baked image
OVERLAY_FILE="/scratch/$USER/my_env.ext3"
ISAACLAB_DIR="/scratch/$USER/IsaacLab"
RUN_DIR="${SLURM_SUBMIT_DIR}"

PERSISTENT_CACHE_DIR="/scratch/$USER/singularity"
PERSISTENT_LOGS_DIR="/scratch/$USER/isaaclab/logs/${SLURM_JOB_ID}"

ROOT_PATH="/isaac-sim"
USER_HOME="/root"

# Cache and Temp setup
CACHE_ROOT="/scratch/$USER/singularity-cache"
mkdir -p "${CACHE_ROOT}/cache/pip"

SCRATCH_TMP="/scratch/$USER/tmp"
mkdir -p "${SCRATCH_TMP}"

export APPTAINER_TMPDIR="${SCRATCH_TMP}"
export APPTAINER_CACHEDIR="${SCRATCH_TMP}/apptainer-cache"

mkdir -p "${PERSISTENT_LOGS_DIR}"
touch "${PERSISTENT_LOGS_DIR}/.keep"

export ISAAC_ARGS="$*"
echo "[NODE] ISAAC_ARGS=${ISAAC_ARGS}"

if [ ! -f "${OVERLAY_FILE}" ]; then
  echo "[NODE] Overlay not found, creating ${OVERLAY_FILE} ..."
  cd /scratch/$USER
  rsync -aP /share/apps/overlay-fs-ext3/overlay-50G-10M.ext3.gz .
  gunzip overlay-50G-10M.ext3.gz
  mv overlay-50G-10M.ext3 ${OVERLAY_FILE}
  echo "[NODE] Overlay created at ${OVERLAY_FILE}"
else
  echo "[NODE] Overlay already exists at ${OVERLAY_FILE}, skipping."
fi

SANDBOX_DIR="/scratch/$USER/isaac_container"

if [ -f "${BUILT_SIF}" ]; then
  echo "[NODE] Built SIF already exists at ${BUILT_SIF}, skipping build."
else
  if [ ! -d "${SANDBOX_DIR}" ]; then
    echo "[NODE] Building sandbox from ${SIF_IMAGE} -> ${SANDBOX_DIR} ..."
    singularity build --sandbox "${SANDBOX_DIR}" "${SIF_IMAGE}"
    echo "[NODE] Sandbox build complete."
  else
    echo "[NODE] Sandbox already exists at ${SANDBOX_DIR}, reusing for install step."
  fi

  echo "[NODE] Installing packages into sandbox..."

  mkdir -p "${SANDBOX_DIR}/root/.cache/pip"
  mkdir -p "${SANDBOX_DIR}/tmp"
  mkdir -p "${SANDBOX_DIR}/scratch"
  mkdir -p "${SANDBOX_DIR}/share/apps"
  mkdir -p "${SANDBOX_DIR}/workspace/isaaclab"
  mkdir -p "${SANDBOX_DIR}/workspace/run"

  singularity exec --writable --no-home \
    -B ${CACHE_ROOT}/cache/pip:${USER_HOME}/.cache/pip:rw \
    -B ${SCRATCH_TMP}:/tmp:rw \
    -B ${ISAACLAB_DIR}:/workspace/isaaclab:rw \
    -B ${RUN_DIR}:/workspace/run:rw \
    "${SANDBOX_DIR}" bash -s <<'SCRIPT'
set -euo pipefail
export PYTHON_EXE="/isaac-sim/kit/python/bin/python3"
export PROJECT_ROOT="/workspace/run"
export UV_LINK_MODE=copy    # avoid hardlink-across-filesystem warnings

echo "[SANDBOX] Installing uv..."
$PYTHON_EXE -m pip install --no-cache-dir uv

if [ -f "$PROJECT_ROOT/external/dino/requirements.txt" ]; then
  echo "[SANDBOX] Fixing DINO requirements..."
  sed -i "/xformers==0.0.18/d" "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i "s/torch==2.0.0/torch>=2.0/g" "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i "s/torchvision==0.15.0/torchvision>=0.15/g" "$PROJECT_ROOT/external/dino/requirements.txt"
  sed -i "/--extra-index-url/d" "$PROJECT_ROOT/external/dino/requirements.txt"
fi

echo "[SANDBOX] Installing project dependencies..."
try_install() {
  local target="$1"
  if [ -d "$target" ]; then
    if [ -f "$target/pyproject.toml" ] || [ -f "$target/setup.py" ]; then
      echo "[SANDBOX] pip installing $target"
      $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "${target}[dev]" || \
        $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "${target}" || \
        echo "[SANDBOX] Warning: install of $target failed, continuing."
    else
      echo "[SANDBOX] Skipping $target — not a Python project (no pyproject.toml/setup.py)"
    fi
  else
    echo "[SANDBOX] Target $target does not exist, skipping"
  fi
}

try_install "$PROJECT_ROOT"

# isaaclab sub-packages don't define a [dev] extra — install plain editable
for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target="/workspace/isaaclab/source/${pkg}"
  if [ -d "$target" ] && ( [ -f "$target/pyproject.toml" ] || [ -f "$target/setup.py" ] ); then
    echo "[SANDBOX] pip installing $target"
    $PYTHON_EXE -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "$target" || \
      echo "[SANDBOX] Warning: install of $target failed, continuing."
  else
    echo "[SANDBOX] Skipping $target — not found or not a Python project"
  fi
done

echo "[SANDBOX] Preparing environment for scripts/install_externals.sh"
SCRIPTDIR="$($PYTHON_EXE - <<'PY'
import sysconfig
print(sysconfig.get_path('scripts'))
PY
)"
echo "[SANDBOX] Python scripts dir: ${SCRIPTDIR}"
export PATH="${SCRIPTDIR}:$PATH"

if ! command -v uv >/dev/null 2>&1; then
  echo "[SANDBOX] uv not found on PATH; attempting reinstall..."
  $PYTHON_EXE -m pip install --no-cache-dir --upgrade --force-reinstall uv || true
  if ! command -v uv >/dev/null 2>&1; then
    echo "[SANDBOX] uv still unavailable; creating small wrapper at /usr/local/bin/uv"
    cat > /usr/local/bin/uv <<'UVWRAP'
#!/bin/sh
exec "${PYTHON_EXE}" -m uv "$@"
UVWRAP
    chmod +x /usr/local/bin/uv || true
    export PATH="/usr/local/bin:$PATH"
  fi
fi

echo "[SANDBOX] uv path: $(command -v uv 2>/dev/null || echo 'none')"
export UV_PYTHON="$PYTHON_EXE"
bash "$PROJECT_ROOT/scripts/install_externals.sh"

echo "[SANDBOX] Verifying install..."
$PYTHON_EXE - <<'PY'
import sys, importlib, pkgutil
print('python', sys.version.split()[0])
print('sys.path:')
for p in sys.path:
  print('  ', p)
try:
  import torch
  print('torch', torch.__version__)
except Exception as e:
  print('[WARN] torch not available:', e)
# Show top-level modules available
print('Top-level modules:')
for m in sorted({name.split('.',1)[0] for _, name, _ in pkgutil.iter_modules()}):
  print('  ', m)
try:
  import isaaclab
  print('isaaclab ->', getattr(isaaclab, '__file__', 'builtin'))
except Exception as e:
  print('[WARN] isaaclab import failed:', e)
PY
SCRIPT

  echo "[NODE] Converting sandbox -> SIF: ${BUILT_SIF} ..."
  # Singularity build uses TMPDIR / SINGULARITY_TMPDIR for staging; point both at scratch
  # so the extraction of the large Isaac Sim image doesn't exhaust node-local /tmp.
  export TMPDIR="${SCRATCH_TMP}"
  export SINGULARITY_TMPDIR="${SCRATCH_TMP}"
  mkdir -p "${SCRATCH_TMP}"

  SKIP_SANDBOX_REMOVE=0
  if singularity build --fakeroot "${BUILT_SIF}" "${SANDBOX_DIR}" 2>&1; then
    echo "[NODE] SIF build complete (fakeroot): ${BUILT_SIF}"
  else
    echo "[NODE] Warning: --fakeroot build failed; attempting non-fakeroot build (scratch tmp: ${SCRATCH_TMP})..."
    if TMPDIR="${SCRATCH_TMP}" SINGULARITY_TMPDIR="${SCRATCH_TMP}" \
       singularity build "${BUILT_SIF}" "${SANDBOX_DIR}" 2>&1; then
      echo "[NODE] SIF build complete (non-fakeroot): ${BUILT_SIF}"
    else
      echo "[NODE] ERROR: could not build SIF from sandbox. Leaving sandbox in place at: ${SANDBOX_DIR}" >&2
      echo "[NODE] Common causes: fakeroot unsupported (no subuid/subgid mapping), /tmp too small, or insufficient privileges." >&2
      echo "[NODE] /tmp usage at failure:"; df -h /tmp "${SCRATCH_TMP}" || true
      echo "[NODE] To fix: confirm SCRATCH_TMP (${SCRATCH_TMP}) has enough space (>100 GB), enable fakeroot, or contact your sysadmin." >&2
      SKIP_SANDBOX_REMOVE=1
    fi
  fi

  if [ "${SKIP_SANDBOX_REMOVE}" -eq 0 ]; then
    echo "[NODE] Removing sandbox to free inodes..."
    rm -rf "${SANDBOX_DIR}"
    echo "[NODE] Sandbox removed."
  else
    echo "[NODE] Preserving sandbox at ${SANDBOX_DIR} for manual conversion/debugging."
  fi
fi

# Determine which image to use for verification (and for run jobs)
if [ -f "${BUILT_SIF}" ]; then
  APPTAINER_IMAGE="${BUILT_SIF}"
  echo "[NODE] Final SIF ready at: ${BUILT_SIF}"
elif [ -d "${SANDBOX_DIR}" ]; then
  APPTAINER_IMAGE="${SANDBOX_DIR}"
  echo "[NODE] No SIF available; will verify using sandbox at: ${SANDBOX_DIR}"
  echo "[NODE] Run jobs will automatically fall back to the sandbox."
else
  echo "[NODE] ERROR: neither SIF nor sandbox found — install may have failed." >&2
  exit 1
fi

echo "[NODE] Verifying image: ${APPTAINER_IMAGE}"
singularity exec --nv --no-home \
  -B ${SCRATCH_TMP}:/tmp:rw \
  -B ${ISAACLAB_DIR}:/workspace/isaaclab:rw \
  -B ${RUN_DIR}:/workspace/run:rw \
  "${APPTAINER_IMAGE}" /isaac-sim/kit/python/bin/python3 -c \
  "import torch, isaaclab; print('[verify] torch', torch.__version__, '| isaaclab', isaaclab.__file__)"