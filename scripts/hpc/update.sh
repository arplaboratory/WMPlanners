#!/usr/bin/env bash

# usage
PKG="$1"
if [ -z "$PKG" ]; then
    echo "Usage: $0 <python-package>"
    exit 1
fi

# name for the temporary SLURM script
TMP_SLURM="install_${PKG}_tmp.slurm"

# Copy the original install.slurm and append pip install commands
cp base.slurm "$TMP_SLURM"

cat <<EOF >> "$TMP_SLURM"

# Install Python package in the container/overlay
singularity exec \\
  --nv --containall \\
  -B "${CACHE_ROOT}/cache/ov:${DOCKER_USER_HOME}/.cache/ov:rw" \\
  -B "${CACHE_ROOT}/cache/pip:${DOCKER_USER_HOME}/.cache/pip:rw" \\
  -B "${CACHE_ROOT}/cache/glcache:${DOCKER_USER_HOME}/.cache/nvidia/GLCache:rw" \\
  -B "${CACHE_ROOT}/cache/computecache:${DOCKER_USER_HOME}/.nv/ComputeCache:rw" \\
  -B "${CACHE_ROOT}/logs:${DOCKER_USER_HOME}/.nvidia-omniverse/logs:rw" \\
  -B "${CACHE_ROOT}/data:${DOCKER_USER_HOME}/.local/share/ov/data:rw" \\
  -B "${CACHE_ROOT}/documents:${DOCKER_USER_HOME}/Documents:rw" \\
  -B "${CACHE_ROOT}/cache/wandb:${DOCKER_USER_HOME}.cache/wandb:rw" \\
  -B "${HOME}/.config/wandb:${DOCKER_USER_HOME}.config/wandb:rw" \\
  $SIF_IMAGE bash -lc "
    python3 -m pip install --user --upgrade pip
    python3 -m pip install --user $PKG
    python3 -c 'import $PKG; print(\"$PKG version:\", $PKG.__version__)'
  "
EOF

# submit the job
echo "SLURM job to install $PKG..."
ssh -o UserKnownHostsFile=/dev/null -o StrictHostKeyChecking=no ${HPC_HOST} "cd ~/rloclab && sbatch --job-name='update_${PKG}_${USER}' --mail-user='${USER}@nyu.edu' '$TMP_SLURM' '$@'"
#sbatch --job-name="update_${PKG}_${USER}" "$TMP_SLURM"

#  clean up
rm "$TMP_SLURM"
