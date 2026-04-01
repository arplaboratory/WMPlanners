#!/usr/bin/env bash
# scripts/apptainer_setup.sh
# Install dependencies and build Apptainer from source with --with-suid.

set -euo pipefail

GOVERSION=${GOVERSION:-1.25.6}
APPTAINER_DIR=${APPTAINER_DIR:-$HOME/apptainer}
INSTALL_PREFIX=${INSTALL_PREFIX:-/usr/local}

echo "== Apptainer build helper =="
echo "Go version: $GOVERSION"
echo "Apptainer source dir: $APPTAINER_DIR"
echo "Install prefix: $INSTALL_PREFIX"

if [ $(id -u) -eq 0 ]; then
  echo "Do not run this script as root. It will use sudo when needed." >&2
  exit 1
fi

ensure_sudo(){
  if ! sudo -v >/dev/null 2>&1; then
    echo "This script requires sudo privileges to install packages and to run 'make install'." >&2
    exit 1
  fi
}

ensure_sudo

echo "[1/6] Installing build dependencies (apt)..."
sudo apt-get update
sudo apt-get install -y \ 
  build-essential libseccomp-dev pkg-config uidmap squashfs-tools fuse2fs fuse-overlayfs \ 
  fakeroot cryptsetup curl wget git libfuse-dev libfuse3-dev libglib2.0-dev zlib1g-dev libsubid-dev ca-certificates

echo "[2/6] Installing Go ${GOVERSION} to /usr/local/go (requires sudo)"
TMPGOPKG="/tmp/go${GOVERSION}.linux-amd64.tar.gz"
if [ ! -x /usr/local/go/bin/go ]; then
  wget -q -O "$TMPGOPKG" "https://dl.google.com/go/go${GOVERSION}.linux-amd64.tar.gz"
  sudo tar -C /usr/local -xzf "$TMPGOPKG"
  rm -f "$TMPGOPKG"
fi

export PATH="/usr/local/go/bin:$PATH"
echo "go version: $(go version)"

echo "[3/6] Clone Apptainer sources into $APPTAINER_DIR"
if [ -d "$APPTAINER_DIR" ]; then
  echo "Apptainer dir exists; pulling latest tags"
  git -C "$APPTAINER_DIR" fetch --tags
else
  git clone https://github.com/apptainer/apptainer.git "$APPTAINER_DIR"
fi

cd "$APPTAINER_DIR"
git fetch --tags
LATEST_TAG=$(git describe --tags --abbrev=0)
echo "Checking out latest tag: $LATEST_TAG"
git checkout "$LATEST_TAG"

echo "[4/6] Configure and build Apptainer with --with-suid"
./mconfig --with-suid --prefix="$INSTALL_PREFIX"
make -C builddir -j"$(nproc)"

echo "[5/6] Installing Apptainer (sudo required)"
sudo make -C builddir install

echo "[6/6] Post-install checks"
STARTER="$INSTALL_PREFIX/libexec/apptainer/bin/starter-suid"
if [ -f "$STARTER" ]; then
  ls -la "$STARTER"
else
  echo "Warning: starter-suid not found at $STARTER" >&2
fi

echo "Ensure subuid/subgid mappings exist for your user (will append if missing)."
if ! grep -q "^${USER}:" /etc/subuid 2>/dev/null; then
  echo "${USER}:100000:65536" | sudo tee -a /etc/subuid
fi
if ! grep -q "^${USER}:" /etc/subgid 2>/dev/null; then
  echo "${USER}:100000:65536" | sudo tee -a /etc/subgid
fi

echo "Apptainer build+install complete. Run 'apptainer --version' to verify."
echo "If starter-suid is not setuid, run: sudo make -C $APPTAINER_DIR/builddir install"

# Optional: pull Isaac Sim base SIF, create sandbox, install project and externals, and build final SIF
WITH_SIF=false
if [ "${1:-}" = "--with-sif" ] || [ "${WITH_SIF_ENV:-}" = "1" ]; then
  WITH_SIF=true
fi

if [ "$WITH_SIF" = true ]; then
  echo "\n[extra] Building and packaging Isaac Sim SIF and project into sandbox..."
  SIF_IMAGE="${SIF_IMAGE:-$HOME/isaac-sim_5.1.0.sif}"
  SANDBOX_DIR="${SANDBOX_DIR:-$HOME/isaac_container}"
  BUILT_SIF="${BUILT_SIF:-$HOME/isaac-sim_ds3001.sif}"
  ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"
  PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$PWD" && pwd)}"
  SCRATCH_TMP="${SCRATCH_TMP:-$HOME/tmp}"

  mkdir -p "$SCRATCH_TMP"

  echo "Pulling base Isaac Sim SIF to $SIF_IMAGE (this may prompt for login)..."
  apptainer pull "$SIF_IMAGE" docker://nvcr.io/nvidia/isaac-sim:5.1.0 || { echo "Failed to pull base SIF"; exit 1; }

  echo "Creating writable sandbox at $SANDBOX_DIR"
  rm -rf "$SANDBOX_DIR" || true
  apptainer build --sandbox "$SANDBOX_DIR" "$SIF_IMAGE" || { echo "sandbox build failed"; exit 1; }

  # Ensure IsaacLab cloned
  if [ ! -d "$ISAACLAB_DIR" ]; then
    echo "Cloning IsaacLab into $ISAACLAB_DIR"
    git clone https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR" || true
  fi

  echo "Installing project and externals inside sandbox (Option A)..."
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

"$PYTHON_EXE" -m pip install --upgrade pip setuptools wheel build
"$PYTHON_EXE" -m pip install --no-cache-dir uv || true

# install project editable
"$PYTHON_EXE" -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "/workspace/run[dev]" || \
  "$PYTHON_EXE" -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "/workspace/run"

for pkg in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl; do
  target="/workspace/isaaclab/source/${pkg}"
  [ -d "$target" ] && "$PYTHON_EXE" -m uv pip install --python "$PYTHON_EXE" --no-cache-dir -e "$target" || true
done

# run original externals installer
bash "/workspace/run/scripts/install_externals.sh" || true
'

  echo "Rebuilding final SIF at $BUILT_SIF"
  apptainer build "$BUILT_SIF" "$SANDBOX_DIR" || { echo "Final SIF build failed"; exit 1; }

  echo "Final SIF created: $BUILT_SIF"
fi

exit 0
