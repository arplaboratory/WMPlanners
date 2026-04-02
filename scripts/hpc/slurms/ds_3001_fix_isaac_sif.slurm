#!/bin/bash
# DS-GA 3001 — Fix NVIDIA Isaac Sim 5.1.0 Permissions
# Re-packs the SIF with 755 recursive permissions on /isaac-sim.
# Run this ONCE to fix the "Permission denied" error on /isaac-sim/python.sh.

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=ds3001_fix_isaac
#SBATCH --account=ds_ga_3001_003-2026sp
#SBATCH --partition=g2-standard-12
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Output location on /scratch
SIF_OUT="/scratch/$USER/isaac-sim_5.1.0_fixed.sif"

# Use node-local SSD (Local disk) to avoid Lustre permission/noexec issues during conversion
TMP_SANDBOX="${SLURM_TMPDIR}/isaac_sandbox"

echo "[NODE] Pulling base image and building sandbox to local SSD..."
# Use apptainer if available; fallback to singularity
CMD="apptainer"
which apptainer &>/dev/null || CMD="singularity"

$CMD build --sandbox "${TMP_SANDBOX}" docker://nvcr.io/nvidia/isaac-sim:5.1.0

echo "[NODE] Fixing permissions recursively on /isaac-sim inside the sandbox..."
# Host-side chmod on the sandbox directory itself as the current user
chmod -R 755 "${TMP_SANDBOX}/isaac-sim"

# Also ensure setuid/setgid bits aren't causing issues (though less likely)
find "${TMP_SANDBOX}/isaac-sim" -type d -exec chmod 755 {} +
find "${TMP_SANDBOX}/isaac-sim" -type f -exec chmod 644 {} +
find "${TMP_SANDBOX}/isaac-sim" -name "*.sh" -exec chmod 755 {} +
find "${TMP_SANDBOX}" -name "python3" -exec chmod 755 {} +

echo "[NODE] Building fixed SIF to ${SIF_OUT}..."
rm -f "${SIF_OUT}"
$CMD build "${SIF_OUT}" "${TMP_SANDBOX}"

echo "[NODE] SUCCESS! Please update your scripts to use: ${SIF_OUT}"
