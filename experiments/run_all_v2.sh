#!/bin/bash
# Submit all 8 v2 experiments: 2 datasets x 2 backbones x 2 dynamics
# Usage: bash experiments/run_all_v2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMPLATE="$SCRIPT_DIR/slurm_template_v2.sh"
GEN_DIR="$SCRIPT_DIR/generated_v2"

mkdir -p "$GEN_DIR"

# Format: dataset|backbone|backbone_name|dynamics|experiment_name|batch_size|img_size
EXPERIMENTS=(
    "lquan9/collect1|cnn|cnn|mlp|cnn_mlp_c1|64|128"
    "lquan9/collect1|cnn|cnn|rssm|cnn_rssm_c1|64|128"
    "lquan9/collect1|dinov2|dinov2_vitb14|mlp|dino_mlp_c1|32|224"
    "lquan9/collect1|dinov2|dinov2_vitb14|rssm|dino_rssm_c1|32|224"
    "lquan9/collect1_unfiltered|cnn|cnn|mlp|cnn_mlp_c1u|64|128"
    "lquan9/collect1_unfiltered|cnn|cnn|rssm|cnn_rssm_c1u|64|128"
    "lquan9/collect1_unfiltered|dinov2|dinov2_vitb14|mlp|dino_mlp_c1u|32|224"
    "lquan9/collect1_unfiltered|dinov2|dinov2_vitb14|rssm|dino_rssm_c1u|32|224"
)

echo "Submitting ${#EXPERIMENTS[@]} v2 world model experiments..."
echo ""

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r DATASET BACKBONE BACKBONE_NAME DYNAMICS EXPERIMENT_NAME BATCH_SIZE IMG_SIZE <<< "$exp"

    OUTFILE="$GEN_DIR/${EXPERIMENT_NAME}.sh"

    sed \
        -e "s|__DATASET__|${DATASET}|g" \
        -e "s|__BACKBONE_NAME__|${BACKBONE_NAME}|g" \
        -e "s|__BACKBONE__|${BACKBONE}|g" \
        -e "s|__DYNAMICS__|${DYNAMICS}|g" \
        -e "s|__EXPERIMENT_NAME__|${EXPERIMENT_NAME}|g" \
        -e "s|__BATCH_SIZE__|${BATCH_SIZE}|g" \
        -e "s|__IMG_SIZE__|${IMG_SIZE}|g" \
        -e "s|EXPERIMENT_NAME_PLACEHOLDER|${EXPERIMENT_NAME}|g" \
        "$TEMPLATE" > "$OUTFILE"

    chmod +x "$OUTFILE"

    echo "Submitting: ${EXPERIMENT_NAME}"
    sbatch "$OUTFILE"
done

echo ""
echo "All ${#EXPERIMENTS[@]} jobs submitted. Monitor with: squeue -u \$USER"
echo ""
echo "Summary:"
echo "  Datasets:  collect1, collect1_unfiltered"
echo "  Backbones: cnn, dinov2"
echo "  Dynamics:  mlp, rssm"
echo "  Total:     ${#EXPERIMENTS[@]} experiments"
echo "  Output:    $GEN_DIR/"
