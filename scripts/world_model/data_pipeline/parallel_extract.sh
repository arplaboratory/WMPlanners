#!/bin/bash
# Parallel MCAP extraction script
# Usage: ./scripts/world_model/data_pipeline/parallel_extract.sh [options] <directory>

# Default values
JOBS=4
DRY_RUN=0
OUTPUT_PARENT="pipeline_out"
MERGE=0
HF_REPO=""

usage() {
    echo "Usage: $0 [options] <directory>"
    echo "Options:"
    echo "  -j, --jobs <n>      Number of parallel jobs (default: 4)"
    echo "  --dry-run           Print commands instead of executing them"
    echo "  --output <dir>      Parent output directory (default: pipeline_out)"
    echo "  --merge             Merge all bags into a single dataset (default: false)"
    echo "  --push-to-hub <id>  Hugging Face Repo ID (e.g. dmody1/wm-data)"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 -j 8 --merge data/my_mcaps"
    exit 1
}

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --output)
            OUTPUT_PARENT="$2"
            shift 2
            ;;
        --merge)
            MERGE=1
            shift
            ;;
        --push-to-hub)
            HF_REPO="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Unknown option: $1"
            usage
            ;;
        *)
            SEARCH_DIR="$1"
            shift
            ;;
    esac
done

if [ -z "$SEARCH_DIR" ]; then
    echo "Error: Directory not specified."
    usage
fi

if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' does not exist."
    exit 1
fi

# Export variables for parallel
export DRY_RUN
export OUTPUT_PARENT
export MERGE
export HF_REPO

# Process a single mcap file (Independent mode)
process_mcap_indep() {
    local file="$1"
    local filename=$(basename "$file")
    local bag_stem="${filename%.*}"
    local robot_name=$(echo "$filename" | cut -d'_' -f1)
    local bag_out="${OUTPUT_PARENT}/${bag_stem}"

    local hf_arg=""
    if [ -n "$HF_REPO" ]; then
        hf_arg="--push-to-hub \"$HF_REPO\""
    fi

    local cmd="python -m scripts.world_model.data_pipeline.run_pipeline \
        --mcap \"$file\" \
        --robot \"$robot_name\" \
        --frame-skip 6 --episode-len 50 --stride 50 \
        --target-count 5000 \
        --output \"$bag_out\" \
        $hf_arg"

    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "\033[0;33m[DRY RUN]\033[0m $cmd"
    else
        echo -e "\033[0;32m[RUNNING]\033[0m $cmd"
        eval "$cmd"
    fi
}

# Extract a single mcap file (Merge mode - Stage 0 only)
extract_only() {
    local file="$1"
    local filename=$(basename "$file")
    local bag_stem="${filename%.*}"
    local robot_name=$(echo "$filename" | cut -d'_' -f1)
    local staging_dir="${OUTPUT_PARENT}/staging/${bag_stem}"

    local cmd="python -m scripts.world_model.extract_dataset \
        --mcap \"$file\" \
        --robot \"$robot_name\" \
        --output \"$staging_dir\""

    if [ "$DRY_RUN" -eq 1 ]; then
        echo -e "\033[0;33m[DRY RUN]\033[0m $cmd"
    else
        echo -e "\033[0;32m[EXTRACTING]\033[0m $filename"
        eval "$cmd" > /dev/null
    fi
}

export -f process_mcap_indep
export -f extract_only

# Check dependencies
if ! command -v parallel &> /dev/null; then
    echo "Error: GNU Parallel is not installed."
    exit 1
fi

echo "Searching for .mcap and .bag files in '$SEARCH_DIR'..."
FILES_COUNT=$(find "$SEARCH_DIR" -name "*.mcap" -o -name "*.bag" | wc -l)
if [ "$FILES_COUNT" -eq 0 ]; then
    echo "No .mcap or .bag files found."
    exit 0
fi

if [ "$MERGE" -eq 1 ]; then
    echo "--- MERGED MODE ---"
    echo "Found $FILES_COUNT files. Extracting in parallel with $JOBS jobs..."
    
    # Define directories for both run and dry-run
    STAGING="${OUTPUT_PARENT}/staging"
    MERGED_DIR="${OUTPUT_PARENT}/merged_extracted"
    
    # 1. Parallel Extraction
    mkdir -p "${STAGING}"
    find "$SEARCH_DIR" -name "*.mcap" -o -name "*.bag" | parallel -j "$JOBS" extract_only {}
    
    # 2. Merge Manifests
    if [ "$DRY_RUN" -eq 0 ]; then
        echo "Merging manifests..."
        python3 -m scripts.world_model.data_pipeline.merge_manifests \
            --staging "${STAGING}" \
            --output "${MERGED_DIR}"
            
        # Final Pipeline Run (Stage 1 & 2)
        echo "Running final pipeline (Build & Select)..."
        hf_arg=""
        if [ -n "$HF_REPO" ]; then
            hf_arg="--push-to-hub \"$HF_REPO\""
        fi
        
        python -m scripts.world_model.data_pipeline.run_pipeline \
            --bags "${MERGED_DIR}" \
            --output "${OUTPUT_PARENT}" \
            --frame-skip 6 --episode-len 50 --stride 50 \
            --target-count 5000 \
            $hf_arg
    else
        hf_arg=""
        if [ -n "$HF_REPO" ]; then
            hf_arg="--push-to-hub \"$HF_REPO\""
        fi
        final_cmd="python -m scripts.world_model.data_pipeline.run_pipeline --bags \"${MERGED_DIR}\" --output \"${OUTPUT_PARENT}\" --frame-skip 6 --episode-len 50 --stride 50 --target-count 5000 $hf_arg"
        echo -e "\033[0;33m[DRY RUN]\033[0m Would merge manifests and run: $final_cmd"
    fi
else
    echo "--- INDEPENDENT MODE ---"
    echo "Found $FILES_COUNT files. Processing in parallel with $JOBS jobs..."
    find "$SEARCH_DIR" -name "*.mcap" -o -name "*.bag" | parallel -j "$JOBS" --joblog "${OUTPUT_PARENT}/parallel_jobs.log" process_mcap_indep {}
fi

echo "Done."
