#!/bin/bash
# TURBO LIBERO evaluation - optimized for ~10x speedup
set -e

echo "========================================"
echo "TURBO OpenPI LIBERO Evaluation"
echo "8 parallel envs, 10 replan, 5 denoise"
echo "========================================"

cd /mnt/nvme1/libero
source .venv_converter/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/openpi/src:$PWD/third_party/libero:$PWD/openpi/packages

mkdir -p docs/results

# TURBO settings
NUM_ENVS="${NUM_ENVS:-8}"
REPLAN="${REPLAN:-10}"
DENOISE="${DENOISE:-5}"
CONFIG="${CONFIG:-pi05_libero}"
CHECKPOINT="${CHECKPOINT:-.cache/pi05_libero}"

suites=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")

for suite in "${suites[@]}"; do
    echo "================================================="
    echo "TURBO eval: $suite"
    echo "  Envs=$NUM_ENVS, Replan=$REPLAN, Denoise=$DENOISE"
    echo "================================================="
    
    MUJOCO_GL=egl python openpi_integration/libero_turbo_eval.py \
        --config-name "$CONFIG" \
        --checkpoint-dir "$CHECKPOINT" \
        --num-envs "$NUM_ENVS" \
        --replan-steps "$REPLAN" \
        --num-denoise-steps "$DENOISE" \
        --task-suite-name "$suite" \
        --collect-timing \
        2>&1 | tee "docs/results/${suite}_turbo.log"
done

echo "========================================"
echo "All TURBO evaluations completed!"
echo "Results in docs/results/*_turbo.log"
echo "========================================"
