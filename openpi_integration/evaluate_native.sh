#!/bin/bash
# Native LIBERO evaluation - no websocket server overhead
# This runs the model directly in the same process as the simulation
set -e

echo "========================================"
echo "NATIVE OpenPI LIBERO Evaluation"
echo "No websocket server - direct inference"
echo "========================================"

source .venv_converter/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/openpi/src:$PWD/third_party/libero:$PWD/openpi/packages

mkdir -p docs/results

# Model configuration
CONFIG_NAME="${CONFIG_NAME:-pi05_libero}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-.cache/pi05_libero}"
DEVICE="${DEVICE:-cuda}"

suites=("libero_spatial" "libero_object" "libero_goal" "libero_10" "libero_90")

for suite in "${suites[@]}"; do
    echo "================================================="
    echo "Evaluating: $suite (native inference)"
    echo "================================================="
    
    MUJOCO_GL=egl python openpi_integration/libero_native_eval.py \
        --config-name "$CONFIG_NAME" \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --pytorch-device "$DEVICE" \
        --task-suite-name "$suite" \
        --collect-timing \
        | tee "docs/results/${suite}_native.log"
done

echo "========================================"
echo "All suites completed!"
echo "Results in docs/results/*_native.log"
echo "========================================"
