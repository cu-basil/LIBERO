#!/bin/bash
# Native LIBERO evaluation launcher
# All logic is in libero_native_eval.py
set -e

cd /mnt/nvme1/libero
source .venv312/bin/activate
export PYTHONPATH=$PWD:$PWD/openpi/src:$PWD/openpi/packages
mkdir -p docs/results

# Pass all arguments through to Python script
MUJOCO_GL=egl python openpi_integration/libero_native_eval.py "$@"
