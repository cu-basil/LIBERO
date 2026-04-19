#!/bin/bash
set -e

echo "Starting OpenPI Server natively using PyTorch and explicit path on port 8011..."
source .venv_converter/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/openpi/src:$PWD/third_party/libero

# Run the server in the background and redirect logs
python openpi/scripts/serve_policy.py --env LIBERO --port 8011 policy:checkpoint --policy.config pi05_libero --policy.dir .cache/pi05_libero > .cache/server.log 2>&1 &
SERVER_PID=$!
trap "echo 'Cleaning up server...'; kill $SERVER_PID" EXIT INT TERM

echo "Waiting for the 12GB PyTorch model to initialize in the background (~40s)..."
sleep 45 

echo "Checking the server logs to make sure it started successfully:"
head -n 20 .cache/server.log

echo "Starting OpenPI Client identically from .venv312 on port 8011..."
source .venv312/bin/activate
export PYTHONPATH=$PYTHONPATH:$PWD/openpi/src:$PWD/third_party/libero:$PWD/openpi/packages
export CLIENT_ARGS="--args.task-suite-name libero_10 --args.port 8011"

echo "Collecting simulation results... saving output to .cache/libero_client_eval.log"
MUJOCO_GL=egl python openpi_integration/libero_client_eval.py $CLIENT_ARGS | tee .cache/libero_client_eval.log

# Cleanup handled by trap
