#!/usr/bin/env bash

# run_singularity.sh
# Hardcoded paths and arguments for simplicity

set -euo pipefail

# Hardcoded paths and arguments
CONTAINER_IMAGE="/cluster/scratch/yixili/cn_env.sif"  # Added .sif extension
PIPELINE_DIR="/cluster/home/yixili/motion_policy"

echo "Starting Singularity container: $CONTAINER_IMAGE"

# Set WANDB_API_KEY as environment variable instead of using wandb login
export WANDB_API_KEY="e69097b8c1bd646d9218e652823487632097445d"

singularity exec \
  --nv \
  --containall --writable-tmpfs \
  --bind "${PIPELINE_DIR}:/motion_policy" \
  --env PYTHONUNBUFFERED=1 \
  --env PYTHONPATH="/motion_policy:\${PYTHONPATH:-}" \
  --env NVIDIA_DRIVER_CAPABILITIES=all \
  --env ACCEPT_EULA=Y \
  "${CONTAINER_IMAGE}" \
  python3 -u /motion_policy/run_training.py configs/train_cfg.yaml

echo "Completed run."
