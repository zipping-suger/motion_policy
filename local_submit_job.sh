#!/usr/bin/env bash

# submit_train_job.sh
# Usage: ./submit_train_job.sh <project_dir> [train_args...]

set -euo pipefail

echo "Generating SLURM job script..."

# Get a timestamp for unique log filename
timestamp=$(date +"%Y%m%d-%H%M%S")
logfile="slurm-train-${timestamp}.out"

cat <<EOT > job.sh
#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=2:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --output=$logfile
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-${timestamp}"

# Activate virtual environment if needed (uncomment and modify as needed)
# source /path/to/your/venv/bin/activate

# Change to project directory and run training script
cd $HOME
cd collision-net/
export CUDA_LAUNCH_BLOCKING=1
python train.py
# python interactive_collision_viewer.py
EOT

echo "Submitting job to SLURM..."
job_output=$(sbatch job.sh "$@")
job_id=$(echo "$job_output" | awk '{print $NF}')
echo "Submitted batch job $job_id"

rm job.sh
