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
cd motion_policy
ulimit -n 4096  
wandb login e69097b8c1bd646d9218e652823487632097445d
python run_training.py configs/train_cfg.yaml 
EOT

echo "Submitting job to SLURM..."
job_output=$(sbatch job.sh "$@")
job_id=$(echo "$job_output" | awk '{print $NF}')
echo "Submitted batch job $job_id"

rm job.sh
