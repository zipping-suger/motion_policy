#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=rtx_3090:1
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --output=slurm-train-20250519-064739.out
#SBATCH --mail-type=END
#SBATCH --mail-user=name@mail
#SBATCH --job-name="training-20250519-064739"

# Activate virtual environment if needed (uncomment and modify as needed)
# source /path/to/your/venv/bin/activate

# Change to project directory and run training script
cd /home/zippingsugar
cd motion_policy
ulimit -n 4096  
wandb login e69097b8c1bd646d9218e652823487632097445d
python run_training.py configs/train_cfg.yaml 
