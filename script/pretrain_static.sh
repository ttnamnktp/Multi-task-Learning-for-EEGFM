#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --partition=P100              # Partition to submit to (A100, V100, etc.)
#SBATCH --gres=gpu:2                 # Request 1 GPU
#SBATCH --time=1-00:00:00               # Time limit for the job (hh:mm:ss)


set -e  # crash nếu lỗi
set -x  # print command

cd /home/infres/ttran-25/eegfm

# Load your conda environment
CONDA_PATH=/home/infres/ttran-25/miniconda3
source "$CONDA_PATH/bin/activate"
conda activate eegpt

# ======================
# kiểm tra GPU
# ======================
nvidia-smi

# Launch the training
srun python -m src.pretrain \
    --config-name config_pretrain_static \
