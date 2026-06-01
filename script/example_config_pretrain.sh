#!/bin/bash
#SBATCH --job-name=example_config_pretrain
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=10

set -e
set -x

cd /home/infres/ttran-25/eegfm

CONDA_PATH=/home/infres/ttran-25/miniconda3
source "$CONDA_PATH/bin/activate"

conda activate eegpt

nvidia-smi

srun python -m src.pretrain \
    --config-name example_config_pretrain