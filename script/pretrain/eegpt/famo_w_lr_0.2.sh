#!/bin/bash
#SBATCH --job-name=pretrain_eegpt_famo
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --partition=A100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=10

set -e
set -x

cd /home/infres/ttran-25/eegfm

CONDA_PATH=/home/infres/ttran-25/miniconda3
source "$CONDA_PATH/bin/activate"

conda activate eegpt

nvidia-smi

srun python -m src.pretrain \
    --config-name example_config_pretrain_famo_eegpt_cons_reg \
    data.batch_size=128 \
    weight_method=famo \
    weight_method.w_lr=0.2
