#!/bin/bash
#SBATCH --job-name=linear_probe_bcic2a_eegpt_byol
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=24:00:00          # hh:mm:ss
#SBATCH --nodes=1             # number of nodes
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --partition=A100         # or V100, A100, etc.

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
for i in {1..9}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset.datasets_dir="/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}" \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/862615/2026-06-24/15-29-44/checkpoints/best-epoch=45-valid/average_valid_loss=0.1024.ckpt\" \
        optimizer.encoder_lr=0
done
