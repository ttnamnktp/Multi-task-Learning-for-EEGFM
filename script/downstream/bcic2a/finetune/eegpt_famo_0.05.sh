#!/bin/bash
#SBATCH --job-name=downstream_bcic2a_eegpt_famo_0.05
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
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853898/2026-06-17/10-21-53/checkpoints/best-epoch=26-valid/average_valid_loss=1.9843.ckpt\"
done
