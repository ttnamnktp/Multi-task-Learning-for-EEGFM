#!/bin/bash
#SBATCH --job-name=linear_probe_bcic2a_eegpt_all_last_ckpt
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

echo "##########################################################"
echo "RECONSTRUCTION"
echo "##########################################################"

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..9}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=bciciv2a/v1 \
        dataset.datasets_dir="/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}" \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877240/2026-06-30/17-51-21/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "CONTRASTIVE"
echo "##########################################################"

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..9}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=bciciv2a/v1 \
        dataset.datasets_dir="/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}" \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877241/2026-06-30/18-14-36/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "LINEAR"
echo "##########################################################"

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..9}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=bciciv2a/v1 \
        dataset.datasets_dir="/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}" \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877244/2026-06-30/19-33-30/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done


echo "##########################################################"
echo "OGR"
echo "##########################################################"

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..9}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=bciciv2a/v1 \
        dataset.datasets_dir="/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}" \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853901/2026-06-17/10-22-49/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done


echo "##########################################################"
echo "ECO FAMO"
echo "##########################################################"

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..9}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=bciciv2a/v1 \
        dataset.datasets_dir="/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}" \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853898/2026-06-17/10-21-53/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done