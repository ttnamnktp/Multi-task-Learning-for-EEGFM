#!/bin/bash
#SBATCH --job-name=linear_probe_physiomi_eegpt_all
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
echo "Train với best pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877240/2026-06-30/17-51-21/checkpoints/best-epoch=31-valid/average_valid_loss=0.7611.ckpt\" \
        optimizer.encoder_lr=0
done


echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877240/2026-06-30/17-51-21/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done


echo "##########################################################"
echo "CONTRASTIVE"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877241/2026-06-30/18-14-36/checkpoints/best-epoch=19-valid/average_valid_loss=3.1485.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877241/2026-06-30/18-14-36/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "LINEAR"
echo "##########################################################"

# echo "=========================================================="
# echo "Train với best pretrained checkpoint"
# echo "=========================================================="

# for i in {0..3}
# do
#     FOLD=$(printf "%02d" $i)

#     srun python -m src.train \
#         --config-name example_config_downstream \
#         dataset=physiomi/v1 \
#         dataset.cv_fold_index=$i \
#         model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877244/2026-06-30/19-33-30/checkpoints/best-epoch=19-valid/average_valid_loss=1.9617.ckpt\" \
#         optimizer.encoder_lr=0
# done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/877244/2026-06-30/19-33-30/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done


echo "##########################################################"
echo "OGR"
echo "##########################################################"

# echo "=========================================================="
# echo "Train với best pretrained checkpoint"
# echo "=========================================================="

# for i in {0..3}
# do
#     FOLD=$(printf "%02d" $i)

#     srun python -m src.train \
#         --config-name example_config_downstream \
#         dataset=physiomi/v1 \
#         dataset.cv_fold_index=$i \
#         model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853901/2026-06-17/10-22-49/checkpoints/best-epoch=6-valid/average_valid_loss=2.0809.ckpt\" \
#         optimizer.encoder_lr=0
# done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853901/2026-06-17/10-22-49/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done


echo "##########################################################"
echo "FAMO"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="


echo "##########################################################"
echo "ECO FAMO lr 0.05 gamma 1e-5"
echo "##########################################################"

# echo "=========================================================="
# echo "Train với best pretrained checkpoint"
# echo "=========================================================="

# for i in {0..3}
# do
#     FOLD=$(printf "%02d" $i)

#     srun python -m src.train \
#         --config-name example_config_downstream \
#         dataset=physiomi/v1 \
#         dataset.cv_fold_index=$i \
#         model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853898/2026-06-17/10-21-53/checkpoints/best-epoch=26-valid/average_valid_loss=1.9843.ckpt\" \
#         optimizer.encoder_lr=0
# done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/853898/2026-06-17/10-21-53/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done


echo "##########################################################"
echo "ECO FAMO lr 0.025 gamma 1e-3"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/860906/2026-06-23/16-21-43/checkpoints/best-epoch=19-valid/average_valid_loss=2.0677.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

for i in {0..3}
do
    FOLD=$(printf "%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=physiomi/v1 \
        dataset.cv_fold_index=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/eegpt/L_822/860906/2026-06-23/16-21-43/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done