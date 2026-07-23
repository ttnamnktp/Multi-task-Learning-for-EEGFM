#!/bin/bash
#SBATCH --job-name=linear_probe_kaggle_ern_cbramod_all
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
echo "LINEAR"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859555/2026-06-22/16-39-13/checkpoints/best-epoch=23-valid/average_valid_loss=0.9693.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859555/2026-06-22/16-39-13/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "OGR"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859558/2026-06-22/16-40-12/checkpoints/best-epoch=27-valid/average_valid_loss=0.8953.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859558/2026-06-22/16-40-12/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "FAMO lr=0.05 gamma=1e-5"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/864960/2026-06-26/16-34-07/checkpoints/best-epoch=22-valid/average_valid_loss=1.4108.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/864960/2026-06-26/16-34-07/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "FAMO lr=0.025 gamma=1e-3"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/864959/2026-06-26/16-33-10/checkpoints/best-epoch=20-valid/average_valid_loss=1.4004.ckpt\" \
        optimizer.encoder_lr=0
done


echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/864959/2026-06-26/16-33-10/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "RECONSTRUCTION"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879352/2026-07-02/15-57-38/checkpoints/best-epoch=47-valid/average_valid_loss=0.0036.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879352/2026-07-02/15-57-38/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "CONTRASTIVE"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879351/2026-07-02/15-21-39/checkpoints/best-epoch=18-valid/average_valid_loss=2.2420.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/879351/2026-07-02/15-21-39/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done

echo "##########################################################"
echo "FAMO ECO"
echo "##########################################################"

echo "=========================================================="
echo "Train với best pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859551/2026-06-22/16-28-32/checkpoints/best-epoch=29-valid/average_valid_loss=0.8738.ckpt\" \
        optimizer.encoder_lr=0
done

echo "=========================================================="
echo "Train với last pretrained checkpoint"
echo "=========================================================="

# Launch the training
for i in {1..5}
do
    SUBJECT=$(printf "A%02d" $i)

    srun python -m src.train \
        --config-name example_config_downstream \
        dataset=kaggle_ern/v1 \
        model=cbramod/downstream \
        dataset.fold=$i \
        model.pretrained_ckpt=\"/home/infres/ttran-25/eegfm/outputs_pretrain/cbramod/L12_8/859551/2026-06-22/16-28-32/checkpoints/last.ckpt\" \
        optimizer.encoder_lr=0
done