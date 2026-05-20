# Project Source Map — `eegfm`

> Được tạo lúc: 2026-05-13 14:53:47  
> Thư mục gốc: `/home/infres/ttran-25/eegfm`  
> Tổng số file: **71**


---

## Mục lục

- [`configs/config_downstream.yaml`](#configsconfig-downstreamyaml)
- [`configs/config_pretrain.yaml`](#configsconfig-pretrainyaml)
- [`configs/config_pretrain_mape.yaml`](#configsconfig-pretrain-mapeyaml)
- [`configs/config_pretrain_static.yaml`](#configsconfig-pretrain-staticyaml)
- [`configs/config_pretrain_tuev.yaml`](#configsconfig-pretrain-tuevyaml)
- [`configs/dataset/bciciv2a/v1.yaml`](#configsdatasetbciciv2av1yaml)
- [`configs/dataset/pretraining_lmdb/v1.yaml`](#configsdatasetpretraining-lmdbv1yaml)
- [`configs/dataset/synthetic/base.yaml`](#configsdatasetsyntheticbaseyaml)
- [`configs/dataset/synthetic/v1.yaml`](#configsdatasetsyntheticv1yaml)
- [`configs/dataset/synthetic/v2.yaml`](#configsdatasetsyntheticv2yaml)
- [`configs/hydra/default.yaml`](#configshydradefaultyaml)
- [`configs/hydra/downstream.yaml`](#configshydradownstreamyaml)
- [`configs/hydra/pretrain.yaml`](#configshydrapretrainyaml)
- [`configs/loss/pretrain_all.yaml`](#configslosspretrain-allyaml)
- [`configs/loss/pretrain_basic.yaml`](#configslosspretrain-basicyaml)
- [`configs/model/cbramod/downstream.yaml`](#configsmodelcbramoddownstreamyaml)
- [`configs/model/cbramod/pretrain.yaml`](#configsmodelcbramodpretrainyaml)
- [`configs/model/cbramod/variants.yaml`](#configsmodelcbramodvariantsyaml)
- [`configs/model/eegpt/downstream.yaml`](#configsmodeleegptdownstreamyaml)
- [`configs/model/eegpt/pretrain.yaml`](#configsmodeleegptpretrainyaml)
- [`configs/model/eegpt/variants.yaml`](#configsmodeleegptvariantsyaml)
- [`configs/module/classification_module.yaml`](#configsmoduleclassification-moduleyaml)
- [`configs/module/eegpt_pretrain_module.yaml`](#configsmoduleeegpt-pretrain-moduleyaml)
- [`configs/weighting/mape.yaml`](#configsweightingmapeyaml)
- [`configs/weighting/static.yaml`](#configsweightingstaticyaml)
- [`script/downstream.sh`](#scriptdownstreamsh)
- [`script/dumpscript.sh`](#scriptdumpscriptsh)
- [`script/dumpscript2.sh`](#scriptdumpscript2sh)
- [`script/pretrain.sh`](#scriptpretrainsh)
- [`script/pretrain_mape.sh`](#scriptpretrain-mapesh)
- [`script/pretrain_static.sh`](#scriptpretrain-staticsh)
- [`script/pretrain_tuev.sh`](#scriptpretrain-tuevsh)
- [`src/pretrain.py`](#srcpretrainpy)
- [`src/train.py`](#srctrainpy)
- [`src/data/__init__.py`](#srcdata--init--py)
- [`src/data/bciciv2a.py`](#srcdatabciciv2apy)
- [`src/data/data.py`](#srcdatadatapy)
- [`src/data/pretrain_data.py`](#srcdatapretrain-datapy)
- [`src/data/registry.py`](#srcdataregistrypy)
- [`src/data/synthetic.py`](#srcdatasyntheticpy)
- [`src/models/__init__.py`](#srcmodels--init--py)
- [`src/models/base_model.py`](#srcmodelsbase-modelpy)
- [`src/models/registry.py`](#srcmodelsregistrypy)
- [`src/models/cbramod/__init__.py`](#srcmodelscbramod--init--py)
- [`src/models/cbramod/cbramod.py`](#srcmodelscbramodcbramodpy)
- [`src/models/cbramod/model_components.py`](#srcmodelscbramodmodel-componentspy)
- [`src/models/eegpt/__init__.py`](#srcmodelseegpt--init--py)
- [`src/models/eegpt/eegpt.py`](#srcmodelseegpteegptpy)
- [`src/models/eegpt/eegpt_downstream.py`](#srcmodelseegpteegpt-downstreampy)
- [`src/module/__init__.py`](#srcmodule--init--py)
- [`src/module/base_module.py`](#srcmodulebase-modulepy)
- [`src/module/registry.py`](#srcmoduleregistrypy)
- [`src/module/cbramod/cbramod_pretrain_module.py`](#srcmodulecbramodcbramod-pretrain-modulepy)
- [`src/module/cbramod/config_builder.py`](#srcmodulecbramodconfig-builderpy)
- [`src/module/cbramod/task/base_task.py`](#srcmodulecbramodtaskbase-taskpy)
- [`src/module/cbramod/task/reconstruction.py`](#srcmodulecbramodtaskreconstructionpy)
- [`src/module/downstream/classification_module.py`](#srcmoduledownstreamclassification-modulepy)
- [`src/module/eegpt/config_builder.py`](#srcmoduleeegptconfig-builderpy)
- [`src/module/eegpt/eegpt_pretrain_module.py`](#srcmoduleeegpteegpt-pretrain-modulepy)
- [`src/module/eegpt/gradient_monitor.py`](#srcmoduleeegptgradient-monitorpy)
- [`src/utils/callbacks.py`](#srcutilscallbackspy)
- [`src/utils/metrics.py`](#srcutilsmetricspy)
- [`src/utils/optimizers.py`](#srcutilsoptimizerspy)
- [`src/utils/schedulers.py`](#srcutilsschedulerspy)
- [`src/visualizer/main_visualizer.py`](#srcvisualizermain-visualizerpy)
- [`src/visualizer/model_visualizer.py`](#srcvisualizermodel-visualizerpy)
- [`src/weighting/__init__.py`](#srcweighting--init--py)
- [`src/weighting/base_weighting.py`](#srcweightingbase-weightingpy)
- [`src/weighting/mape_weighting.py`](#srcweightingmape-weightingpy)
- [`src/weighting/registry.py`](#srcweightingregistrypy)
- [`src/weighting/static_weighting.py`](#srcweightingstatic-weightingpy)

---


## `configs/config_downstream.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/config_downstream.yaml`  
**Kích thước:** `0.4 KB`  
**Loại:** `.yaml`

```yaml
defaults:
  - _self_
  - dataset: bciciv2a/v1
  - model: eegpt/downstream
  - module: classification_module
  - hydra: downstream
  
seed: 42

data:
  batch_size: 32
  num_workers: 4

trainer:
  max_epochs: 50
  accelerator: auto
  devices: 1
  precision: 32

optimizer:
  name: adamw
  encoder_lr: 0.000001
  head_lr: 0.001
  adapter_lr: 0.001
  weight_decay: 0.01

scheduler:
  name: cosine
```

---


## `configs/config_pretrain.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/config_pretrain.yaml`  
**Kích thước:** `0.9 KB`  
**Loại:** `.yaml`

```yaml
# configs/config_pretrain_flexible.yaml
defaults:
  - _self_
  - dataset: pretraining_lmdb/v1
  - model: eegpt/pretrain
  - loss: pretrain_all # hiện chưa dùng cái này
  - weighting: static
  - module: eegpt_pretrain_module
  - hydra: default

seed: 42

# Model configuration
# model:
#   variant: large  # Dễ dàng thay đổi variant
#   img_size: [19, 3200]
#   patch_size: 64

# Training
data:
  batch_size: 32
  num_workers: 8
  scale_div: 100.0
  augment_prob: 0.25
  mask_ratio: 0.4

trainer:
  max_epochs: 100 # set to 5 for debugging, defaults is 100
  accelerator: auto
  devices: 1
  precision: 32
  # accumulate_grad_batches: 4 # to solve memory problem when batch_size is too large
  # gradient_clip_val: 1.0
  # gradient_clip_algorithm: norm

optimizer:
  name: adamw
  lr: 0.0001
  weight_decay: 0.01

scheduler:
  name: cosine

# Logging
# logging:
#   log_root: ./logs
#   auto_name: true  # Tự động generate tên experiment
```

---


## `configs/config_pretrain_mape.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/config_pretrain_mape.yaml`  
**Kích thước:** `0.4 KB`  
**Loại:** `.yaml`

```yaml
defaults:
  - _self_
  - dataset: synthetic/v2
  - model: eegpt/v1
  - module: eegpt_pretrain_module
  - hydra: default

seed: 42

weighting:
  name: mape
  params:
    eps: 1.0e-8

data:
  batch_size: 32
  num_workers: 4

trainer:
  max_epochs: 20
  accelerator: auto
  devices: 1
  precision: 32

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.01

scheduler:
  name: cosine
```

---


## `configs/config_pretrain_static.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/config_pretrain_static.yaml`  
**Kích thước:** `0.4 KB`  
**Loại:** `.yaml`

```yaml
defaults:
  - _self_
  - dataset: synthetic/v2
  - model: eegpt/v1
  - module: eegpt_pretrain_module
  - hydra: default
seed: 42

weighting:
  name: static
  params:
    weights:
      loss1: 1.0
      loss2: 1.0
      
data:
  batch_size: 32
  num_workers: 4

trainer:
  max_epochs: 20
  accelerator: auto
  devices: 1
  precision: 32

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.01

scheduler:
  name: cosine
```

---


## `configs/config_pretrain_tuev.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/config_pretrain_tuev.yaml`  
**Kích thước:** `0.4 KB`  
**Loại:** `.yaml`

```yaml
defaults:
  - _self_
  - dataset: pretraining_lmdb/v1
  - model: eegpt/pretrain
  - module: eegpt_pretrain_module
  - hydra: default
seed: 42

weighting:
  name: static
  params:
    weights:
      loss1: 1.0
      loss2: 1.0

data:
  batch_size: 32
  num_workers: 4

trainer:
  max_epochs: 20
  accelerator: auto
  devices: 1
  precision: 32

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.01

scheduler:
  name: cosine
```

---


## `configs/dataset/bciciv2a/v1.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/dataset/bciciv2a/v1.yaml`  
**Kích thước:** `0.4 KB`  
**Loại:** `.yaml`

```yaml
name: bciciv2a
num_classes: 4
num_channels: 22
use_chans_num : 15
time_points : 1024
use_channels_names:
    - FP1
    - FP2
    - F7
    - F3
    - FZ
    - F4
    - F8
    # - T7
    - C3
    - CZ
    - C4
    # - T8
    # - P7
    - P3
    - PZ
    - P4
    # - P8
    - O1
    - O2
datasets_dir: /home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_A01
```

---


## `configs/dataset/pretraining_lmdb/v1.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/dataset/pretraining_lmdb/v1.yaml`  
**Kích thước:** `0.1 KB`  
**Loại:** `.yaml`

```yaml
name: pretraining_lmdb
scale_div: 100
datasets_dir: /home/infres/tran-24/EEGPT_MTL/tueg_16s_200fs_60notch_test
```

---


## `configs/dataset/synthetic/base.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/dataset/synthetic/base.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml
name: synthetic
```

---


## `configs/dataset/synthetic/v1.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/dataset/synthetic/v1.yaml`  
**Kích thước:** `0.2 KB`  
**Loại:** `.yaml`

```yaml
defaults:
  - _self_
  - synthetic/base

name: synthetic
num_channels: 22
seq_len: 1000
num_classes: 4

train_samples: 512
val_samples: 128
test_samples: 128
```

---


## `configs/dataset/synthetic/v2.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/dataset/synthetic/v2.yaml`  
**Kích thước:** `0.2 KB`  
**Loại:** `.yaml`

```yaml
defaults:
  - _self_
  - synthetic/base

name: synthetic
num_channels: 58
seq_len: 1024
num_classes: 4

train_samples: 512
val_samples: 128
test_samples: 128
```

---


## `configs/hydra/default.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/hydra/default.yaml`  
**Kích thước:** `0.3 KB`  
**Loại:** `.yaml`

```yaml
# configs/hydra/default.yaml
run:
  # Tự động tạo thư mục: outputs/tên_dataset/tên_model/ngày/giờ
  dir: outputs/${dataset.name}/${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
sweep:
  dir: multirun/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
```

---


## `configs/hydra/downstream.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/hydra/downstream.yaml`  
**Kích thước:** `0.3 KB`  
**Loại:** `.yaml`

```yaml
# configs/hydra/downstream.yaml
run:
  # Tự động tạo thư mục: outputs/tên_dataset/tên_model/ngày/giờ
  dir: outputs_downstream/${dataset.name}/${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
sweep:
  dir: multirun_downstream/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
```

---


## `configs/hydra/pretrain.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/hydra/pretrain.yaml`  
**Kích thước:** `0.3 KB`  
**Loại:** `.yaml`

```yaml
# configs/hydra/pretrain.yaml
run:
  # Tự động tạo thư mục: outputs/tên_dataset/tên_model/ngày/giờ
  dir: outputs_pretrain/${dataset.name}/${model.name}/${now:%Y-%m-%d}/${now:%H-%M-%S}
sweep:
  dir: multirun_pretrain/${dataset.name}/${model.name}/${now:%Y-%m-%d_%H-%M-%S}
```

---


## `configs/loss/pretrain_all.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/loss/pretrain_all.yaml`  
**Kích thước:** `0.5 KB`  
**Loại:** `.yaml`

```yaml
# configs/loss/pretrain_all.yaml
components:
  contrastive:
    enabled: true
    weight: 1.0
    alignment_type: infoNCE  # infoNCE, mse, decoupled_contrastive
    temperature: 0.2
    
  reconstruction:
    enabled: true
    weight: 1.0
    
  token_prediction:
    enabled: true
    weight: 1.0
    
  order_prediction:
    enabled: false
    weight: 1.0

# Best validation losses (for MTL_GM)
best_losses:
  contrastive: 1.1362
  reconstruction: 0.0041
  token_prediction: 0.3134
  order_prediction: 0.5
```

---


## `configs/loss/pretrain_basic.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/loss/pretrain_basic.yaml`  
**Kích thước:** `0.2 KB`  
**Loại:** `.yaml`

```yaml
# configs/loss/pretrain_basic.yaml
components:
  reconstruction:
    enabled: true
    weight: 1.0
    
  contrastive:
    enabled: false
    weight: 1.0

best_losses:
  reconstruction: 0.0041
```

---


## `configs/model/cbramod/downstream.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/model/cbramod/downstream.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml

```

---


## `configs/model/cbramod/pretrain.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/model/cbramod/pretrain.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml

```

---


## `configs/model/cbramod/variants.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/model/cbramod/variants.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml

```

---


## `configs/model/eegpt/downstream.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/model/eegpt/downstream.yaml`  
**Kích thước:** `0.2 KB`  
**Loại:** `.yaml`

```yaml
name: eegpt_downstream
pretrained_ckpt: /home/infres/ttran-25/eegfm/outputs/pretraining_lmdb/eegpt/2026-05-12/15-36-15/checkpoints/best-epoch=23-valid_loss=1.2120.ckpt
```

---


## `configs/model/eegpt/pretrain.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/model/eegpt/pretrain.yaml`  
**Kích thước:** `0.6 KB`  
**Loại:** `.yaml`

```yaml
# configs/model/eegpt/pretrain.yaml
name: eegpt

# Chọn variant
variant: large

# Data shape
img_size: [19, 3200]
patch_size: 64

# Các thông số khác sẽ được lấy từ variant
# Có thể override ở đây nếu cần
# embed_dim: 512  # uncomment để override

# Các thông số này khi xử lý nên được lấy từ variant, 
# hiện tại đang debug nên tạm thời override
# embed_dim: 512
# embed_num: 4
# encoder_depth: 6
# predictor_depth: 2
# reconstructor_depth: 2
# num_heads: 8
# gpt_num_hidden_layers: 10
# gpt_num_attention_heads: 16
```

---


## `configs/model/eegpt/variants.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/model/eegpt/variants.yaml`  
**Kích thước:** `1.1 KB`  
**Loại:** `.yaml`

```yaml
# configs/model/eegpt/variants.yaml
variants:
  tiny1:
    embed_dim: 64
    embed_num: 1
    encoder_depth: 2
    predictor_depth: 2
    reconstructor_depth: 4
    num_heads: 4

  tiny2:
    embed_dim: 128
    embed_num: 4
    encoder_depth: 4
    predictor_depth: 4
    reconstructor_depth: 4
    num_heads: 4
    
  base1:
    embed_dim: 256
    embed_num: 1
    encoder_depth: 6
    predictor_depth: 6
    reconstructor_depth: 6
    num_heads: 4
    gpt_num_hidden_layers: 6
    
  large:
    embed_dim: 512
    embed_num: 4
    encoder_depth: 8
    predictor_depth: 8
    reconstructor_depth: 8
    num_heads: 8
    gpt_num_hidden_layers: 10
    gpt_num_attention_heads: 16
    
  L_822:
    embed_dim: 512
    embed_num: 4
    encoder_depth: 8
    predictor_depth: 2
    reconstructor_depth: 2
    num_heads: 8

gpt_default:
  embed_dim: 768
  num_hidden_layers_embedding_model: 2
  num_hidden_layers: 4
  num_attention_heads: 12
  intermediate_dim_factor: 4
  hidden_activation: gelu_new
  dropout: 0.1
  num_hidden_layers_unembedding_model: 2

order_default:
  num_layers: 2
  num_heads: 8
```

---


## `configs/module/classification_module.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/module/classification_module.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml
name: classification
```

---


## `configs/module/eegpt_pretrain_module.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/module/eegpt_pretrain_module.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml
name: eegpt_pretrain_module
```

---


## `configs/weighting/mape.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/weighting/mape.yaml`  
**Kích thước:** `0.0 KB`  
**Loại:** `.yaml`

```yaml
name: mape
params:
  eps: 1.0e-8
```

---


## `configs/weighting/static.yaml`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/configs/weighting/static.yaml`  
**Kích thước:** `0.1 KB`  
**Loại:** `.yaml`

```yaml
# configs/weighting/static.yaml
name: static
params:
  weights:
    loss1: 1.0
    loss2: 1.0
```

---


## `script/downstream.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/downstream.sh`  
**Kích thước:** `0.9 KB`  
**Loại:** `.sh`

```bash
#!/bin/bash
#SBATCH --job-name=downstream
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
        --config-name config_downstream \
        dataset.datasets_dir=/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}
done
```

---


## `script/dumpscript.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/dumpscript.sh`  
**Kích thước:** `0.8 KB`  
**Loại:** `.sh`

```bash
#!/bin/bash
#SBATCH --job-name=dumpscript
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=24:00:00          # hh:mm:ss
#SBATCH --nodes=1             # number of nodes
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --partition=P100         # or V100, A100, etc.

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
        --config-name config_dump \
        dataset.datasets_dir=/home/infres/ttran-25/project/datasets/downstream/lmdb_bciciv2a_0_38Hz/LOSO_${SUBJECT}
done
```

---


## `script/dumpscript2.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/dumpscript2.sh`  
**Kích thước:** `0.7 KB`  
**Loại:** `.sh`

```bash
#!/bin/bash
#SBATCH --job-name=dumpscript
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=24:00:00          # hh:mm:ss
#SBATCH --nodes=1             # number of nodes
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --partition=P100         # or V100, A100, etc.

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
srun python -m src.train \
    --config-name config2 \
```

---


## `script/pretrain.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/pretrain.sh`  
**Kích thước:** `0.7 KB`  
**Loại:** `.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=1-00:00:00          # hh:mm:ss
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
srun python -m src.pretrain \
    --config-name config_pretrain
```

---


## `script/pretrain_mape.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/pretrain_mape.sh`  
**Kích thước:** `0.7 KB`  
**Loại:** `.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=24:00:00          # hh:mm:ss
#SBATCH --nodes=1             # number of nodes
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --partition=P100         # or V100, A100, etc.

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
    --config-name config_pretrain_mape \
```

---


## `script/pretrain_static.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/pretrain_static.sh`  
**Kích thước:** `0.7 KB`  
**Loại:** `.sh`

```bash
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
```

---


## `script/pretrain_tuev.sh`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/script/pretrain_tuev.sh`  
**Kích thước:** `0.7 KB`  
**Loại:** `.sh`

```bash
#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --output=bash_logs/%x_%j.out
#SBATCH --error=bash_logs/%x_%j.err
#SBATCH --time=24:00:00          # hh:mm:ss
#SBATCH --nodes=1             # number of nodes
#SBATCH --gres=gpu:1             # number of GPUs
#SBATCH --partition=P100         # or V100, A100, etc.

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
    --config-name config_pretrain
```

---


## `src/pretrain.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/pretrain.py`  
**Kích thước:** `1.6 KB`  
**Loại:** `.py`

```python
import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.core.hydra_config import HydraConfig

from src.module.registry import get_module
import src.module
from src.data.data import EEGDataModule

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    # 0. Call hydra
    hydra_dir = HydraConfig.get().runtime.output_dir
    print(f"Hydra output directory: {hydra_dir}")

    # 1. Initialize Data & Model
    datamodule = EEGDataModule(cfg)
    module_cls = get_module(cfg.module.name)
    model = module_cls(cfg)

    # 2. Logger
    logger = TensorBoardLogger(
        save_dir=hydra_dir, 
        name="tb_logs", 
    )
    print("Logger dir:", logger.log_dir)

    # 3. Callbacks
    ckpt = ModelCheckpoint(
        dirpath=os.path.join(hydra_dir, "checkpoints"),
        monitor="valid_loss",   # dùng đúng metric pretrain
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{valid_loss:.4f}",
        verbose=True
    )
    print("Checkpoint dir:", ckpt.dirpath)

    # 4. Khởi tạo Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        # accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=logger,
        callbacks=[ckpt]
    )

    trainer.fit(model, datamodule=datamodule)

if __name__ == '__main__':
    main()
```

---


## `src/train.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/train.py`  
**Kích thước:** `1.4 KB`  
**Loại:** `.py`

```python
import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.core.hydra_config import HydraConfig

from src.module.registry import get_module
import src.module
from src.data.data import EEGDataModule
from src.utils.callbacks import build_callbacks

@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    # 0. Call hydra
    hydra_dir = HydraConfig.get().runtime.output_dir
    print(f"Hydra output directory: {hydra_dir}")

     # 1. Initialize Data & Model
    datamodule = EEGDataModule(cfg)
    module_cls = get_module(cfg.module.name)
    model = module_cls(cfg)

    # 2. Logger
    logger = TensorBoardLogger(
        save_dir=hydra_dir, 
        name="tb_logs", 
    )
    print("Logger dir:", logger.log_dir)

    # 3. Callbacks
    callbacks, ckpt = build_callbacks(cfg)

    # 4. Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule)

    # 5. Tester
    trainer.test(
        ckpt_path="best",
        datamodule=datamodule
    )

if __name__ == '__main__':
    main()
```

---


## `src/data/__init__.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/data/__init__.py`  
**Kích thước:** `0.1 KB`  
**Loại:** `.py`

```python
from . import synthetic
from . import bciciv2a
from . import pretrain_data
```

---


## `src/data/bciciv2a.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/data/bciciv2a.py`  
**Kích thước:** `1.5 KB`  
**Loại:** `.py`

```python
import torch
from torch.utils.data import Dataset
from .registry import register_dataset
import lmdb
import pickle
import numpy as np
import torch.nn.functional as F

_lmdb_envs: dict = {}  # module-level cache

@register_dataset("bciciv2a")
class BCICIV2aDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        db_path = cfg.datasets_dir

        if db_path not in _lmdb_envs:
            _lmdb_envs[db_path] = lmdb.open(
                db_path,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False
            )
        self.db = _lmdb_envs[db_path]

        with self.db.begin(write=False) as txn:
            all_keys = pickle.loads(txn.get(b"__keys__"))

        self.keys = all_keys[split]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]

        with self.db.begin(write=False) as txn:
            raw = txn.get(key.encode())

        data = pickle.loads(raw)

        x = torch.tensor(data["sample"], dtype=torch.float32)   # (22,1000)
        y = torch.tensor(data["label"], dtype=torch.long)

        if self.cfg.time_points != 1000:
            x = F.interpolate(
                x.unsqueeze(0),
                size=self.cfg.time_points,
                mode="linear",
                align_corners=False
            ).squeeze(0)

        x = x / 100.0

        return x, y
```

---


## `src/data/data.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/data/data.py`  
**Kích thước:** `1.2 KB`  
**Loại:** `.py`

```python
from torch.utils.data import DataLoader
import lightning as L
from src.data.registry import get_dataset

class EEGDataModule(L.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        ds_cls = get_dataset(self.cfg.dataset.name)

        self.train_ds = ds_cls(self.cfg.dataset, split="train")
        self.val_ds = ds_cls(self.cfg.dataset, split="val")
        self.test_ds  = ds_cls(self.cfg.dataset, split="test")
    
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.data.batch_size,
            shuffle=True,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
            persistent_workers=True
        )
```

---


## `src/data/pretrain_data.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/data/pretrain_data.py`  
**Kích thước:** `1.3 KB`  
**Loại:** `.py`

```python
import torch
from torch.utils.data import Dataset
from .registry import register_dataset
import pickle
import os
import lmdb

_lmdb_envs: dict = {}  # module-level cache

@register_dataset("pretraining_lmdb")
class PretrainingLMDBDataset(Dataset):
    def __init__(self, cfg, split="train"):
        self.cfg = cfg
        self.split = split
        self.datasets_dir = cfg.datasets_dir
        
        if split == "train":
            self.datasets_dir = os.path.join(self.datasets_dir, "train")
        else:
            self.datasets_dir = os.path.join(self.datasets_dir, "val")

        if self.datasets_dir not in _lmdb_envs:
            _lmdb_envs[self.datasets_dir] = lmdb.open(
                str(self.datasets_dir),
                readonly=True,
                lock=False,
                readahead=True,
                meminit=False,
            )
        self.db = _lmdb_envs[self.datasets_dir]

        self.scale_div = cfg.scale_div
        with self.db.begin(write=False) as txn:
            self.keys = pickle.loads(txn.get(b"__keys__"))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with self.db.begin(write=False) as txn:
            x = pickle.loads(txn.get(self.keys[idx].encode())) / self.scale_div
        return torch.tensor(x, dtype=torch.float32), torch.tensor(0) # this is fake label
```

---


## `src/data/registry.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/data/registry.py`  
**Kích thước:** `0.3 KB`  
**Loại:** `.py`

```python
DATASET_REGISTRY = {}

def register_dataset(name):
    def wrapper(cls):
        DATASET_REGISTRY[name] = cls
        return cls
    return wrapper


def get_dataset(name):
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {name} not registered. Available: {list(DATASET_REGISTRY.keys())}")
    return DATASET_REGISTRY[name]
```

---


## `src/data/synthetic.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/data/synthetic.py`  
**Kích thước:** `0.6 KB`  
**Loại:** `.py`

```python
# src/data/synthetic.py
import torch
from torch.utils.data import Dataset
from .registry import register_dataset


@register_dataset("synthetic")
class SyntheticEEGDataset(Dataset):
    def __init__(self, cfg, split="train"):
        c = cfg
        if split == "train":
            n = c.train_samples
        elif split == "val":
            n = c.val_samples
        else:
            n = c.test_samples

        self.x = torch.randn(n, c.num_channels, c.seq_len)
        self.y = torch.randint(0, c.num_classes, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

---


## `src/models/__init__.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/__init__.py`  
**Kích thước:** `0.0 KB`  
**Loại:** `.py`

```python
from .eegpt import *
from .cbramod import *
```

---


## `src/models/base_model.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/base_model.py`  
**Kích thước:** `0.3 KB`  
**Loại:** `.py`

```python
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def get_param_groups(self):
        """
        Default:
        all params use one LR
        """
        return self.parameters()
```

---


## `src/models/registry.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/registry.py`  
**Kích thước:** `0.3 KB`  
**Loại:** `.py`

```python
MODEL_REGISTRY = {}

def register_model(name):
    def wrapper(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return wrapper


def get_model(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Model {name} not found. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
```

---


## `src/models/cbramod/__init__.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/cbramod/__init__.py`  
**Kích thước:** `0.0 KB`  
**Loại:** `.py`

```python
from .cbramod import CBraMod
```

---


## `src/models/cbramod/cbramod.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/cbramod/cbramod.py`  
**Kích thước:** `1.8 KB`  
**Loại:** `.py`

```python
from src.models.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_components import *

@register_model("cbramod")
class CBraMod(nn.Module):

    def __init__(
        self,
        img_size=(19, 30, 200),
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=30,
        n_layer=12,
        nhead=8,
        need_mask=True,
        mask_ratio=0.5,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.need_mask = need_mask
        self.mask_ratio = mask_ratio

        self.patch_embedding = PatchEmbedding(
            in_dim=in_dim,
            out_dim=out_dim,
            d_model=d_model,
            seq_len=seq_len
        )

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            norm_first=True,
            activation=F.gelu,
        )

        self.encoder = TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            enable_nested_tensor=False,
        )

        self.apply(self._init_weights)

    def forward(self, x, return_mask=False):

            mask = None

            if return_mask:
                mask = self._make_mask(x)

            z = self.patch_embedding(x, mask)
            z = self.encoder(z)

            return {
                "latent": z,
                "mask": mask
            }

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def _make_mask(self, x):
        bz, ch, patch, _ = x.shape

        mask = torch.zeros((bz, ch, patch), device=x.device)
        mask = mask.bernoulli_(self.mask_ratio)

        return mask
```

---


## `src/models/cbramod/model_components.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/cbramod/model_components.py`  
**Kích thước:** `10.8 KB`  
**Loại:** `.py`

```python
import copy
from typing import Optional, Any, Union, Callable

import torch
import torch.nn as nn
# import torch.nn.functional as F
import warnings
from torch import Tensor
from torch.nn import functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, enable_nested_tensor=True, mask_check=True):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:

        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerEncoderLayer(nn.Module):
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn_s = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)
        self.self_attn_t = nn.MultiheadAttention(d_model//2, nhead // 2, dropout=dropout,
                                                 bias=bias, batch_first=batch_first,
                                                 **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False) -> Tensor:

        x = src
        x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, is_causal=is_causal)
        x = x + self._ff_block(self.norm2(x))
        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        bz, ch_num, patch_num, patch_size = x.shape
        xs = x[:, :, :, :patch_size // 2]
        xt = x[:, :, :, patch_size // 2:]
        xs = xs.transpose(1, 2).contiguous().view(bz*patch_num, ch_num, patch_size // 2)
        xt = xt.contiguous().view(bz*ch_num, patch_num, patch_size // 2)
        xs = self.self_attn_s(xs, xs, xs,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=False)[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, patch_size//2).transpose(1, 2)
        xt = self.self_attn_t(xt, xt, xt,
                              attn_mask=attn_mask,
                              key_padding_mask=key_padding_mask,
                              need_weights=False)[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, patch_size//2)
        x = torch.concat((xs, xt), dim=3)
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)



def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")

def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_seq_len(
        src: Tensor,
        batch_first: bool
) -> Optional[int]:

    if src.is_nested:
        return None
    else:
        src_size = src.size()
        if len(src_size) == 2:
            # unbatched: S, E
            return src_size[0]
        else:
            # batched: B, S, E if batch_first else S, B, E
            seq_len_pos = 1 if batch_first else 0
            return src_size[seq_len_pos]


def _detect_is_causal_mask(
        mask: Optional[Tensor],
        is_causal: Optional[bool] = None,
        size: Optional[int] = None,
) -> bool:
    """Return whether the given attention mask is causal.

    Warning:
    If ``is_causal`` is not ``None``, its value will be returned as is.  If a
    user supplies an incorrect ``is_causal`` hint,

    ``is_causal=False`` when the mask is in fact a causal attention.mask
       may lead to reduced performance relative to what would be achievable
       with ``is_causal=True``;
    ``is_causal=True`` when the mask is in fact not a causal attention.mask
       may lead to incorrect and unpredictable execution - in some scenarios,
       a causal mask may be applied based on the hint, in other execution
       scenarios the specified mask may be used.  The choice may not appear
       to be deterministic, in that a number of factors like alignment,
       hardware SKU, etc influence the decision whether to use a mask or
       rely on the hint.
    ``size`` if not None, check whether the mask is a causal mask of the provided size
       Otherwise, checks for any causal mask.
    """
    # Prevent type refinement
    make_causal = (is_causal is True)

    if is_causal is None and mask is not None:
        sz = size if size is not None else mask.size(-2)
        causal_comparison = _generate_square_subsequent_mask(
            sz, device=mask.device, dtype=mask.dtype)

        # Do not use `torch.equal` so we handle batched masks by
        # broadcasting the comparison.
        if mask.size() == causal_comparison.size():
            make_causal = bool((mask == causal_comparison).all())
        else:
            make_causal = False

    return make_causal


def _generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),  # torch.device('cpu'),
        dtype: torch.dtype = torch.get_default_dtype(),
) -> Tensor:
    r"""Generate a square causal mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


class PatchEmbedding(nn.Module):
    def __init__(self, in_dim, out_dim, d_model, seq_len):
        super().__init__()
        self.d_model = d_model
        self.positional_encoding = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=(19, 7), stride=(1, 1), padding=(9, 3),
                      groups=d_model),
        )
        self.mask_encoding = nn.Parameter(torch.zeros(in_dim), requires_grad=False)
        # self.mask_encoding = nn.Parameter(torch.randn(in_dim), requires_grad=True)

        self.proj_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1, 49), stride=(1, 25), padding=(0, 24)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),

            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
            nn.GroupNorm(5, 25),
            nn.GELU(),
        )
        self.spectral_proj = nn.Sequential(
            nn.Linear(101, d_model),
            nn.Dropout(0.1),
            # nn.LayerNorm(d_model, eps=1e-5),
        )
        # self.norm1 = nn.LayerNorm(d_model, eps=1e-5)
        # self.norm2 = nn.LayerNorm(d_model, eps=1e-5)
        # self.proj_in = nn.Sequential(
        #     nn.Linear(in_dim, d_model, bias=False),
        # )


    def forward(self, x, mask=None):
        bz, ch_num, patch_num, patch_size = x.shape
        if mask == None:
            mask_x = x
        else:
            mask_x = x.clone()
            mask_x[mask == 1] = self.mask_encoding

        mask_x = mask_x.contiguous().view(bz, 1, ch_num * patch_num, patch_size)
        patch_emb = self.proj_in(mask_x)
        patch_emb = patch_emb.permute(0, 2, 1, 3).contiguous().view(bz, ch_num, patch_num, self.d_model)

        mask_x = mask_x.contiguous().view(bz*ch_num*patch_num, patch_size)
        spectral = torch.fft.rfft(mask_x, dim=-1, norm='forward')
        spectral = torch.abs(spectral).contiguous().view(bz, ch_num, patch_num, 101)
        spectral_emb = self.spectral_proj(spectral)
        # print(patch_emb[5, 5, 5, :])
        # print(spectral_emb[5, 5, 5, :])
        patch_emb = patch_emb + spectral_emb

        positional_embedding = self.positional_encoding(patch_emb.permute(0, 3, 1, 2))
        positional_embedding = positional_embedding.permute(0, 2, 3, 1)

        patch_emb = patch_emb + positional_embedding

        return patch_emb


def _weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
```

---


## `src/models/eegpt/__init__.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/eegpt/__init__.py`  
**Kích thước:** `0.1 KB`  
**Loại:** `.py`

```python
from .eegpt import EEGPTModel
from .eegpt_downstream import *
```

---


## `src/models/eegpt/eegpt.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/eegpt/eegpt.py`  
**Kích thước:** `34.6 KB`  
**Loại:** `.py`

```python
import copy
import random
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch
import math 

from src.models.base_model import BaseModel
from src.models.registry import register_model

# CHANNEL_DICT = {k.upper():v for v,k in enumerate(
#                      [      'FP1', 'FPZ', 'FP2', 
#                         "AF7", 'AF3', 'AF4', "AF8", 
#             'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
#         'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
#             'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
#         'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#              'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
#                       'PO7', "PO5", 'PO3', 'POZ', 'PO4', "PO6", 'PO8', 
#                                'O1', 'OZ', 'O2', ])}

# use_channels_names = [      'FP1', 'FPZ', 'FP2', 
#                                'AF3', 'AF4', 
#             'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 
#         'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8', 
#             'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
#         'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
#              'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8', 
#                       'PO7', 'PO3', 'POZ',  'PO4', 'PO8', 
#                                'O1', 'OZ', 'O2', ]

CHANNEL_DICT = {k.upper():v for v,k in enumerate(
                     [
                "FP1","FP2","F7","F3","FZ","F4","F8","T3","C3","CZ","C4",
                "T4","T5","P3","PZ","P4","T6","O1","O2"
            ])}

use_channels_names = [
    "FP1","FP2","F7","F3","FZ","F4","F8","T3","C3","CZ","C4",
    "T4","T5","P3","PZ","P4","T6","O1","O2"
]

################################# Utils ######################################

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def apply_mask(mask, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, C, D = x.shape
    if len(mask.shape)==2:
        mN, mC = mask.shape
        
        mask_keep = mask.reshape((1,mN*mC,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
        masked_x = masked_x.contiguous().view((B,mN,mC,D))
    else:
        mN = mask.shape[0]
        
        mask_keep = mask.reshape((1,mN,1)).repeat((B, 1, D))
        masked_x = torch.gather(x.reshape((B, N*C, D)), dim=-2, index=mask_keep)
    return masked_x

def apply_mask_t(mask_t, x):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), C, D (feature-dim)]
    :param mask: tensor [mN, mC] containing indices of patches in [N, C] to keep 
    """    
    B, N, D = x.shape
    mN = mask_t.shape[0]
    
    mask_keep = mask_t.reshape((1,mN,1)).repeat((B, 1, D))
    masked_x = torch.gather(x, dim=1, index=mask_keep)
    return masked_x

def repeat_interleave_batch(x, B, repeat):
    N = len(x) // B
    x = torch.cat([
        torch.cat([x[i*B:(i+1)*B] for _ in range(repeat)], dim=0)
        for i in range(N)
    ], dim=0)
    return x

# helper functions
def exists(val):
    return val is not None

# rotary embedding helper functions
def rotate_half(x):
    # x = rearrange(x, '... (d r) -> ... d r', r = 2)
    x = x.reshape((*x.shape[:-1],x.shape[-1]//2, 2))
    x1, x2 = x.unbind(dim = -1)
    x = torch.stack((-x2, x1), dim = -1)
    # return rearrange(x, '... d r -> ... (d r)')
    return x.flatten(-2)

def apply_rotary_emb(freqs, t, start_index=0, scale=1.):
    """
    Apply rotary positional embeddings to a tensor.

    The rotary embedding rotates each dimension of the input tensor `t`
    based on the corresponding frequency in `freqs`, using cosine and sine functions. 
    This rotation helps the model preserve positional information.

    Parameters:
    - freqs (Tensor): The frequency embeddings (sine and cosine values precomputed).
    - t (Tensor): The input tensor to which the rotary embeddings are applied.
    - start_index (int): Start index where the rotation will begin within the tensor `t`.
    - scale (float): Scaling factor for the rotation applied.

    Returns:
    - Tensor: The tensor `t` after rotary positional embeddings have been applied.
    """

    freqs = freqs.to(t.device)

    rot_dim = freqs.shape[-1]

    end_index = start_index + rot_dim

    assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'

    t_left, t_middle, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]

    # Apply rotary embeddings to the middle segment.
    t_rotated_middle = (t_middle * freqs.cos() * scale) + (rotate_half(t_middle) * freqs.sin() * scale)

    return torch.cat((t_left, t_rotated_middle, t_right), dim=-1)

################################# RoPE Model Begin ######################################
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000, learned_freq=False, interpolate_factor=1.0):
        """
        Rotary Positional Embedding module to encode sequential information into embeddings.
        
        Parameters:
            dim (int): Dimension of the frequency embedding.
            theta (float): A hyperparameter that influences the scale of the frequency embedding.
            learned_freq (bool): Whether the frequencies are learnable parameters.
            interpolate_factor (float): Scaling factor for interpolated positional encoding.
        """
        super().__init__()
        assert interpolate_factor >= 1.0, "Interpolate factor must be >= 1.0"
        
        # Initialize frequency parameters
        self.freqs = nn.Parameter(
            1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim)),
            requires_grad = learned_freq)
        
        self.interpolate_factor = interpolate_factor
        self.cache = {}

    def prepare_freqs(self, num_patches, device='cuda', dtype=torch.float32, offset=0):
        """
        Prepares the frequency embeddings for the given number of patches.
        
        Parameters:
            num_patches (tuple): Tuple specifying the dimensions (C, N) where
                                 C is the channels and N is the number of positions.
            device (str): Device to store the frequencies on (e.g., 'cuda' or 'cpu').
            dtype (torch.dtype): Data type for the frequencies.
            offset (float): Offset added to position indexes before scaling.
            
        Returns:
            torch.Tensor: Prepared frequency embeddings with shape [C * N, dim].
        """
        C, N = num_patches
        cache_key = f'freqs:{num_patches}'
        
        # Return cached result if available
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Generate sequence positions and apply offset and scale
        seq_pos = torch.arange(N, device=device, dtype=dtype).repeat_interleave(repeats=C)
        seq_pos = (seq_pos + offset) / self.interpolate_factor
        
        # Compute outer product of positions and frequencies, then expand along the last dimension
        freqs_scaled = torch.outer(seq_pos.type(self.freqs.dtype), self.freqs).repeat_interleave(repeats=2, dim=-1)
        
        # Cache and return the computed frequencies
        self.cache[cache_key] = freqs_scaled
        return freqs_scaled
    
    


################################# EEGPT Model Begin ######################################

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
    
    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features 
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.use_rope = use_rope
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.is_causal = is_causal
        self.return_attention= return_attention

    def forward(self, x, freqs=None):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3,B,nh,t,d
        q, k, v = qkv[0], qkv[1], qkv[2] # B,nh,t,d
        
        if self.use_rope:# RoPE
            q = apply_rotary_emb(freqs, q)
            k = apply_rotary_emb(freqs, k)
        if self.return_attention:
            if self.is_causal:
                attn_mask = torch.ones(q.size(-2), q.size(-2), dtype=torch.bool).tril(diagonal=0)
                attn_maak = torch.zeros(q.size(-2), q.size(-2))
                attn_mask = attn_maak.masked_fill(torch.logical_not(attn_mask), -float('inf'))
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))) + attn_mask, dim=-1)
            else:
                attn_weight = torch.softmax((q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))), dim=-1)
            return attn_weight
        # efficient attention using Flash Attention CUDA kernels
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=self.is_causal)
        x = y.transpose(1, 2).contiguous().view(B, T, C) #(B, nh, T, hs) -> (B, T, hs*nh)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, is_causal=False, use_rope=False, return_attention=False):
        super().__init__()
        
        self.return_attention= return_attention
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, is_causal=is_causal, use_rope=use_rope, return_attention = return_attention)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, freqs=None):
        y = self.attn(self.norm1(x), freqs)
        if self.return_attention: return y
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.proj = nn.Conv2d(1, embed_dim, kernel_size=(1,patch_size), 
                              stride=(1, patch_size if patch_stride is None else patch_stride))
        
    def forward(self, x):
        # x: B,C,T
        x = x.unsqueeze(1)# B, 1, C, T
        x = self.proj(x).transpose(1,3) # B, T, C, D
        return x
class PatchNormEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=(64, 1000), patch_size=16, patch_stride=None, embed_dim=768):
        super().__init__()
        
        assert img_size[1] % patch_size==0
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        
        if patch_stride is None:
            self.num_patches = ((img_size[0]), (img_size[1] // patch_size))
        else:
            self.num_patches = ((img_size[0]), ((img_size[1] - patch_size) // patch_stride + 1))

        self.unfold = torch.nn.Unfold(kernel_size=(1, patch_size), stride = (1, patch_stride if patch_stride is not None else patch_size))

        self.proj = nn.Linear(patch_size, embed_dim)#+2

    def forward(self, x):
        # x: B,C,T
        B,C,T = x.shape
        x = x.unsqueeze(1) # B 1 C T
        
        x = self.unfold(x)
        
        x = x.transpose(-1,-2)
        
        x = x.view(B, C, -1, self.patch_size).contiguous()
        x = x.transpose(1,2)
        
        # m = torch.mean(x, dim=-1).unsqueeze(-1)
        # v = torch.std( x, dim=-1).unsqueeze(-1)
        x = torch.layer_norm(x, (self.patch_size,))
        # x = torch.cat([x,m,v], dim=-1) # B, T, C, P
        # print(x)
        
        x = self.proj(x) # B, T, C, D
        
        return x

class EEGTransformerReconstructor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches,
        patch_size=64,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        embed_dim=768,
        reconstructor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_inp_embed = use_inp_embed
        self.use_pos_embed = use_pos_embed
        self.num_patches = num_patches
        
        if use_inp_embed:
            self.reconstructor_embed = nn.Linear(embed_dim, reconstructor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed           = nn.Parameter(torch.zeros(1, 1, embed_num, reconstructor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token          = nn.Parameter(torch.zeros(1, 1, reconstructor_embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (reconstructor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), reconstructor_embed_dim)
        # --
        self.reconstructor_blocks = nn.ModuleList([
            Block(
                dim=reconstructor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.reconstructor_norm = norm_layer(reconstructor_embed_dim)
        self.reconstructor_proj = nn.Linear(reconstructor_embed_dim, patch_size, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.reconstructor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
    
    def forward(self, x, chan_ids=None, mask_x=None, mask_y=None):
        # conditions: (Nq, D) as qurey for downstream 
        # mask_x/mask_y: (mN, mC) one number index like (n*C+c) in matrix (N,C)
        
        chan_ids = chan_ids.to(x).long()
        
        # -- map from encoder-dim to pedictor-dim
        if self.use_inp_embed:
            x = self.reconstructor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D= x.shape
        # assert mN == N, f"{mN},{N}"
        ############## Mask x ###############
        # -- add channels positional embedding to x
        chan_embed = self.chan_embed(chan_ids).unsqueeze(0) # (1,C) -> (1,1,C,D)
        # -- get freqs for RoPE
        
        if mask_x is not None:
            mask_x       = mask_x.to(x.device)
            mask_x       = torch.floor(mask_x[:,0] / C).long().to(x.device)    # select first as represent
            
            freqs_x      = self.time_embed.prepare_freqs((1, N), x.device, x.dtype)
            freqs_x      = freqs_x.contiguous().view((1,N,self.time_embed_dim))
            freqs_x      = apply_mask_t(mask_x, freqs_x)                                       # 1, mN, 1, D
            freqs_x      = freqs_x.contiguous().view((mask_x.shape[0], 1, self.time_embed_dim)) # mN, D//2
            freqs_x      = freqs_x.repeat((1, eN, 1)).flatten(0,1)
            
        else:
            freqs_x      = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) # NC, time_dim
            
        ############# Mask y ################
        if mask_y is not None:
            mask_y       = mask_y.to(x.device)
            
            # create query mask_token ys
            N_y          = mask_y.shape[0]
            chan_embed   = chan_embed.repeat((1,N,1,1))
            chan_embed   = apply_mask(mask_y, chan_embed)
            
            freqs        = self.time_embed.prepare_freqs((C, N), x.device, x.dtype) # NC, time_dim
            freqs_y      = freqs.contiguous().view((1, N, C, self.time_embed_dim))
            freqs_y      = apply_mask(mask_y, freqs_y)        # 1, mN, mC, D
            freqs_y      = freqs_y.contiguous().view((N_y, self.time_embed_dim))
            
            y = self.mask_token.repeat((B, N_y, 1)) + chan_embed
            
            
            if self.use_pos_embed:
                x        = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
                
            # -- concat query mask_token ys
            x           = x.flatten(1,2) # B N E D -> B NE D
            x           = torch.cat([x,y], dim=1)
            freqs_x     = torch.cat([freqs_x, freqs_y], dim=0).to(x)
            
            
            # -- fwd prop
            for blk in self.reconstructor_blocks:
                x = blk(x, freqs_x) # B, NC, D
                if blk.return_attention==True: return x
            
            
            x = x[:,-N_y:,:]      # B, N_y, D
            
            x = self.reconstructor_norm(x) 
                
            x = self.reconstructor_proj(x)
            
            return x
        
class EEGTransformerPredictor(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        num_patches,
        embed_dim=768,
        embed_num=1,
        use_pos_embed = False,
        use_inp_embed = True,
        use_part_pred = False,
        predictor_embed_dim=384,
        depth=6,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.use_part_pred = use_part_pred
        self.use_pos_embed = use_pos_embed
        self.use_inp_embed = use_inp_embed
        self.num_patches = num_patches
        self.embed_num = embed_num
        
        if use_inp_embed:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        
        if use_pos_embed:
            self.pos_embed   = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
            trunc_normal_(self.pos_embed, std=init_std)
        
        self.mask_token      = nn.Parameter(torch.zeros(1, 1, embed_num, predictor_embed_dim))
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # --
        self.time_embed_dim = (predictor_embed_dim//num_heads)//2
        self.time_embed = RotaryEmbedding(dim=self.time_embed_dim, interpolate_factor=interpolate_factor)
        
        # --
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, is_causal=False, use_rope=True, 
                return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)
        # ------
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, mask_x=None, mask_t=None):
        # conditions: (Nq, D) as qurey for downstream 
        # mask_t: mN one number index like (n*C+c) in matrix (N,1)
        
        # -- map from encoder-dim to pedictor-dim
        if self.use_part_pred:
            inp_x = x
            
        if self.use_inp_embed:
            x = self.predictor_embed(x)

        C, N        = self.num_patches
        B, mN, eN, D    = x.shape
        
        ############## Mask x ###############
        # -- get freqs for RoPE
        freqs = self.time_embed.prepare_freqs((eN, N), x.device, x.dtype) # NC, time_dim
        
        if mask_x is not None:
            mask_x       = mask_x
            mask_x       = torch.floor(mask_x[:,0] / C).long()
            ############# Mask y ################
            if mask_t is None:
                mask_t       = torch.tensor(list(set(list(range(0,N))) - set(mask_x.tolist()))).long()
            # -- concat query mask_token ys
            N_y              = mask_t.shape[0]
            y                = self.mask_token.repeat((B, N_y, 1, 1))
            x                = torch.cat([x,y], dim=1)
            
            # -- masked index of tensor x rearrange to normal index
            mask_id          = torch.concat([mask_x.to(x.device), mask_t.to(x.device)], dim=0)            
            x                = torch.index_select(x, dim=1, index=torch.argsort(mask_id))    
            
        if self.use_pos_embed:
            x                = x + self.pos_embed.repeat((B, x.shape[1], 1, 1)).to(x.device)
            
        B, N, eN, D    = x.shape
        x = x.flatten(1,2)
        
        # -- fwd prop
        for blk in self.predictor_blocks:
            x = blk(x, freqs) # B, NC, D
            if blk.return_attention==True: return x
        
        # -- reshape back
        x = x.reshape((B, N, eN, D))
        
        x = self.predictor_norm(x) 
            
        x = self.predictor_proj(x)
        
        if self.use_part_pred and mask_x is not None:
            cmb_x = torch.index_select(x, dim=1, index=mask_t.to(x.device)) 
            cmb_x = torch.concat([inp_x, cmb_x], dim=1)
            cmb_x = torch.index_select(cmb_x, dim=1, index=torch.argsort(mask_id)) 
            return x, cmb_x
        return x

class EEGTransformer(nn.Module):
    """ EEG Transformer """
    def __init__(
        self,
        img_size=(64,1000),
        patch_size=64,
        patch_stride=None,
        embed_dim=768,
        embed_num=1,
        predictor_embed_dim=384,
        depth=12,
        predictor_depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        patch_module=PatchEmbed,# PatchNormEmbed
        init_std=0.02,
        interpolate_factor = 2.,
        return_attention_layer=-1,
        **kwargs
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim
        self.embed_num = embed_num
        
        self.num_heads = num_heads
        
        # --
        self.patch_embed = patch_module(
            img_size=img_size,
            patch_size=patch_size,
            patch_stride=patch_stride,
            embed_dim=embed_dim)
        self.num_patches = self.patch_embed.num_patches
        # --
        
        self.chan_embed = nn.Embedding(len(CHANNEL_DICT), embed_dim)
        # --
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                is_causal=False, use_rope= False, return_attention=(i+1)==return_attention_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # ------
        self.init_std = init_std
        self.summary_token = nn.Parameter(torch.zeros(1, embed_num, embed_dim))
            
        trunc_normal_(self.summary_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()
        
    def prepare_chan_ids(self, channels):
        chan_ids = []
        for ch in channels:
            ch = ch.upper().strip('.')
            assert ch in CHANNEL_DICT
            chan_ids.append(CHANNEL_DICT[ch])
        return torch.tensor(chan_ids).unsqueeze_(0).long()
    
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, chan_ids=None, mask_x=None, mask_t=None):
        # x.shape B, C, T
        # mask_x.shape mN, mC
        # mask_t.shape mN
        
        # -- patchify x
        x = self.patch_embed(x) #
        B, N, C, D = x.shape
        
        assert N==self.num_patches[1] and C==self.num_patches[0], f"{N}=={self.num_patches[1]} and {C}=={self.num_patches[0]}"
        
        if chan_ids is None:
            chan_ids = torch.arange(0,C)     
        chan_ids = chan_ids.to(x)
        
        # -- add channels positional embedding to x
        x = x + self.chan_embed(chan_ids.long()).unsqueeze(0) # (1,C) -> (1,1,C,D)
        
        if mask_x is not None:
            mask_x = mask_x.to(x.device)
            x = apply_mask(mask_x, x)# B, mN, mC, D
            B, N, C, D = x.shape
            
        
        x = x.flatten(0, 1) # BmN, mC, D
        
        # -- concat summary token
        summary_token = self.summary_token.repeat((x.shape[0], 1, 1))
        x = torch.cat([x,summary_token], dim=1)  # BmN, mC+embed_num, D
        
        # -- fwd prop
        for i, blk in enumerate(self.blocks):
            x = blk(x) # B*N, mC+1, D
            if blk.return_attention==True: return x

        x = x[:, -summary_token.shape[1]:, :]
        
        if self.norm is not None:
            x = self.norm(x) 

        
        x = x.flatten(-2)
        x = x.reshape((B, N, -1))
        # -- reshape back
            
        if mask_t is not None:
            mask_t = mask_t.to(x.device)
            x = apply_mask_t(mask_t, x)# B, mN, D        
        
        x = x.reshape((B, N, self.embed_num, -1))
        
        return x

# =========================================================
# Main Model
# =========================================================

@register_model("eegpt")
class EEGPTModel(BaseModel):
    """
    Forward:
        classification / finetune -> pooled feature

    forward_pretrain:
        latent, predicted_latent, reconstructed_patches
    """

    def __init__(self, model_cfg):
        super().__init__(model_cfg)

        self.encoder = EEGTransformer(**model_cfg['encoder'])
        self.predictor = EEGTransformerPredictor(**model_cfg['predictor'])
        self.reconstructor = EEGTransformerReconstructor(**model_cfg['reconstructor'])

        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
            
        self.chans_id       = self.encoder.prepare_chan_ids(use_channels_names)
        self.USE_LOSS_A = True
        self.USE_LN     = True
        self.USE_SKIP   = True
                
    def make_masks(self, num_patchs, mC_x=12, p_n_y=0.5, p_c_y=0.2):
        
        C, N = num_patchs
        
        while True:
            mask_x = []# mN, mC
            mask_y = []
            mask_y_bx = []
            for i in range(N):
                c_idx = torch.randperm(C) + i*C
                if random.random()>p_n_y:
                    mask_x.append(c_idx[:mC_x])
                    mask_y_bx.append(c_idx[mC_x:])
                else:
                    mask_y.append(c_idx)
            if len(mask_x)==0: continue
            if len(mask_y_bx)==0: continue
            mask_y_bx = torch.cat(mask_y_bx, dim=0)
            mask_y_bx = mask_y_bx[torch.rand(mask_y_bx.shape)<p_c_y]
            if len(mask_y_bx)==0: continue
            break
        
        return torch.stack(mask_x, dim=0), torch.cat(mask_y+[mask_y_bx], dim=0)
    
    def forward_target(self, x, mask_y):
        with torch.no_grad():
            h = self.target_encoder(x, self.chans_id.to(x))
            h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim
            C, N = self.encoder.num_patches
            assert x.shape[-1]%N==0 and x.shape[-2]%C == 0
            block_size_c, block_size_n = x.shape[-2]//C, x.shape[-1]//N
            x = x.view(x.shape[0], C, block_size_c, N, block_size_n)
            x = x.permute(0, 3, 1, 2, 4).contiguous() # B, N, C, bc, bn
            x = x.view(x.shape[0], C, N, block_size_c * block_size_n)
            y = apply_mask(mask_y.to(x.device), x)
            if self.USE_LN:
                y = F.layer_norm(y, (y.size(-1),))
            return h, y

    def forward_context(self, x, mask_x, mask_y):
        z = self.encoder(x, self.chans_id.to(x), mask_x=mask_x)
        z, comb_z = self.predictor(z, mask_x=mask_x)
        if not self.USE_SKIP:
            comb_z = z
        r = self.reconstructor(comb_z, self.chans_id.to(x), mask_y=mask_y)
        return z, r
```

---


## `src/models/eegpt/eegpt_downstream.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/models/eegpt/eegpt_downstream.py`  
**Kích thước:** `2.9 KB`  
**Loại:** `.py`

```python
from src.models.base_model import BaseModel
from src.models.registry import register_model
from .eegpt import *
import torch.nn as nn
import torch

@register_model("eegpt_downstream")
class EEGPTDownstream(BaseModel):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.use_channels_num = cfg.dataset.use_chans_num
        self.use_channels_names = cfg.dataset.use_channels_names
        self.num_classes = cfg.dataset.num_classes
        self.time_points = cfg.dataset.time_points
        self.encoder = encoder = EEGTransformer(
            img_size=[self.use_channels_num, self.time_points],
            patch_size=32*2,
            embed_num=4,
            embed_dim=512,
            depth=8,
            num_heads=8,
            mlp_ratio=4.0,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            init_std=0.02,
            qkv_bias=True, 
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.chans_id = encoder.prepare_chan_ids(self.use_channels_names)

        self.load_pretrained_encoder(cfg)

        self.adapter = nn.Sequential(
            nn.Conv1d(cfg.dataset.num_channels , self.use_channels_num, kernel_size=1)
        )

        self.linear_probe1 = nn.Linear(2048, 16)
        self.linear_probe2 = nn.Linear(16*16, self.num_classes)
        self.drop = torch.nn.Dropout(p=0.50)

    def load_pretrained_encoder(self, cfg):
        path = cfg.model.get("pretrained_ckpt", None)

        if path is None:
            print("No checkpoint provided. Random init encoder.")
            return

        try:
            ckpt = torch.load(path, map_location="cpu")

            self.encoder.load_state_dict(
                ckpt["state_dict"],
                strict=False
            )

            print(f"Loaded checkpoint: {path}")

        except FileNotFoundError:
            print(f"Checkpoint not found: {path}")
            print("Use random initialized encoder.")

        except Exception as e:
            raise RuntimeError(f"Failed loading checkpoint: {e}")

    def forward(self, x):

        feat = self.adapter(x)

        feat = self.encoder(feat, self.chans_id.to(feat))

        feat = feat.flatten(2)
        
        feat = self.linear_probe1(self.drop(feat))

        feat = feat.flatten(1)

        logits = self.linear_probe2(feat)

        return logits

    def get_param_groups(self):

        return [
            {
                "params": self.encoder.parameters(),
                "lr": self.cfg.optimizer.encoder_lr,
                "name": "encoder"
            },
            {
                "params": self.adapter.parameters(),
                "lr": self.cfg.optimizer.adapter_lr,
                "name": "adapter"
            },
            {
                "params": list(self.linear_probe1.parameters()) + list(self.linear_probe2.parameters()),
                "lr": self.cfg.optimizer.head_lr,
                "name": "head"
            }
        ]
```

---


## `src/module/__init__.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/__init__.py`  
**Kích thước:** `0.1 KB`  
**Loại:** `.py`

```python
from .downstream.classification_module import *
from .eegpt.eegpt_pretrain_module import *
```

---


## `src/module/base_module.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/base_module.py`  
**Kích thước:** `2.0 KB`  
**Loại:** `.py`

```python
import lightning as L
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection
from src.utils.metrics import accuracy
from src.utils.optimizers import build_optimizer
from src.utils.schedulers import build_scheduler

class BaseModule(L.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def compute_loss(self, logits, y):
        raise NotImplementedError

    def build_metrics(self):
        return None

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        acc = accuracy(logits, y)

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        # Log learning rates
        opt = self.optimizers()
        for i, group in enumerate(opt.param_groups):
            name = group.get("name", f"group_{i}") # if forget naming, auto name it group
            self.log(f"lr_{name}", group["lr"], prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        acc = accuracy(logits, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.compute_loss(logits, y)

        acc = accuracy(logits, y)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):

        param_groups = self.get_param_groups()
        optimizer = build_optimizer(self.cfg, param_groups)
        scheduler = build_scheduler(self.cfg, optimizer)

        if scheduler is None:
            return optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }

    def get_param_groups(self):
        return self.model.get_param_groups()
```

---


## `src/module/registry.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/registry.py`  
**Kích thước:** `0.4 KB`  
**Loại:** `.py`

```python
MODULE_REGISTRY = {}

def register_module(name):
    def wrapper(cls):
        MODULE_REGISTRY[name] = cls
        return cls
    return wrapper


def get_module(name):
    if name not in MODULE_REGISTRY:
        raise ValueError(
            f"Unknown module {name}. "
            f"Available: {list(MODULE_REGISTRY.keys())}"
        )
    return MODULE_REGISTRY[name]
```

---


## `src/module/cbramod/cbramod_pretrain_module.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/cbramod/cbramod_pretrain_module.py`  
**Kích thước:** `3.1 KB`  
**Loại:** `.py`

```python
import torch
import torch.nn.functional as F

from src.module.base_module import BaseModule
from src.module.registry import register_module
from src.models.registry import get_model
from .task.reconstruction import ReconstructionTask


@register_module("cbramod_pretrain_module")
class CBraModPretrain(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)

        builder = CBraModConfigBuilder()
        model_cfg = builder.build(cfg)

        model_cls = get_model(cfg.model.name)

        self.model = model_cls(**model_cfg)

        self.loss_fn = torch.nn.MSELoss()

        # ===== TASKS =====

        self.tasks = nn.ModuleDict({

            "reconstruction": ReconstructionTask(
                d_model=model_cfg["d_model"],
                out_dim=model_cfg["out_dim"]
            ),

            # future tasks
            # "contrastive": ContrastiveTask(...),
            # "classification": ClassificationTask(...),

        })

        # optional weighting
        self.task_weights = {
            "reconstruction": 1.0,
        }

    # =========================
    # shared step
    # =========================
    def shared_step(self, batch):

        shared_output = self.model(
            batch[0],
            return_mask=True
        )

        total_loss = 0
        loss_dict = {}

        for task_name, task in self.tasks.items():

            task_output = task(shared_output, batch)

            task_loss = task_output["loss"]

            weighted_loss = (
                self.task_weights[task_name]
                * task_loss
            )

            total_loss += weighted_loss

            loss_dict[f"{task_name}_loss"] = task_loss

        loss_dict["loss"] = total_loss

        return loss_dict

    # =========================
    # train
    # =========================
    def training_step(self, batch, batch_idx):

        loss_dict = self.shared_step(batch)

        for k, v in loss_dict.items():

            self.log(
                f"train/{k}",
                v,
                prog_bar=(k == "loss"),
                on_epoch=True,
                on_step=False
            )

        return loss_dict["loss"]

    # =========================
    # VALIDATION (FIXED)
    # =========================
    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)
        loss = loss_dict["loss"]

        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True
        )

        return loss

    # =========================
    # optimizer
    # =========================
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.optim.lr,
            weight_decay=self.cfg.optim.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler
        }
```

---


## `src/module/cbramod/config_builder.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/cbramod/config_builder.py`  
**Kích thước:** `1.5 KB`  
**Loại:** `.py`

```python
# module/cbramod/config_builder.py
import yaml
from omegaconf import OmegaConf

class CBraModConfigBuilder:

    def __init__(self):
        self.variants = self._load_variants()

    def _load_variants(self):
        with open("/home/infres/ttran-25/eegfm/configs/model/cbramod/variants.yaml", "r") as f:
            return yaml.safe_load(f)

    def build(self, cfg):
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

        variant = model_cfg.get("variant", "base")
        base_cfg = self.variants["variants"][variant].copy()

        # merge override
        for k, v in model_cfg.items():
            if v is not None:
                base_cfg[k] = v

        # derive values
        img_size = base_cfg["img_size"]
        patch_size = base_cfg["patch_size"]

        seq_len = img_size[1] // patch_size
        num_patches = (img_size[0], seq_len)

        return {
            # core model config
            "img_size": img_size,
            "patch_size": patch_size,
            "seq_len": seq_len,
            "num_patches": num_patches,

            "in_dim": base_cfg["in_dim"],
            "out_dim": base_cfg["out_dim"],
            "d_model": base_cfg["d_model"],
            "dim_feedforward": base_cfg["dim_feedforward"],
            "n_layer": base_cfg["n_layer"],
            "nhead": base_cfg["nhead"],

            # 👉 model-owned behavior (IMPORTANT FIX)
            "need_mask": base_cfg.get("need_mask", True),
            "mask_ratio": base_cfg.get("mask_ratio", 0.5),
        }
```

---


## `src/module/cbramod/task/base_task.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/cbramod/task/base_task.py`  
**Kích thước:** `0.1 KB`  
**Loại:** `.py`

```python
import torch.nn

class BaseTask(nn.Module):

    def forward(self, shared_output, batch):
        raise NotImplementedError
```

---


## `src/module/cbramod/task/reconstruction.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/cbramod/task/reconstruction.py`  
**Kích thước:** `0.7 KB`  
**Loại:** `.py`

```python
import torch.nn

class ReconstructionTask(BaseTask):

    def __init__(self, d_model, out_dim):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.GELU(),
            nn.Linear(d_model, out_dim)
        )

        self.loss_fn = nn.MSELoss()

    def forward(self, shared_output, batch):

        z = shared_output["latent"]
        mask = shared_output["mask"]

        x = batch[0]

        pred = self.decoder(z)

        if mask is not None:
            pred = pred[mask == 1]
            target = x[mask == 1]
        else:
            target = x

        loss = self.loss_fn(pred, target)

        return {
            "loss": loss,
            "pred": pred
        }
```

---


## `src/module/downstream/classification_module.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/downstream/classification_module.py`  
**Kích thước:** `0.5 KB`  
**Loại:** `.py`

```python
from src.module.base_module import BaseModule
from src.models.registry import get_model
import torch.nn.functional as F
from src.module.registry import register_module

@register_module("classification")
class ClassificationModule(BaseModule):
    def __init__(self, cfg):
        super().__init__(cfg)

        model_cls = get_model(cfg.model.name)
        self.model = model_cls(cfg)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, logits, y):
        return F.cross_entropy(logits, y)
```

---


## `src/module/eegpt/config_builder.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/eegpt/config_builder.py`  
**Kích thước:** `6.4 KB`  
**Loại:** `.py`

```python
# EEGPTConfigBuilder
from pathlib import Path
import yaml
from omegaconf import OmegaConf

class EEGPTConfigBuilder:
    """Build complete EEGPT config from variants + overrides"""
    
    def __init__(self):
        self.variants = self._load_variants()
        
    def _load_variants(self):
        """Load variants from yaml file"""
        variants_path = "/home/infres/ttran-25/eegfm/configs/model/eegpt/variants.yaml"
        with open(variants_path, 'r') as f:
            return yaml.safe_load(f)
    
    def build(self, cfg):
        """
        Build complete model config from:
        - Base variant config
        - Override from cfg.model
        - Calculate derived values
        """
        model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
        
        # Get variant
        variant_name = model_cfg.get('variant', 'L_822')
        variant_cfg = self.variants['variants'][variant_name].copy()
        
        # Merge: cfg.model overrides variant
        for key in variant_cfg:
            if key not in model_cfg or model_cfg[key] is None:
                model_cfg[key] = variant_cfg[key]
        
        # Calculate derived values
        img_size = model_cfg['img_size']
        patch_size = model_cfg['patch_size']
        seq_len = img_size[1] // patch_size
        num_patches = (img_size[0], seq_len)
        
        embed_dim = model_cfg['embed_dim']
        embed_num = model_cfg['embed_num']
        num_heads = model_cfg['num_heads']
        
        # Get defaults
        # gpt_default = self.variants['gpt_default']
        # order_default = self.variants['order_default']
        
        # Build complete config
        complete_cfg = {
            # Common params
            'img_size': img_size,
            'patch_size': patch_size,
            'num_patches': num_patches,
            'embed_dim': embed_dim,
            'embed_num': embed_num,
            'num_heads': num_heads,
            
            # Encoder
            'encoder': {
                'img_size': img_size,
                'patch_size': patch_size,
                'embed_dim': embed_dim,
                'embed_num': embed_num,
                'depth': model_cfg['encoder_depth'],
                'num_heads': num_heads,
                'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
                'qkv_bias': model_cfg.get('qkv_bias', True),
                'drop_rate': model_cfg.get('drop_rate', 0.0),
                'attn_drop_rate': model_cfg.get('attn_drop_rate', 0.0),
                'drop_path_rate': model_cfg.get('drop_path_rate', 0.0),
                'init_std': model_cfg.get('init_std', 0.02),
            },
            
            # Predictor
            'predictor': {
                'num_patches': num_patches,
                'embed_dim': embed_dim,
                'embed_num': embed_num,
                'predictor_embed_dim': model_cfg.get('predictor_embed_dim', embed_dim),
                'depth': model_cfg['predictor_depth'],
                'num_heads': num_heads,
                'use_part_pred': model_cfg.get('use_part_pred', True),
                'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
                'qkv_bias': model_cfg.get('qkv_bias', True),
                'drop_rate': model_cfg.get('drop_rate', 0.0),
                'attn_drop_rate': model_cfg.get('attn_drop_rate', 0.0),
                'drop_path_rate': model_cfg.get('drop_path_rate', 0.0),
                'init_std': model_cfg.get('init_std', 0.02),
            },
            
            # Reconstructor
            'reconstructor': {
                'num_patches': num_patches,
                'patch_size': patch_size,
                'embed_dim': embed_dim,
                'embed_num': embed_num,
                'reconstructor_embed_dim': model_cfg.get('reconstructor_embed_dim', embed_dim),
                'depth': model_cfg['reconstructor_depth'],
                'num_heads': num_heads,
                'mlp_ratio': model_cfg.get('mlp_ratio', 4.0),
                'qkv_bias': model_cfg.get('qkv_bias', True),
                'drop_rate': model_cfg.get('drop_rate', 0.0),
                'attn_drop_rate': model_cfg.get('attn_drop_rate', 0.0),
                'drop_path_rate': model_cfg.get('drop_path_rate', 0.0),
                'init_std': model_cfg.get('init_std', 0.02),
            },
            
            # Projector
            # 'projector': {
            #     'in_dim': embed_dim,
            #     'hidden_dim': embed_dim * 2,
            #     'out_dim': model_cfg.get('projector_out_dim', embed_dim),
            #     'dropout_rate': model_cfg.get('projector_dropout', 0.2),
            # },
            
            # # Contrastive predictor
            # 'contrastive_predictor': {
            #     'in_dim': model_cfg.get('projector_out_dim', embed_dim),
            #     'hidden_dim': model_cfg.get('projector_out_dim', embed_dim) // 2,
            #     'out_dim': model_cfg.get('projector_out_dim', embed_dim),
            #     'dropout_rate': model_cfg.get('contrastive_predictor_dropout', 0.1),
            # },
            
            # # GPT decoder
            # 'gpt_decoder': {
            #     'embed_dim': model_cfg.get('gpt_embed_dim', gpt_default['embed_dim']),
            #     'num_hidden_layers': model_cfg.get('gpt_num_hidden_layers', gpt_default['num_hidden_layers']),
            #     'num_attention_heads': model_cfg.get('gpt_num_attention_heads', gpt_default['num_attention_heads']),
            #     'intermediate_dim_factor': gpt_default['intermediate_dim_factor'],
            #     'hidden_activation': gpt_default['hidden_activation'],
            #     'dropout': gpt_default['dropout'],
            #     'n_positions': seq_len,
            #     'in_dim': embed_dim,
            # },
            
            # # Order classifier
            # 'order_classifier': {
            #     'feature_dim': embed_dim * embed_num,
            #     'num_subsamples': seq_len,
            #     'num_classes': seq_len,
            #     'num_layers': model_cfg.get('order_num_layers', order_default['num_layers']),
            #     'num_heads': model_cfg.get('order_num_heads', order_default['num_heads']),
            #     'hidden_dim': embed_dim,
            # },
            
            # # Pairwise order classifier
            # 'pairwise_order_classifier': {
            #     'feature_dim': embed_dim * embed_num,
            #     'hidden_dim': (embed_dim * embed_num) // 2,
            #     'num_pair': seq_len * 2,
            # },
        }
        
        return complete_cfg
```

---


## `src/module/eegpt/eegpt_pretrain_module.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/eegpt/eegpt_pretrain_module.py`  
**Kích thước:** `6.5 KB`  
**Loại:** `.py`

```python
import torch.nn.functional as F
import torch

from src.module.registry import register_module
from src.module.base_module import BaseModule
from src.models.registry import get_model
from src.weighting.registry import get_weighting

from .config_builder import EEGPTConfigBuilder
from .gradient_monitor import GradientMonitor

@register_module("eegpt_pretrain_module")
class EEGPretrain(BaseModule):

    def __init__(self, cfg):
        super().__init__(cfg)

        # ===== model =====
        config_builder = EEGPTConfigBuilder() # resolve config for model
        model_cfg = config_builder.build(cfg)

        model_cls = get_model(cfg.model.name)
        self.model = model_cls(model_cfg)

        # ===== loss weighting strategy =====
        weighting_cls = get_weighting(cfg.weighting.name)
        self.loss_weighting = weighting_cls(
            # loss_names=["loss1", "loss2", "contrast"],
            loss_names=["loss1", "loss2"],
            **cfg.weighting.get("params", {})
        )

    # ==============================================================
    # Shared step
    # ==============================================================
    def shared_step(self, batch):
        x, _ = batch

        mask_x, mask_y = self.model.make_masks(
            self.model.encoder.num_patches
        )

        h, y = self.model.forward_target(x, mask_y)
        z, r = self.model.forward_context(x, mask_x, mask_y)

        loss_dict = self._compute_raw_losses(h, z, y, r)
        
        return loss_dict

    # ==============================================================
    # Loss
    # ==============================================================
    def _compute_raw_losses(self, h, z, y, r):
        """Compute individual losses WITHOUT weighting"""
        loss1 = F.mse_loss(h, z)
        loss2 = F.mse_loss(y, r)
        # loss_con = self._contrastive_loss(z, h)

        return {
            "loss1": loss1,
            "loss2": loss2,
            # "contrast": loss_con,
        }
    
    # ==============================================================
    # Train / Val
    # ==============================================================
    def training_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)

        # Let weighting strategy handle updates
        self.loss_weighting.on_train_step(loss_dict, batch_idx)

        # Compute final weighted loss
        total_loss = self.loss_weighting.get_weighted_loss(loss_dict)
        loss_dict["loss"] = total_loss

        # Log loss
        self.log_dict(
            {f"train_{k}": v for k, v in loss_dict.items()},
            on_epoch=True, on_step=False, sync_dist=True
        )
        # Log gradient conflicts
        if batch_idx % 50 == 0:
            self.grad_monitor.log_conflict(
                loss_dict["loss1"],
                loss_dict["loss2"]
            )

        # Log current weights
        for k, w in self.loss_weighting.weights.items():
            self.log(f"weight_{k}", w, on_step=False, on_epoch=True)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict = self.shared_step(batch)

        # Let weighting strategy accumulate val stats
        self.loss_weighting.on_validation_step(loss_dict, batch_idx)

        total_loss = self.loss_weighting.get_weighted_loss(loss_dict)
        loss_dict["loss"] = total_loss

        self.log_dict(
            {f"valid_{k}": v for k, v in loss_dict.items()},
            on_epoch=True, on_step=False, sync_dist=True
        )
        return loss_dict["loss"]

    # ==============================================================
    # EMA update
    # ==============================================================
    def on_fit_start(self):
        self.grad_monitor = GradientMonitor(self)

        self.loss_weighting.on_fit_start()

        total_steps = self.trainer.estimated_stepping_batches
        self.momentum_scheduler = (
            0.996 + i * (1.0 - 0.996) / total_steps
            for i in range(total_steps + 1)
        )

    def on_train_batch_end(self, *args):
        with torch.no_grad():
            m = next(self.momentum_scheduler)

            for q, k in zip(self.model.encoder.parameters(),
                            self.model.target_encoder.parameters()):
                k.data.mul_(m).add_((1 - m) * q.detach())

    def on_validation_epoch_end(self):
        """Trigger weight update after validation epoch completes"""
        self.loss_weighting.on_validation_epoch_end()

    # ==============================================================
    # Optimizer
    # ==============================================================
    def configure_optimizers(self):

        def is_decay(n, p):
            return not (("bias" in n) or (len(p.shape) == 1))

        decay, no_decay = [], []

        for module in [self.model.encoder,
                       self.model.predictor,
                       self.model.reconstructor]:

            for n, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (decay if is_decay(n, p) else no_decay).append(p)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-2},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=6e-5
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=5e-4,
            steps_per_epoch=len(self.trainer.datamodule.train_dataloader()),
            epochs=self.cfg.trainer.max_epochs,
            pct_start=0.2,
            div_factor=2,
            final_div_factor=8,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    # ==============================================================
    # Contrastive
    # =============================================================
    # def _contrastive_loss(self, z, h):

    #     def _pool_tokens(x):
    #         if x.dim() == 4:
    #             x = x.squeeze(2)
    #         return x.mean(dim=1)

    #     z = _pool_tokens(z)
    #     h = _pool_tokens(h)

    #     z = F.normalize(z, dim=-1)
    #     h = F.normalize(h, dim=-1)

    #     logits = torch.matmul(z, h.T) / self.temperature
    #     labels = torch.arange(z.size(0), device=z.device)

    #     return 0.5 * (
    #         F.cross_entropy(logits, labels) +
    #         F.cross_entropy(logits.T, labels)
    #     )
```

---


## `src/module/eegpt/gradient_monitor.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/module/eegpt/gradient_monitor.py`  
**Kích thước:** `1.6 KB`  
**Loại:** `.py`

```python
# src/analysis/gradient_monitor.py

import torch

class GradientMonitor:

    def __init__(self, pl_module):
        self.model = pl_module.model
        self.pl_module = pl_module

    def collect_modules(self):
        modules = {}

        for i, blk in enumerate(self.model.encoder.blocks):
            modules[f"enc.block.{i}"] = blk.attn
            modules[f"enc.mlp.{i}"] = blk.mlp

        return modules

    def get_grads(self, loss, modules):
        params = []
        names = []

        for name, m in modules.items():
            for p in m.parameters():
                if p.requires_grad:
                    params.append(p)
                    names.append(name)

        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True
        )

        out = {}
        for name, g in zip(names, grads):
            if g is None:
                continue
            out.setdefault(name, []).append(g.detach().flatten())

        return {k: torch.cat(v) for k, v in out.items()}

    def log_conflict(self, loss_a, loss_b, step=True):
        modules = self.collect_modules()

        ga = self.get_grads(loss_a, modules)
        gb = self.get_grads(loss_b, modules)

        for name in modules:
            if name not in ga or name not in gb:
                continue

            cos = torch.dot(ga[name], gb[name]) / (
                torch.norm(ga[name]) * torch.norm(gb[name]) + 1e-8
            )

            self.pl_module.log(
                f"conflict/{name}/cos",
                cos,
                on_step=True
            )
```

---


## `src/utils/callbacks.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/utils/callbacks.py`  
**Kích thước:** `0.4 KB`  
**Loại:** `.py`

```python
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

def build_callbacks(cfg):
    ckpt = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best-{epoch}-{val_acc:.4f}"
    )

    early_stop = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=10
    )

    return [ckpt, early_stop], ckpt
```

---


## `src/utils/metrics.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/utils/metrics.py`  
**Kích thước:** `0.1 KB`  
**Loại:** `.py`

```python
import torch

def accuracy(logits, y):
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean()
```

---


## `src/utils/optimizers.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/utils/optimizers.py`  
**Kích thước:** `0.6 KB`  
**Loại:** `.py`

```python
import torch

def build_optimizer(cfg, params):

    name = cfg.optimizer.get("name", "adamw")
    lr = cfg.optimizer.get("lr", 1e-3)
    wd = cfg.optimizer.get("weight_decay", 0.0)

    # nếu là iterator parameters()
    if not isinstance(params, (list, tuple)):
        params = [{"params": params, "lr": lr}]

    if name == "adam":
        return torch.optim.Adam(params)

    if name == "adamw":
        return torch.optim.AdamW(params, weight_decay=wd)

    if name == "sgd":
        return torch.optim.SGD(params, momentum=0.9, weight_decay=wd)

    raise ValueError(f"Unknown optimizer: {name}")
```

---


## `src/utils/schedulers.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/utils/schedulers.py`  
**Kích thước:** `0.5 KB`  
**Loại:** `.py`

```python
import torch

def build_scheduler(cfg, optimizer):
    name = cfg.scheduler.get("name", None)

    if name is None:
        return None

    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.trainer.max_epochs
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=10,
            gamma=0.5
        )

    raise ValueError(f"Unknown scheduler: {name}")
```

---


## `src/visualizer/main_visualizer.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/visualizer/main_visualizer.py`  
**Kích thước:** `0.6 KB`  
**Loại:** `.py`

```python
import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.module.registry import get_module
from src.visualizer.model_visualizer import print_model_tree
import src.module
from src.data.data import EEGDataModule
import os
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path='../../configs', config_name='config_pretrain_mape')
def main(cfg: DictConfig):

    module_cls = get_module(cfg.module.name)
    model = module_cls(cfg)
    print_model_tree(model)

if __name__ == '__main__':
    main()
```

---


## `src/visualizer/model_visualizer.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/visualizer/model_visualizer.py`  
**Kích thước:** `0.4 KB`  
**Loại:** `.py`

```python
import torch.nn as nn

def print_model_tree(model: nn.Module, indent: int = 0):
    """
    Pretty print model architecture as a tree (like torch summary)
    """
    space = "  " * indent

    for name, module in model.named_children():
        print(f"{space}({name}): {module.__class__.__name__}")

        # nếu module có children → recurse
        if len(list(module.children())) > 0:
            print_model_tree(module, indent + 1)
```

---


## `src/weighting/__init__.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/weighting/__init__.py`  
**Kích thước:** `0.1 KB`  
**Loại:** `.py`

```python
from src.weighting.static_weighting import StaticWeighting
from src.weighting.mape_weighting import MAPEWeighting
```

---


## `src/weighting/base_weighting.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/weighting/base_weighting.py`  
**Kích thước:** `1.1 KB`  
**Loại:** `.py`

```python
# src/weighting/base_weighting.py

from abc import ABC, abstractmethod
from typing import Dict
import torch


class BaseLossWeighting(ABC):
    """
    Base class for all loss weighting strategies
    """
    
    def __init__(self, loss_names, **kwargs):
        self.loss_names = loss_names
        self.weights = {k: 1.0 / len(loss_names) for k in loss_names}
    
    @abstractmethod
    def on_train_step(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int):
        """Called after each training step"""
        pass
    
    @abstractmethod
    def on_validation_step(self, loss_dict: Dict[str, torch.Tensor], batch_idx: int):
        """Called after each validation step"""
        pass
    
    @abstractmethod
    def on_fit_start(self):
        """Called when training starts"""
        pass

    @abstractmethod
    def on_validation_epoch_end(self):
        """Called after validation epoch ends"""
        pass
    
    def get_weighted_loss(self, loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute weighted sum of losses"""
        total = sum(self.weights[k] * loss_dict[k] for k in self.loss_names)
        return total
```

---


## `src/weighting/mape_weighting.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/weighting/mape_weighting.py`  
**Kích thước:** `3.8 KB`  
**Loại:** `.py`

```python
# src/weighting/mape_weighting.py

import torch
from src.weighting.base_weighting import BaseLossWeighting
from src.weighting.registry import register_weighting

@register_weighting("mape")
class MAPEWeighting(BaseLossWeighting):
    """
    MAPE-based adaptive loss weighting
    """

    def __init__(self, loss_names, eps=1e-8, **kwargs):
        super().__init__(loss_names)
        
        self.eps = eps

        # statistics
        self.init_train = {}
        self.init_val = {}
        self.train_buf = {k: [] for k in loss_names}
        self.val_buf = {k: [] for k in loss_names}

        # control
        self.global_step = 0
        self.update_step = 0
        self._initialized = False

    def on_fit_start(self):
        """Reset state at start of training"""
        self._initialized = False
        self.global_step = 0

    def on_train_step(self, loss_dict, batch_idx):
        """Update statistics and initialize if needed"""
        # Initialize on first step
        if not self._initialized:
            self._set_initial_losses(loss_dict, is_train=True)
            self._initialized = True
        
        # Log losses
        self.global_step += 1
        for k in self.loss_names:
            self.train_buf[k].append(loss_dict[k].detach())

    def on_validation_step(self, loss_dict, batch_idx):
        """Accumulate validation losses"""
        # Initialize validation baseline if needed
        if not self.init_val:
            self._set_initial_losses(loss_dict, is_train=False)
        
        for k in self.loss_names:
            self.val_buf[k].append(loss_dict[k].detach())

    def on_validation_epoch_end(self):
        """Update weights after validation epoch"""
        if len(self.val_buf[self.loss_names[0]]) > 0:  # Check if we have validation data
            self._update_weights()

    def _set_initial_losses(self, loss_dict, is_train=True):
        """Initialize L_i(0) for train or validation"""
        target_dict = self.init_train if is_train else self.init_val
        for k in self.loss_names:
            target_dict[k] = loss_dict[k].detach()

    def _should_update(self):
        """Check if we have enough data to update"""
        # Check both train and val buffers have data
        has_train = len(self.train_buf[self.loss_names[0]]) > 0
        has_val = len(self.val_buf[self.loss_names[0]]) > 0
        return has_train and has_val

    def _compute_statistics(self):
        """Compute overfitting (O) and generalization (G) metrics"""
        O, G = {}, {}

        # compute for each loss
        for k in self.loss_names:

            Lt_raw = torch.stack(self.train_buf[k]).mean()
            Lv_raw = torch.stack(self.val_buf[k]).mean()
            # normalize
            Lt = (Lt_raw) / (self.init_train[k] + self.eps)
            Lv = (Lv_raw) / (self.init_val[k] + self.eps)

            O[k] = (1 - Lt) - (1 - Lv)
            G[k] = (1 - Lv)

        return O, G

    def _update_weights(self):
        """MAPE weight update rule"""
        if not self._should_update():
            return
        
        O, G = self._compute_statistics()
        O_vec = torch.stack([O[k] for k in self.loss_names])
        G_vec = torch.stack([G[k] for k in self.loss_names])

        A = torch.outer(O_vec, O_vec) + 0.1 * torch.eye(len(O_vec), device=O_vec.device)
        w_vec = 0.5 * torch.linalg.solve(A, G_vec)

        # Ensure non-negative weights
        w_vec = torch.clamp(w_vec, min=0.0)

        # Normalize
        total = torch.sum(w_vec)
        new_w = {}
        for i, k in enumerate(self.loss_names):
            new_w[k] = (w_vec[i] / (total + self.eps)).item()
        self.weights = new_w

        # Reset buffers (keep initial losses for next epoch)
        self.train_buf = {k: [] for k in self.loss_names}
        self.val_buf = {k: [] for k in self.loss_names}
        self.update_step += 1
```

---


## `src/weighting/registry.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/weighting/registry.py`  
**Kích thước:** `0.3 KB`  
**Loại:** `.py`

```python
# src/weighting/registry.py

_WEIGHTING_REGISTRY = {}


def register_weighting(name):
    def decorator(cls):
        _WEIGHTING_REGISTRY[name] = cls
        return cls
    return decorator


def get_weighting(name):
    if name not in _WEIGHTING_REGISTRY:
        raise ValueError(f"Unknown weighting: {name}")
    return _WEIGHTING_REGISTRY[name]
```

---


## `src/weighting/static_weighting.py`

**Đường dẫn đầy đủ:** `/home/infres/ttran-25/eegfm/src/weighting/static_weighting.py`  
**Kích thước:** `0.7 KB`  
**Loại:** `.py`

```python
# src/weighting/static_weighting.py

from src.weighting.base_weighting import BaseLossWeighting
from src.weighting.registry import register_weighting

@register_weighting("static")
class StaticWeighting(BaseLossWeighting):
    """
    Fixed weights - no adaptation
    """
    
    def __init__(self, loss_names, weights=None, **kwargs):
        super().__init__(loss_names)
        
        if weights is not None:
            assert set(weights.keys()) == set(loss_names)
            self.weights = weights
    
    def on_train_step(self, loss_dict, batch_idx):
        pass
    
    def on_validation_step(self, loss_dict, batch_idx):
        pass
    
    def on_fit_start(self):
        pass
    
    def on_validation_epoch_end(self):
        pass
```

---
