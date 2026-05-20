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
