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