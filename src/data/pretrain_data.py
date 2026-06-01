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