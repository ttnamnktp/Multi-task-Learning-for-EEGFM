import pickle
from typing import Dict, List, Tuple
import lmdb
import torch
from torch.utils.data import Dataset
from src.data.registry import register_dataset


class KaggleERNDataset(Dataset):
    _env_cache: Dict[str, lmdb.Environment] = {}

    def __init__(self, data_dir: str, keys: List[str], scale_div: float = 100.0):
        super().__init__()
        if data_dir not in KaggleERNDataset._env_cache:
            KaggleERNDataset._env_cache[data_dir] = lmdb.open(
                data_dir, readonly=True, lock=False,
                readahead=True, meminit=False, subdir=True,
                max_readers=128,
            )
        self.env = KaggleERNDataset._env_cache[data_dir]
        self.keys = keys
        self.scale_div = float(scale_div)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        k = self.keys[idx]
        with self.env.begin(write=False) as txn:
            byteflow = txn.get(k.encode())
        if byteflow is None:
            raise ValueError(f"Key '{k}' not found in LMDB")
        pair = pickle.loads(byteflow)
        x = torch.from_numpy(pair["sample"]).float() / self.scale_div
        y = torch.tensor(pair["label"], dtype=torch.long)
        return x, y

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys


@register_dataset("kaggle_ern")
class KaggleERNDatasetSplit(KaggleERNDataset):
    _split_cache: dict = {}

    def __init__(self, cfg, split: str):
        fold_num = getattr(cfg, "fold", 1)

        cache_key = (cfg.data_dir, fold_num)
        if cache_key not in KaggleERNDatasetSplit._split_cache:
            KaggleERNDatasetSplit._split_cache[cache_key] = _read_fold_keys(
                data_dir=cfg.data_dir,
                fold_num=fold_num,
            )

        splits = KaggleERNDatasetSplit._split_cache[cache_key]
        super().__init__(
            data_dir=cfg.data_dir,
            keys=splits[split],
            scale_div=getattr(cfg, "scale_div", 100.0),
        )


# -----------------------------------------------
# Helper
# -----------------------------------------------

def _read_fold_keys(data_dir: str, fold_num: int) -> Dict[str, List[str]]:
    """Đọc train/val/test keys của một fold từ LMDB."""
    env = lmdb.open(data_dir, readonly=True, lock=False,
                    readahead=False, meminit=False, subdir=True)
    try:
        with env.begin(write=False) as txn:
            keys_packed = txn.get(b"__keys__")
            if keys_packed is None:
                raise ValueError("Key '__keys__' not found. Did preprocessing finish?")
            all_folds = pickle.loads(keys_packed)
    finally:
        env.close()

    fold_key = f"fold_{fold_num}"
    if fold_key not in all_folds:
        raise ValueError(
            f"Fold key '{fold_key}' not found. Available: {list(all_folds.keys())}"
        )

    fold_data = all_folds[fold_key]
    return {
        "train": fold_data["train"],
        "val":   fold_data["val"],
        "test":  fold_data["test"],
    }

