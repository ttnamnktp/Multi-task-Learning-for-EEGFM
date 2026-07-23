import pickle
from typing import Dict, List, Optional, Tuple
import random
import lmdb
import torch
import json

from torch.utils.data import Dataset
from src.data.registry import register_dataset
# from src.data.utils import build_subjectwise_split


class PhysioMIDataset(Dataset):
    # Cache env theo data_dir, tránh mở lại nhiều lần trong cùng process
    _env_cache: Dict[str, lmdb.Environment] = {}

    def __init__(self, data_dir: str, keys: List[str], scale_div: float = 100.0):
        super().__init__()
        if data_dir not in PhysioMIDataset._env_cache:
            PhysioMIDataset._env_cache[data_dir] = lmdb.open(
                data_dir, readonly=True, lock=False,
                readahead=True, meminit=False, subdir=True,
                max_readers=128,  # cho phép nhiều reader đồng thời
            )
        self.env = PhysioMIDataset._env_cache[data_dir]
        self.keys = keys
        self.scale_div = float(scale_div)

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int):
        k = self.keys[idx]
        with self.env.begin(write=False) as txn:
            pair = pickle.loads(txn.get(k.encode()))
        x = pair["sample"] / self.scale_div
        y = pair["label"]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys


@register_dataset("physiomi")
class PhysioMIDatasetSplit(PhysioMIDataset):
    _split_cache: dict = {}

    def __init__(self, cfg, split: str):
        cache_key = (
            cfg.data_dir,
            getattr(cfg, "seed", 7),
            getattr(cfg, "test_fold_index", 4),
            getattr(cfg, "cv_fold_index", 0),
            getattr(cfg, "folds_json", None),
        )

        if cache_key not in PhysioMIDatasetSplit._split_cache:
            train_keys, val_keys, test_keys = build_subjectwise_split(
                data_dir=cfg.data_dir,
                seed=getattr(cfg, "seed", 7),
                test_fold_index=getattr(cfg, "test_fold_index", 4),
                cv_fold_index=getattr(cfg, "cv_fold_index", 0),
                folds_json=getattr(cfg, "folds_json", None),
            )
            PhysioMIDatasetSplit._split_cache[cache_key] = {
                "train": train_keys,
                "val":   val_keys,
                "test":  test_keys,
            }

        keys = PhysioMIDatasetSplit._split_cache[cache_key][split]
        super().__init__(
            data_dir=cfg.data_dir,
            keys=keys,
            scale_div=getattr(cfg, "scale_div", 100.0),
        )

# -----------------------------------------------
# Helper functions
# -----------------------------------------------

import json
import pickle
import random
from typing import Dict, List, Optional, Tuple



def _read_all_keys(lmdb_dir: str) -> List[str]:
    env = lmdb.open(lmdb_dir, readonly=True, lock=False,
                    readahead=True, meminit=False, subdir=True)
    try:
        with env.begin(write=False) as txn:
            keys_dict = pickle.loads(txn.get(b"__keys__"))
    finally:
        env.close()

    all_keys: List[str] = []
    for bucket in ("train", "eval", "val", "test"):
        if bucket in keys_dict:
            all_keys.extend(keys_dict[bucket])
    return sorted(set(all_keys))


def _extract_subject(key: str) -> str:
    return key.split("R", 1)[0]


def _subjects_and_buckets(all_keys: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    subj2keys: Dict[str, List[str]] = {}
    for k in all_keys:
        s = _extract_subject(k)
        subj2keys.setdefault(s, []).append(k)
    return sorted(subj2keys.keys()), subj2keys


def _make_5_folds(subjects: List[str], seed: int) -> List[List[str]]:
    rng = random.Random(seed)
    subs = subjects[:]
    rng.shuffle(subs)
    n = len(subs)
    base, rem = divmod(n, 5)
    sizes = [base + (1 if i < rem else 0) for i in range(5)]
    folds, i = [], 0
    for sz in sizes:
        folds.append(subs[i:i + sz])
        i += sz
    return folds


def _folds_from_json(json_path: str) -> Tuple[List[List[str]], Optional[int]]:
    with open(json_path) as f:
        spec = json.load(f)
    folds = [spec["folds"][str(i)] for i in range(5)]
    return folds, spec.get("test_fold_index", None)


def build_subjectwise_split(
    data_dir: str,
    *,
    seed: int = 7,
    test_fold_index: int = 4,
    cv_fold_index: int = 0,
    folds_json: Optional[str] = None,
) -> Tuple[List[str], List[str], List[str]]:
    all_keys = _read_all_keys(data_dir)
    subjects, subj2keys = _subjects_and_buckets(all_keys)

    if folds_json:
        folds, fixed_idx = _folds_from_json(folds_json)
        if fixed_idx is not None:
            test_fold_index = int(fixed_idx)
    else:
        folds = _make_5_folds(subjects, seed=seed)

    non_test_ids = [i for i in range(5) if i != test_fold_index]
    val_fold_id  = non_test_ids[cv_fold_index]
    test_subjects  = set(folds[test_fold_index])
    val_subjects   = set(folds[val_fold_id])
    train_subjects = set(
        s for fid in non_test_ids if fid != val_fold_id
        for s in folds[fid]
    )

    def keys_for(subj_set):
        return [k for s in subj_set for k in subj2keys.get(s, [])]

    return keys_for(train_subjects), keys_for(val_subjects), keys_for(test_subjects)