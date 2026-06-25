import os
import pickle
import random
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.registry import register_dataset


# -----------------------------------------------
# Dataset base
# -----------------------------------------------

class TUABDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        scale_div: float = 100.0,
        transform=None,
    ):
        super().__init__()
        assert len(files) == len(labels)
        self.files = files
        self.labels = labels
        self.scale_div = float(scale_div)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        with open(self.files[idx], "rb") as fh:
            pair = pickle.load(fh)
        x = torch.from_numpy(pair["X"].astype(np.float32))  # [C, T]
        if self.transform is not None:
            x = self.transform(x)
        x = (x / self.scale_div).contiguous()
        y = torch.tensor(int(pair["y"]), dtype=torch.long)
        return x, y

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys


# -----------------------------------------------
# Split dataset
# -----------------------------------------------

@register_dataset("tuab")
class TUABDatasetSplit(TUABDataset):
    _split_cache: dict = {}

    def __init__(self, cfg, split: str):
        cache_key = _make_cache_key(cfg)
        if cache_key not in TUABDatasetSplit._split_cache:
            splits = _build_splits(cfg)
            TUABDatasetSplit._split_cache[cache_key] = splits

        splits = TUABDatasetSplit._split_cache[cache_key]
        if split not in splits:
            raise KeyError(
                f"Split '{split}' not available. "
                f"Available: {list(splits.keys())}"
            )
        files_s, labels_s = splits[split]
        super().__init__(
            files=files_s,
            labels=labels_s,
            scale_div=getattr(cfg, "scale_div", 100.0),
            transform=getattr(cfg, "transform", None),
        )


# -----------------------------------------------
# Cache key
# -----------------------------------------------

def _make_cache_key(cfg) -> tuple:
    return (
        cfg.data_dir,
        tuple(sorted(_to_str_list(getattr(cfg, "val_subject_ids", None)))),
        int(getattr(cfg, "seed", 7)),
    )


# -----------------------------------------------
# Split builder
# -----------------------------------------------

def _build_splits(cfg) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Scan train/ và eval/ riêng, sau đó tách train -> train+val
    dựa trên val_subject_ids.
    """
    data_dir = cfg.data_dir
    seed     = int(getattr(cfg, "seed", 7))
    val_ids  = _to_str_list(getattr(cfg, "val_subject_ids", None))

    # test split: scan eval/
    te_files, te_labels, _ = _scan_tuab(data_dir, "eval")

    # train+val split: scan train/
    tr_files, tr_labels, tr_subj_ids = _scan_tuab(data_dir, "train")
    uniq_train = sorted(set(tr_subj_ids))

    if not val_ids:
        # Không có val → trả về train + test
        return {
            "train": (tr_files, tr_labels),
            "test":  (te_files, te_labels),
        }

    missing = set(val_ids) - set(uniq_train)
    if missing:
        raise ValueError(f"Validation subjects not found in train/: {sorted(missing)}")

    va_files, va_labels = _subset_by_subject(tr_files, tr_labels, tr_subj_ids, val_ids)
    tr_files, tr_labels = _subset_excluding(tr_files, tr_labels, tr_subj_ids, val_ids)

    if not tr_files:
        raise ValueError("No subjects remain for train after validation split.")

    return {
        "train": (tr_files, tr_labels),
        "val":   (va_files, va_labels),
        "test":  (te_files, te_labels),
    }


# -----------------------------------------------
# File scanning
# -----------------------------------------------

_SUB_RE = re.compile(r"^(.*?)_s\d+_", re.IGNORECASE)


def _scan_tuab(root: str, partition: str) -> Tuple[List[str], List[int], List[str]]:
    """
    Scan <root>/<partition>/{0,1}/*.pkl.
    Subject ID là string prefix trước '_s###_', ví dụ:
    'aaaaagvr_s005_t000_67.pkl' -> 'aaaaagvr'
    """
    files, labels, subj_ids = [], [], []
    found_any = False

    for cls in ("0", "1"):
        cls_dir = os.path.join(root, partition, cls)
        if not os.path.isdir(cls_dir):
            continue
        found_any = True
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".pkl"):
                continue
            m = _SUB_RE.match(fname)
            if not m:
                raise ValueError(
                    f"Cannot parse subject prefix from filename: {fname} "
                    f"(expected 'PREFIX_s###_*.pkl')"
                )
            files.append(os.path.join(cls_dir, fname))
            labels.append(int(cls))
            subj_ids.append(m.group(1))

    if not found_any:
        raise FileNotFoundError(
            f"No class dirs '0' or '1' found under {os.path.join(root, partition)}"
        )
    if not files:
        raise FileNotFoundError(
            f"No .pkl files found under {os.path.join(root, partition)}/{{0,1}}"
        )
    return files, labels, subj_ids


# -----------------------------------------------
# Subset helpers
# -----------------------------------------------

def _subset_by_subject(
    files: Sequence[str],
    labels: Sequence[int],
    subj_ids: Sequence[str],
    chosen,
) -> Tuple[List[str], List[int]]:
    chosen_set = set(chosen)
    pairs = [(fp, lb) for fp, lb, sid in zip(files, labels, subj_ids) if sid in chosen_set]
    if not pairs:
        return [], []
    f, y = zip(*pairs)
    return list(f), list(y)


def _subset_excluding(
    files: Sequence[str],
    labels: Sequence[int],
    subj_ids: Sequence[str],
    excluded,
) -> Tuple[List[str], List[int]]:
    excl = set(excluded)
    pairs = [(fp, lb) for fp, lb, sid in zip(files, labels, subj_ids) if sid not in excl]
    if not pairs:
        return [], []
    f, y = zip(*pairs)
    return list(f), list(y)


# -----------------------------------------------
# OmegaConf-safe helpers
# -----------------------------------------------

def _to_str_list(val) -> List[str]:
    """Normalize int/str/list/ListConfig -> List[str]."""
    if val is None:
        return []
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return [str(v) for v in val]
    return [str(val)]