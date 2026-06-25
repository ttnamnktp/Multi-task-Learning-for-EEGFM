import os
import re
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.registry import register_dataset


# -----------------------------------------------
# Dataset base
# -----------------------------------------------

class SleepEDFDataset(Dataset):
    def __init__(
        self,
        files: List[str],
        labels: List[int],
        scale_div: float = 1.0,
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
        x = torch.load(self.files[idx], weights_only=True)   # [C, T], float32
        if self.transform is not None:
            x = self.transform(x)
        x = (x / self.scale_div).contiguous()
        y = int(self.labels[idx])
        return x, torch.tensor(y, dtype=torch.long)

    @staticmethod
    def collate(batch):
        xs = torch.stack([b[0] for b in batch])
        ys = torch.stack([b[1] for b in batch])
        return xs, ys


# -----------------------------------------------
# Split dataset
# -----------------------------------------------

@register_dataset("sleepedf")
class SleepEDFDatasetSplit(SleepEDFDataset):
    # cache: (data_dir, split_mode, cache_key_args) -> {"train":..,"val":..,"test":..}
    _split_cache: dict = {}

    def __init__(self, cfg, split: str):
        split_mode = getattr(cfg, "split_mode", "subject_explicit")
        seed       = int(getattr(cfg, "seed", 7))

        cache_key = _make_cache_key(cfg)
        if cache_key not in SleepEDFDatasetSplit._split_cache:
            files, labels, subj_ids = _scan_auto(cfg.data_dir)
            splits = _build_splits(cfg, files, labels, subj_ids, split_mode, seed)
            SleepEDFDatasetSplit._split_cache[cache_key] = splits

        splits = SleepEDFDatasetSplit._split_cache[cache_key]
        if split not in splits:
            raise KeyError(
                f"Split '{split}' not available for split_mode='{split_mode}'. "
                f"Available: {list(splits.keys())}"
            )
        files_s, labels_s = splits[split]
        super().__init__(
            files=files_s,
            labels=labels_s,
            scale_div=getattr(cfg, "scale_div", 1.0),
            transform=getattr(cfg, "transform", None),
        )


# -----------------------------------------------
# Cache key
# -----------------------------------------------

def _make_cache_key(cfg) -> tuple:
    split_mode = getattr(cfg, "split_mode", "subject_explicit")
    base = (cfg.data_dir, split_mode, int(getattr(cfg, "seed", 7)))

    if split_mode == "subject_explicit":
        return base + (
            tuple(sorted(_to_int_list(getattr(cfg, "train_subject_ids", None)))),
            tuple(sorted(_to_int_list(getattr(cfg, "val_subject_ids",   None)))),
            tuple(sorted(_to_int_list(getattr(cfg, "test_subject_ids",  None)))),
        )
    else:
        return base + (
            tuple(sorted(_to_int_list(getattr(cfg, "test_subject_ids", None)))),
            float(getattr(cfg, "val_ratio", 0.1)),
            bool(getattr(cfg, "stratified_val", True)),
        )


def _resolve_test_ids(cfg, uniq: List[int]) -> List[int]:
    te = getattr(cfg, "test_subject_ids", None)
    if te is None:
        raise ValueError("This split mode requires `test_subject_ids`.")
    te_ids = sorted(_to_int_list(te))
    missing = set(te_ids) - set(uniq)
    if missing:
        raise ValueError(f"Test subjects not found: {sorted(missing)}")
    return te_ids


# -----------------------------------------------
# Split builders
# -----------------------------------------------

def _build_splits(
    cfg,
    files: List[str],
    labels: List[int],
    subj_ids: List[int],
    split_mode: str,
    seed: int,
) -> Dict[str, Tuple[List[str], List[int]]]:
    uniq = sorted(set(subj_ids))

    if split_mode == "subject_explicit":
        return _split_subject_explicit(cfg, files, labels, subj_ids, uniq)

    elif split_mode == "mixed":
        return _split_mixed(cfg, files, labels, subj_ids, uniq, seed)

    elif split_mode == "subject_random":
        return _split_subject_random(cfg, files, labels, subj_ids, uniq, seed)

    else:
        raise ValueError(f"Unknown split_mode: {split_mode!r}")


def _split_subject_explicit(cfg, files, labels, subj_ids, uniq):
    tr_set = set(_to_int_list(getattr(cfg, "train_subject_ids", None)))
    va_set = set(_to_int_list(getattr(cfg, "val_subject_ids",   None)))
    te_set = set(_to_int_list(getattr(cfg, "test_subject_ids",  None)))
    
    if not tr_set:
        tr_set = set(uniq) - va_set - te_set

    if (tr_set & va_set) or (tr_set & te_set) or (va_set & te_set):
        raise ValueError("train/val/test subject sets must be disjoint.")
    missing = (tr_set | va_set | te_set) - set(uniq)
    if missing:
        raise ValueError(f"Subjects not found in dataset: {sorted(missing)}")

    out = {
        "train": _subset_by_subject(files, labels, subj_ids, tr_set),
        "test":  _subset_by_subject(files, labels, subj_ids, te_set),
    }
    if va_set:
        out["val"] = _subset_by_subject(files, labels, subj_ids, va_set)
    return out


def _split_mixed(cfg, files, labels, subj_ids, uniq, seed):
    te_ids = _resolve_test_ids(cfg, uniq)
    val_ratio     = float(getattr(cfg, "val_ratio", 0.1))
    stratified    = bool(getattr(cfg, "stratified_val", True))

    te_f, te_y   = _subset_by_subject(files, labels, subj_ids, te_ids)
    pool_f, pool_y = _subset_excluding(files, labels, subj_ids, te_ids)

    if not (0.0 < val_ratio < 1.0):
        return {
            "train": (pool_f, pool_y),
            "test":  (te_f, te_y),
        }

    if stratified:
        idx_tr, idx_va = _stratified_idx_split(pool_y, val_ratio, seed)
    else:
        rng = random.Random(seed)
        idxs = list(range(len(pool_f)))
        rng.shuffle(idxs)
        k = max(1, int(round(len(pool_f) * val_ratio)))
        idx_va, idx_tr = idxs[:k], idxs[k:]

    return {
        "train": ([pool_f[i] for i in idx_tr], [pool_y[i] for i in idx_tr]),
        "val":   ([pool_f[i] for i in idx_va], [pool_y[i] for i in idx_va]),
        "test":  (te_f, te_y),
    }


def _split_subject_random(cfg, files, labels, subj_ids, uniq, seed):
    te_ids    = _resolve_test_ids(cfg, uniq)
    val_ratio = float(getattr(cfg, "val_ratio", 0.1))

    te_f, te_y = _subset_by_subject(files, labels, subj_ids, te_ids)
    pool_subjects = sorted(set(uniq) - set(te_ids))
    if not pool_subjects:
        raise ValueError("No subjects remain for train/val after test split.")

    rng = random.Random(seed)
    rng.shuffle(pool_subjects)
    n_val      = max(1, int(round(len(pool_subjects) * val_ratio)))
    va_subjects = pool_subjects[:n_val]
    tr_subjects = pool_subjects[n_val:]

    return {
        "train": _subset_by_subject(files, labels, subj_ids, tr_subjects),
        "val":   _subset_by_subject(files, labels, subj_ids, va_subjects),
        "test":  (te_f, te_y),
    }


# -----------------------------------------------
# File scanning
# -----------------------------------------------

_SUB_RE = re.compile(r"^s(\d+)_", re.IGNORECASE)


def _scan_auto(root: str) -> Tuple[List[str], List[int], List[int]]:
    if any(os.path.isdir(os.path.join(root, str(c))) for c in range(5)):
        return _scan_nosplit(root)
    return _scan_fold_layout(root)


def _scan_nosplit(root: str) -> Tuple[List[str], List[int], List[int]]:
    files, labels, subj_ids = [], [], []
    found_any = False
    for cls in map(str, range(5)):
        cls_dir = os.path.join(root, cls)
        if not os.path.isdir(cls_dir):
            continue
        found_any = True
        for fname in sorted(os.listdir(cls_dir)):
            if not fname.endswith(".pt"):
                continue
            m = _SUB_RE.search(fname)
            if not m:
                raise ValueError(f"Cannot parse subject id from filename: {fname}")
            files.append(os.path.join(cls_dir, fname))
            labels.append(int(cls))
            subj_ids.append(int(m.group(1)))
    if not found_any:
        raise FileNotFoundError(f"No class dirs 0..4 found under {root}")
    if not files:
        raise FileNotFoundError(f"No .pt files found under {root}/{{0..4}}")
    return files, labels, subj_ids


def _scan_fold_layout(root: str) -> Tuple[List[str], List[int], List[int]]:
    files, labels, subj_ids = [], [], []
    found_any = False
    for fold in ("TrainFold", "ValidFold", "TestFold"):
        fold_dir = os.path.join(root, fold)
        if not os.path.isdir(fold_dir):
            continue
        for cls in map(str, range(5)):
            cls_dir = os.path.join(fold_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            found_any = True
            for fname in sorted(os.listdir(cls_dir)):
                if not fname.endswith(".pt"):
                    continue
                m = _SUB_RE.search(fname)
                if not m:
                    raise ValueError(f"Cannot parse subject id from filename: {fname}")
                files.append(os.path.join(cls_dir, fname))
                labels.append(int(cls))
                subj_ids.append(int(m.group(1)))
    if not found_any:
        raise FileNotFoundError(f"No SleepEDF data found under {root}")
    return files, labels, subj_ids


# -----------------------------------------------
# Subset helpers
# -----------------------------------------------

def _subset_by_subject(
    files: Sequence[str],
    labels: Sequence[int],
    subj_ids: Sequence[int],
    chosen,
) -> Tuple[List[str], List[int]]:
    chosen_set = set(int(s) for s in chosen)
    f, y = zip(
        *[(fp, lb) for fp, lb, sid in zip(files, labels, subj_ids) if sid in chosen_set]
    ) if any(sid in chosen_set for sid in subj_ids) else ([], [])
    return list(f), list(y)


def _subset_excluding(
    files: Sequence[str],
    labels: Sequence[int],
    subj_ids: Sequence[int],
    excluded,
) -> Tuple[List[str], List[int]]:
    excl = set(int(s) for s in excluded)
    f, y = zip(
        *[(fp, lb) for fp, lb, sid in zip(files, labels, subj_ids) if sid not in excl]
    ) if any(sid not in excl for sid in subj_ids) else ([], [])
    return list(f), list(y)


def _stratified_idx_split(
    labels: Sequence[int], val_ratio: float, seed: int
) -> Tuple[List[int], List[int]]:
    rng = random.Random(seed)
    by_cls: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_cls.setdefault(int(y), []).append(i)
    tr, va = [], []
    for idxs in by_cls.values():
        rng.shuffle(idxs)
        n = len(idxs)
        k = 0 if n == 1 else min(max(1, int(round(n * val_ratio))), n - 1)
        va.extend(idxs[:k])
        tr.extend(idxs[k:])
    rng.shuffle(tr)
    rng.shuffle(va)
    return tr, va

# -----------------------------------------------
# OmegaConf-safe helper
# -----------------------------------------------

def _to_int_list(val) -> List[int]:
    """Normalize bất kỳ dạng nào (int, list, ListConfig, ...) thành List[int]."""
    if val is None:
        return []
    # OmegaConf ListConfig hoặc list/tuple thông thường
    if hasattr(val, "__iter__") and not isinstance(val, (str, bytes)):
        return [int(v) for v in val]
    # scalar
    return [int(val)]