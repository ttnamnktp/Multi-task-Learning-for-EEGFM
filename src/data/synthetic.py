# # src/data/synthetic.py
# import torch
# from torch.utils.data import Dataset
# from .registry import register_dataset


# @register_dataset("synthetic")
# class SyntheticEEGDataset(Dataset):
#     def __init__(self, cfg, split="train"):
#         c = cfg
#         if split == "train":
#             n = c.train_samples
#         elif split == "val":
#             n = c.val_samples
#         else:
#             n = c.test_samples

#         self.x = torch.randn(n, c.num_channels, c.seq_len)
#         self.y = torch.randint(0, c.num_classes, (n,))

#     def __len__(self):
#         return len(self.y)

#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx]

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

        self.x = torch.randn(
            n,
            c.num_channels,
            10,   # num patches
            200   # patch dim
        )

        self.y = torch.zeros(n).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]