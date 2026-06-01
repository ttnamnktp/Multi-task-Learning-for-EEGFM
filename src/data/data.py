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