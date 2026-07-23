import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from hydra.core.hydra_config import HydraConfig
import os

from src.pretrain_module.registry import get_module
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
        save_dir=cfg.resume.original_run_dir if cfg.resume.enabled else hydra_dir,
        name="tb_logs",
        version=cfg.resume.tb_version if cfg.resume.enabled else None,
    )
    print("Logger dir:", logger.log_dir)

    # 3. Callbacks
    # Định nghĩa thư mục lưu checkpoint động dựa trên việc có resume hay không
    checkpoint_dir = (
        os.path.join(cfg.resume.original_run_dir, "checkpoints") 
        if cfg.resume.enabled 
        else os.path.join(hydra_dir, "checkpoints")
    )
    ckpt = ModelCheckpoint(
        dirpath=checkpoint_dir, # Thay hydra_dir bằng checkpoint_dir đã phân nhánh
        monitor="valid/average_valid_loss",   # dùng đúng metric pretrain
        mode="min",
        save_top_k=1,
        filename="best-{epoch}-{valid/average_valid_loss:.4f}",
        verbose=True,
        save_last=True,
    )
    print("Checkpoint dir:", ckpt.dirpath)

    lr_monitor = LearningRateMonitor(
        logging_interval="step"
    )

    # 4. Khởi tạo Trainer
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=logger,
        callbacks=[ckpt, lr_monitor],
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps
    )

    trainer.fit(
        model, 
        datamodule=datamodule,
        ckpt_path=cfg.resume.ckpt_path if cfg.resume.enabled else None,
        )

if __name__ == '__main__':
    main()