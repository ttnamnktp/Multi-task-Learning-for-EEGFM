import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.core.hydra_config import HydraConfig
import shutil
from pathlib import Path

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
        callbacks=callbacks,
        gradient_clip_val=1.0,       # Thêm dòng này để clip grad norm = 1.0
        gradient_clip_algorithm="norm" # Mặc định là theo norm
    )

    trainer.fit(model, datamodule=datamodule)

    # 5. Tester - BEST
    print(" ============== Testing BEST checkpoint... ============== ")
    trainer.test(
        ckpt_path="best",
        datamodule=datamodule
    )

    # 6. Tester - LAST
    print(" ============== Testing LAST checkpoint... ============== ")
    trainer.test(
        ckpt_path="last",
        datamodule=datamodule
    )

    # 7. Tự động xóa checkpoint sau khi test xong
    print(" ============== Cleaning up checkpoints... ============== ")
    # Lấy đường dẫn thư mục checkpoints từ callback
    for callback in trainer.callbacks:
        if isinstance(callback, ModelCheckpoint):
            ckpt_dir = callback.dirpath
            if ckpt_dir and Path(ckpt_dir).exists():
                print(f"Removing checkpoint directory: {ckpt_dir}")
                shutil.rmtree(ckpt_dir) # Xóa toàn bộ thư mục chứa file .ckpt

if __name__ == '__main__':
    main()