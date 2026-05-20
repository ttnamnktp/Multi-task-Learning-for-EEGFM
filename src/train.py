import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.core.hydra_config import HydraConfig

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
        callbacks=callbacks
    )

    trainer.fit(model, datamodule=datamodule)

    # 5. Tester
    trainer.test(
        ckpt_path="best",
        datamodule=datamodule
    )

if __name__ == '__main__':
    main()