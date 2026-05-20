import hydra
import lightning as L
from omegaconf import DictConfig
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from src.module.registry import get_module
from src.visualizer.model_visualizer import print_model_tree
import src.module
from src.data.data import EEGDataModule
import os
from hydra.core.hydra_config import HydraConfig

@hydra.main(version_base=None, config_path='../../configs', config_name='config_pretrain_mape')
def main(cfg: DictConfig):

    module_cls = get_module(cfg.module.name)
    model = module_cls(cfg)
    print_model_tree(model)

if __name__ == '__main__':
    main()