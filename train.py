import numpy as np
import os
import random
from typing import Dict, List, Optional, Tuple
import wandb
#
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import torch

from monai.utils import set_determinism

from utils.hydra import instantiate_list

@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg: DictConfig):
    
    if cfg.seed.seed:
        torch.manual_seed(cfg.seed.seed)
    model = instantiate(cfg.model, _recursive_ = False)
    
    if cfg.seed.seed:
        set_determinism(cfg.seed.seed)
    dm = instantiate(cfg.data, _recursive_ = False)
    
    logger = instantiate_list(cfg.logger)
    callbacks = instantiate_list(cfg.callbacks)
    trainer = instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if logger:
        for x in trainer.loggers:
            #x.log_hyperparams(cfg)
            x.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
        
    trainer.fit(model, dm)
    
    if cfg.valid:
        trainer.predict(model, dataloaders=dm.val_dataloader(), ckpt_path='best')
    
    wandb.finish()

if __name__ == "__main__":
    main()