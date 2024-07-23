from typing import Any, Callable, Dict, List

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

def instantiate_list(cfg: DictConfig) -> List:
    """make list of instantiated objects"""
    
    targets = instantiate(cfg)
    if OmegaConf.is_dict(targets):
        targets = list(targets.values())

    return targets