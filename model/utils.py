import argparse
from collections import OrderedDict
import itertools
import json
import numpy as np
from typing import Any, Callable, Dict, List

from hydra.utils import instantiate, get_class
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import lightning.pytorch as pl

def instantiate_scheduler(optimizer, cfg: DictConfig):
    def add_optimizer(optimizer, cfg):
        if isinstance(cfg, Dict):
            for k, v in cfg.items():
                cfg[k] = add_optimizer(optimizer, v)
            if '_target_' in cfg:
                if issubclass(get_class(cfg.get('_target_')), lr_scheduler.LRScheduler):
                    cfg.update(dict(optimizer=optimizer))
            return cfg
        elif isinstance(cfg, List):
            return [add_optimizer(optimizer, x) for x in cfg]
        else:
            return cfg
    
    _cfg = OmegaConf.to_container(cfg, resolve=True)
    _cfg = add_optimizer(optimizer, _cfg)
    return instantiate(_cfg)

def load_pretrained_net(net, path):
    '''allow partial loading
    '''

    device = next(net.parameters()).device

    # load from checkpoint or state_dict
    print(f'trying to load pretrained from {path}')
    try:
        state_dict = torch.load(path, map_location=device)['state_dict']
    except:
        state_dict = torch.load(path, map_location=device)

    all_okay = True

    new_weights = net.state_dict()

    # partial loading. check key and shape
    for k in new_weights.keys():
        if not k in state_dict.keys():
            print(f'{k} is missing in pretrained')
            all_okay = False
        else:
            if new_weights[k].shape != state_dict[k].shape:
                print(f'skip {k}, required shape: {new_weights[k].shape}, pretrained shape: {state_dict[k].shape}')
                all_okay = False
            else:       
                new_weights[k] = state_dict[k]
    
    try:
        net.load_state_dict(new_weights)
        if all_okay:
            print('<All weights loaded successfully>')
    except:
        print(f'cannot load {path}. using intial net.')
        pass
    
    return net

