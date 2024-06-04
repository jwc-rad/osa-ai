import glob
import importlib
import json
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
from sklearn.model_selection import KFold
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset

from monai.transforms import Compose, LoadImage, BorderPad

from utils.misc import instantiate_list

class PSGBase(Dataset):
    def __init__(self, transform):
        super().__init__()
        self.prepare_transforms(transform)
        
    def __len__(self):        
        return self.image_size
    
    def __getitem__(self, index):
        read_items = self.read_data(index)

        return_items = self.run_transform(read_items)
        
        return return_items
    
    ## override this to define transforms
    def prepare_transforms(self, transform):
        tfm = instantiate_list(transform)
        self.run_transform = Compose(tfm) 
    
    ## override this to define self.keys, paths, and etc.
    def prepare_data(self):
        pass
        
    ## override this to read data by index
    def read_data(self, index):
        read_items = {}
        metadata = {}
        for k, p in self.image_paths.items():
            imageX_path = p[index % self.image_size]
            read_items[k] = imageX_path
            metadata[f'{k}_path'] = imageX_path
                
        if hasattr(self, 'label_paths'):
            label_path = self.label_paths[index % self.image_size]
            read_items['label'] = label_path
            read_items['label_raw'] = label_path
            metadata['label_path'] = label_path
            
        if hasattr(self, 'stage_paths'):
            stage_path = self.stage_paths[index % self.image_size]
            read_items['stage'] = stage_path
            read_items['stage_raw'] = stage_path
            metadata['stage_path'] = stage_path
                        
        read_items['metadata'] = metadata
        return read_items    
    
    
class PSG_V0(PSGBase):
    def __init__(
        self, transform, image_dir: Dict, label_dir=None, phase='train', iterations_per_epoch: int = None,
        **prepare_data_kwargs,
    ):
        """
        image_dir: channel_name: path_to_dir
        """
        super().__init__(transform)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch
        
        self.prepare_data(**prepare_data_kwargs)
        
    def __len__(self):        
        return self.image_size if self.iterations_per_epoch is None else self.iterations_per_epoch
        
    def prepare_data(self, dataset_json, cv_fold=0, image_extension='npy', label_extension='npy', **kwargs):
        with open(dataset_json, 'r') as f:
            dsj = json.load(f)
        this_split = dsj['split'][str(cv_fold)][self.phase]
        
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f'*.{image_extension}')))
            _keys = [os.path.basename(x).split(f'.{image_extension}')[0] for x in _paths]
            all_keys.append(set(_keys))
        if getattr(self, 'label_dir', None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.label_dir, f'*.{label_extension}')))
            _keys = [os.path.basename(x).split(f'.{label_extension}')[0] for x in _paths]
            all_keys.append(set(_keys))            
        _c_keys = sorted(set.intersection(*all_keys))
        _filtered_keys = [x for x in _c_keys if x in this_split]
        
        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f'*.{image_extension}')))
            _paths = [x for x in _paths if os.path.basename(x).split(f'.{image_extension}')[0] in _filtered_keys]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths  
        self.image_size = image_size
        
        if getattr(self, 'label_dir', None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.label_dir, f'*.{label_extension}')))
            _paths = [x for x in _paths if os.path.basename(x).split(f'.{label_extension}')[0] in _filtered_keys]
            self.label_paths = _paths
            
class PSG_V1(PSGBase):
    def __init__(
        self, transform, image_dir: Dict, label_dir=None, stage_dir=None, phase='train', iterations_per_epoch: int = None,
        **prepare_data_kwargs,
    ):
        """
        image_dir: channel_name: path_to_dir
        """
        super().__init__(transform)
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.stage_dir = stage_dir
        self.phase = phase
        self.iterations_per_epoch = iterations_per_epoch
        
        self.prepare_data(**prepare_data_kwargs)
        
    def __len__(self):        
        return self.image_size if self.iterations_per_epoch is None else self.iterations_per_epoch
        
    def prepare_data(self, dataset_json, cv_fold=0, image_extension='npy', label_extension='npy', stage_extension='npy', **kwargs):
        with open(dataset_json, 'r') as f:
            dsj = json.load(f)
        this_split = dsj['split'][str(cv_fold)][self.phase]
        
        all_keys = []
        for d in self.image_dir.values():
            _paths = sorted(glob.glob(os.path.join(d, f'*.{image_extension}')))
            _keys = [os.path.basename(x).split(f'.{image_extension}')[0] for x in _paths]
            all_keys.append(set(_keys))
        if getattr(self, 'label_dir', None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.label_dir, f'*.{label_extension}')))
            _keys = [os.path.basename(x).split(f'.{label_extension}')[0] for x in _paths]
            all_keys.append(set(_keys))            
        _c_keys = sorted(set.intersection(*all_keys))
        _filtered_keys = [x for x in _c_keys if x in this_split]
        
        images_paths = {}
        image_size = -1
        for k, d in self.image_dir.items():
            _paths = sorted(glob.glob(os.path.join(d, f'*.{image_extension}')))
            _paths = [x for x in _paths if os.path.basename(x).split(f'.{image_extension}')[0] in _filtered_keys]
            images_paths[k] = _paths
            image_size = len(_paths)
        self.image_paths = images_paths  
        self.image_size = image_size
        
        if getattr(self, 'label_dir', None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.label_dir, f'*.{label_extension}')))
            _paths = [x for x in _paths if os.path.basename(x).split(f'.{label_extension}')[0] in _filtered_keys]
            self.label_paths = _paths
            
        if getattr(self, 'stage_dir', None) is not None:
            _paths = sorted(glob.glob(os.path.join(self.stage_dir, f'*.{stage_extension}')))
            _paths = [x for x in _paths if os.path.basename(x).split(f'.{stage_extension}')[0] in _filtered_keys]
            self.stage_paths = _paths
            
            