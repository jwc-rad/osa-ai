import itertools
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, auc, RocCurveDisplay

from omegaconf import OmegaConf
from hydra.utils import instantiate

import torch
import lightning.pytorch as pl
import wandb

from monai.metrics import ROCAUCMetric, ConfusionMatrixMetric, get_confusion_matrix, compute_confusion_matrix_metric
from monai.networks.utils import one_hot

from utils.metrics import psg_event_analysis

from .utils import load_pretrained_net, instantiate_scheduler


class BaseModel(pl.LightningModule):           
    def load_pretrained(self, path):
        self.load_pretrained_nets(path, nets=self.net_names)
    
    def load_pretrained_nets(self, path, nets=[]):
        '''For loading state_dict of part of the model.
        Loading full model should be done by "load_from_checkpoint" (Lightning)
        '''
        
        device = next(self.parameters()).device
        
        # load from checkpoint or state_dict
        print(f'trying to load pretrained from {path}')
        try:
            state_dict = torch.load(path, map_location=device)['state_dict']
        except:
            state_dict = torch.load(path, map_location=device)
        
        if len(nets)==0:
            self.load_state_dict(state_dict)
        
        all_keys_match = True
        changed = False
        for name in nets:
            if hasattr(self, name):
                net = getattr(self, name)
                new_weights = net.state_dict()
                
                # first check if pretrained has all keys
                keys_match = True
                for k in new_weights.keys():
                    if not f'{name}.{k}' in state_dict.keys():
                        keys_match = False
                        all_keys_match = False
                        print(f"not loading {name} because keys don't match")
                        break
                if keys_match:
                    for k in new_weights.keys():
                        new_weights[k] = state_dict[f'{name}.{k}']
                    net.load_state_dict(new_weights)
                    changed = True
                        
        if changed:
            if all_keys_match:
                print('<All keys matched successfully>')
        else:
            print(f'nothing is loaded from {path}')
    
    
class PSGSegmentationModel(BaseModel):
    def __init__(self, **opt):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.is_train = opt['train']
        self.use_wandb = 'wandb' in opt['logger']
        
        # define networks
        if self.is_train:
            self.net_names = ['netS']
        else:
            self.net_names = ['netS']
                    
        for net in self.net_names:
            setattr(self, net, instantiate(OmegaConf.select(opt['networks'], net), _convert_='partial'))
            pretrained = OmegaConf.select(opt['networks'], f'pretrained.{net}')
            if pretrained:
                net = getattr(self, net)
                net = load_pretrained_net(net, pretrained)
        
        # define loss functions
        if self.is_train:            
            self.criterionSeg = instantiate(opt['loss_seg'], _convert_='partial')

        # define inferer
        if 'inferer' in opt:
            self.inferer = instantiate(opt['inferer'], _convert_='partial')
            
        # define metrics
        if 'metrics' in opt:
            self.metrics = {}
            for k, v in opt['metrics'].items():
                if '_target_' in v:
                    self.metrics[k] = instantiate(v, _convert_='partial')

    ### custom methods
    def set_input(self, batch):
        self.image = batch['image']
        if 'label' in batch.keys():
            self.label = batch['label']
        if 'stage' in batch.keys():
            self.stage = batch['stage']
            
    def _step_forward(self, batch, batch_idx):
        self.set_input(batch)
        
        self.image_seg = self.forward(self.image)

    ### pl methods
    
    def configure_optimizers(self):        
        netparams = [getattr(self, n).parameters() for n in ['netS'] if hasattr(self, n)]
        optimizer_GF = instantiate(self.hparams['optimizer'], params=itertools.chain(*netparams))
        
        optimizers = [optimizer_GF]
        schedulers = [{
            k: instantiate_scheduler(optimizer, v) if k=='scheduler' else v 
            for k,v in self.hparams['scheduler'].items()
        } for optimizer in optimizers]
        
        return optimizers, schedulers
    
    def forward(self, x):
        out = self.netS(x)
        return out
                    
    def training_step(self, batch, batch_idx):
        stage = 'train'
        self._step_forward(batch, batch_idx)
        
        loss = 0
        bs = self.image.size(0)
                                    
        # Classification loss: S(A) ~ Ya
        w0 = self.hparams['lambda_seg']
        if w0 > 0:            
            loss_S = self.criterionSeg(self.image_seg, self.label)
            self.log('loss/seg', loss_S, batch_size=bs, on_step=True, on_epoch=True)
        else:
            loss_S = 0
        loss += loss_S * w0  
        
        # combine loss
        #self.log('loss/train', loss, batch_size=bs, on_step=True, on_epoch=True)

        return loss
        
    def validation_step(self, batch, batch_idx):
        stage = 'valid'
        self.set_input(batch)
        bs = self.image.size(0)  
        #outputs = self.forward(self.image)
        outputs = self.inferer(self.image, self.forward)
        
        loss = None
        loss = self.criterionSeg(outputs, self.label)

        if hasattr(self, 'metrics'):
            bin_outputs = torch.softmax(outputs, 1)
            #bin_outputs = one_hot(outputs.argmax(1, keepdim=True), outputs.shape[1])
            for k in self.metrics.keys():
                if hasattr(self, 'stage'):
                    self.metrics[k](bin_outputs.float(), self.label.float(), mask=self.stage.float())
                else:            
                    self.metrics[k](bin_outputs.float(), self.label.float())
                if self.global_step == 0 and self.use_wandb:
                    for x in k.split('__'):
                        wandb.define_metric(f'metrics/valid_{x}', summary='max')
                    
        self.log(f'loss/{stage}', loss, batch_size=bs, on_step=True, on_epoch=True)
                    
        return loss
    
    def on_validation_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
    
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    kks = k.split('__')
                    for i in range(len(mean_metric)):
                        mmetric = mean_metric[i].item()
                        self.log(f'metrics/valid_{kks[i]}', mmetric)                    
                else:
                    mean_metric = mean_metric.item()
                    self.log(f'metrics/valid_{k}', mean_metric)
                self.metrics[k].reset()
            
    def predict_step(self, batch, batch_idx):
        self.set_input(batch)
        outputs = self.inferer(self.image, self.forward)
        self.outputs = outputs
        return None
    
    def test_step(self, batch, batch_idx):
        self.set_input(batch)
        outputs = self.inferer(self.image, self.forward)
        self.outputs = outputs
        
        if hasattr(self, 'metrics'):
            bin_outputs = torch.softmax(outputs, 1)
            #bin_outputs = one_hot(outputs.argmax(1, keepdim=True), outputs.shape[1])
            
            for k in self.metrics.keys():
                if hasattr(self, 'stage'):
                    self.metrics[k](bin_outputs.float(), self.label.float(), mask=self.stage.float())
                else:
                    self.metrics[k](bin_outputs.float(), self.label.float())
        return None
    
    def on_test_epoch_end(self):
        if not hasattr(self, 'metrics'):
            return
    
        for k in self.metrics.keys():
            if self.metrics[k].get_buffer() is not None:
                mean_metric = self.metrics[k].aggregate()
                if isinstance(mean_metric, list):
                    kks = k.split('__')
                    for i in range(len(mean_metric)):
                        mmetric = mean_metric[i].item()
                        self.log(f'test_metrics/{kks[i]}', mmetric)                    
                else:
                    mean_metric = mean_metric.item()
                    self.log(f'test_metrics/{k}', mean_metric)
                self.metrics[k].reset()