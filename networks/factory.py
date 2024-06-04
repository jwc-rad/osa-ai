from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.common_types import _size_any_t, _ratio_any_t
from einops.layers.torch import Rearrange
from einops import rearrange

from monai.networks.layers.utils import get_act_layer, get_dropout_layer, get_norm_layer

class LSTM(nn.LSTM):
    def __init__(self, *args, return_hidden=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._return_hidden = return_hidden
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        if self._return_hidden:
            return output, hidden
        else:
            return output

class GRU(nn.GRU):
    def __init__(self, *args, return_hidden=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._return_hidden = return_hidden
    def forward(self, input, hx=None):
        output, hidden = super().forward(input, hx=hx)
        if self._return_hidden:
            return output, hidden
        else:
            return output

class MLP(nn.Sequential):
    """Modified from torchvision.ops.misc.MLP
    input shape: (batch, size, channel) if use_linear else (batch, channel, size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[List[int]] = None,
        norm: Optional[Union[Tuple, str]] = None,
        act: Optional[Union[Tuple, str]] = None,
        bias: bool = True,
        dropout: Optional[Union[Tuple, str, float]] = None,
        spatial_dims: int = 1,
        use_linear= True,
    ) -> None:
        if hidden_channels is None:
            hidden_channels = []
        pwconv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][spatial_dims - 1]
        
        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            if use_linear:
                layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            else:
                layers.append(pwconv(in_dim, hidden_dim, kernel_size=1, bias=bias))
            if norm is not None:
                layers.append(get_norm_layer(name=norm, spatial_dims=1, channels=hidden_dim))    
            if act is not None:
                layers.append(get_act_layer(act))
            if i < len(hidden_channels) - 1 and dropout is not None:
                layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
            in_dim = hidden_dim

        if dropout is not None:
            layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
        
        if use_linear:
            layers.append(nn.Linear(in_dim, out_channels, bias=bias))
        else:
            layers.append(pwconv(in_dim, out_channels, kernel_size=1, bias=bias))
                
        super().__init__(*layers)
        
class MLPUpsample(nn.Sequential):
    """
    input shape: (batch, channel, size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[List[int]] = None,
        norm: Optional[Union[Tuple, str]] = None,
        act: Optional[Union[Tuple, str]] = None,
        bias: bool = True,
        dropout: Optional[Union[Tuple, str, float]] = None,
        spatial_dims: int = 1,
        #use_linear= True,
        size: Optional[_size_any_t] = None, 
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = 'nearest', 
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ) -> None:
        if hidden_channels is None:
            hidden_channels = []
            
        pwconv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][spatial_dims - 1]

        layers = []
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            layers.append(pwconv(in_dim, hidden_dim, kernel_size=1, bias=bias))
            if norm is not None:
                layers.append(get_norm_layer(name=norm, spatial_dims=1, channels=hidden_dim))    
            if act is not None:
                layers.append(get_act_layer(act))
            if i < len(hidden_channels) - 1 and dropout is not None:
                layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
            in_dim = hidden_dim

        if dropout is not None:
            layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
        layers.append(pwconv(in_dim, out_channels, kernel_size=1, bias=bias))
                
        layers.append(nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor))
                
        super().__init__(*layers)
        
class UpsampleMLP(nn.Sequential):
    """
    input shape: (batch, channel, size)
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[List[int]] = None,
        norm: Optional[Union[Tuple, str]] = None,
        act: Optional[Union[Tuple, str]] = None,
        bias: bool = True,
        dropout: Optional[Union[Tuple, str, float]] = None,
        spatial_dims: int = 1,
        #use_linear= True,
        size: Optional[_size_any_t] = None, 
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = 'nearest', 
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
    ) -> None:
        if hidden_channels is None:
            hidden_channels = []
            
        pwconv = [nn.Conv1d, nn.Conv2d, nn.Conv3d][spatial_dims - 1]

        layers = []
        layers.append(nn.Upsample(size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor))
        
        in_dim = in_channels
        for i, hidden_dim in enumerate(hidden_channels):
            layers.append(pwconv(in_dim, hidden_dim, kernel_size=1, bias=bias))
            if norm is not None:
                layers.append(get_norm_layer(name=norm, spatial_dims=1, channels=hidden_dim))    
            if act is not None:
                layers.append(get_act_layer(act))
            if i < len(hidden_channels) - 1 and dropout is not None:
                layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
            in_dim = hidden_dim

        if dropout is not None:
            layers.append(get_dropout_layer(name=dropout, dropout_dim=1))
        layers.append(pwconv(in_dim, out_channels, kernel_size=1, bias=bias))
                
        super().__init__(*layers)
        
        
        
class GlobalAveragePooling(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        keepdim: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.keepdim = keepdim

        types = (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d)
        self.pool = types[spatial_dims - 1]
        
    def forward(self, x):
        x = self.pool(1)(x)
        if not self.keepdim:
            x = x.flatten(start_dim=1)
        return x
    
class GlobalMaxPooling(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        keepdim: bool = False,
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.keepdim = keepdim

        types = (nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)
        self.pool = types[spatial_dims - 1]
        
    def forward(self, x):
        x = self.pool(1)(x)
        if not self.keepdim:
            x = x.flatten(start_dim=1)
        return x
    
class ReturnLast(nn.Module):
    def __init__(
        self,
    ) -> None:    
        super().__init__()
    
    def forward(self, x):
        x = x.flatten(start_dim=2)
        return x[...,-1]