from copy import deepcopy
import numpy as np
from typing import Callable, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn

from monai.networks.blocks import ADN
from monai.networks.layers.convutils import same_padding, stride_minus_kernel_padding
from monai.networks.layers.factories import Conv, Pad, Pool
from monai.networks.layers.utils import get_act_layer
from monai.utils import ensure_tuple_rep

class Convolution(nn.Sequential):
    """
    extension of monai.networks.blocks.convolutions.Convolution
        - padding_mode
    """
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        strides: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        adn_ordering: str = "NDA",
        act: Optional[Union[Tuple, str]] = "PRELU",
        norm: Optional[Union[Tuple, str]] = "INSTANCE",
        dropout: Optional[Union[Tuple, str, float]] = None,
        dropout_dim: Optional[int] = 1,
        dilation: Union[Sequence[int], int] = 1,
        groups: int = 1,
        bias: bool = True,
        conv_only: bool = False,
        is_transposed: bool = False,
        padding: Optional[Union[Sequence[int], int]] = None,
        output_padding: Optional[Union[Sequence[int], int]] = None,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_transposed = is_transposed
        if padding is None:
            padding = same_padding(kernel_size, dilation)
        conv_type = Conv[Conv.CONVTRANS if is_transposed else Conv.CONV, self.spatial_dims]

        conv: nn.Module
        if is_transposed:
            if output_padding is None:
                output_padding = stride_minus_kernel_padding(1, strides)
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                output_padding=output_padding,
                groups=groups,
                bias=bias,
                dilation=dilation,
                padding_mode=padding_mode,
            )
        else:
            conv = conv_type(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
                padding_mode=padding_mode,
            )

        self.add_module("conv", conv)

        if conv_only:
            return
        if act is None and norm is None and dropout is None:
            return
        
        self.add_module(
            "adn",
            ADN(
                ordering=adn_ordering,
                in_channels=out_channels,
                act=act,
                norm=norm,
                norm_dim=self.spatial_dims,
                dropout=dropout,
                dropout_dim=dropout_dim,
            ),
        )
        

class StackedConvBasicBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        stride: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        padding_mode: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
    ):
        super().__init__()
        
        m = [
            Convolution(
                spatial_dims,
                in_channels,
                out_channels,
                strides=1 if pooling else stride,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                act=act,
                norm=norm,
                dropout=dropout,
            )
        ]
        
        for _ in range(num_convs - 1):
            m.append(Convolution(
                spatial_dims,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=kernel_size,
                padding_mode=padding_mode,
                act=act,
                norm=norm,
                dropout=dropout,                
            ))
        
        if pooling:
            m += [Pool[pooling, spatial_dims](stride)]
        
        self.convs = nn.Sequential(*m)

    def forward(self, x):
        return self.convs(x)
    
class StackedConvBlocks(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: Union[Sequence[int], int],
        num_convs: int = 2,
        stride: Union[Sequence[int], int] = 1,
        kernel_size: Union[Sequence[int], int] = 3,
        padding_mode: str = "zeros",
        pooling: str = None,
        norm: Union[Tuple, str] = "instance",
        act: Union[Tuple, str] = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: Optional[Union[Tuple, str, float]] = None,
        block: nn.Module = StackedConvBasicBlock,
    ) -> None:
        layers = []
        
        if isinstance(out_channels, int):
            layers.append(block(spatial_dims, in_channels, out_channels, num_convs, stride, kernel_size, padding_mode, pooling, norm, act, dropout))    
        else:
            _channels = [in_channels] + out_channels
            for ich, och in zip(_channels[:-1], _channels[1:]):
                layers.append(block(spatial_dims, ich, och, num_convs, stride, kernel_size, padding_mode, pooling, norm, act, dropout))
        
        super().__init__(*layers)
