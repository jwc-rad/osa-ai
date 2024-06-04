import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange

class EncoderDecoderAttentionPoolClassifier(nn.Module):
    """
    encoder: input (batch, channel, size), e.g. CNN
    decoder: input (batch, size, channel), e.g. nn.LSTM
    attention: input (batch, size, channel), e.g. nn.TransformerEncoderLayer
    pool: input (batch, channel, size), e.g. nn.AdaptiveAvgPool1d(1), flatten
    
    """
    def __init__(
        self,
        encoder_class=nn.Identity,
        encoder_cfg=None,
        decoder_class=nn.Identity,
        decoder_cfg=None,
        attention_class=nn.Identity,
        attention_cfg=None,
        pool_class=nn.Identity,
        pool_cfg=None,
        classifier_class=nn.Identity,
        classifier_cfg=None,
    ) -> None:
        super().__init__()
        
        if encoder_cfg is None:
            encoder_cfg = {}
        if decoder_cfg is None:
            decoder_cfg = {}
        if attention_cfg is None:
            attention_cfg = {}
        if pool_cfg is None:
            pool_cfg = {}
        if classifier_cfg is None:
            classifier_cfg = {}
                
        self.encoder = encoder_class(**encoder_cfg)
        self.decoder = decoder_class(**decoder_cfg)
        self.attention = attention_class(**attention_cfg)
        self.pool = pool_class(**pool_cfg)
        self.classifier = classifier_class(**classifier_cfg)
        
    def forward(self, x):
        x_shape = x.shape[2:]
        einops_dims = ' '.join([x for i,x in enumerate(['h','w','d']) if i < len(x_shape)])
        channel_first_to_last = f"b c {einops_dims} -> b ({einops_dims}) c"
        channel_last_to_first = f"b ({einops_dims}) c -> b c {einops_dims}"
        
        x = self.encoder(x)
        
        x = rearrange(x, channel_first_to_last)
        x = self.decoder(x)
        x = self.attention(x)
        x = rearrange(x, channel_last_to_first)
        
        x = self.pool(x)
        x = self.classifier(x)
        return x
    
    
class EncoderDecoderAttentionSegmentor(nn.Module):
    """
    encoder: input (batch, channel, size), e.g. CNN
    decoder: input (batch, size, channel), e.g. nn.LSTM
    attention: input (batch, size, channel), e.g. nn.TransformerEncoderLayer
    segmentor: input (batch, channel, size)
    """
    def __init__(
        self,
        encoder_class=nn.Identity,
        encoder_cfg=None,
        preattention_class=nn.Identity,
        preattention_cfg=None,
        decoder_class=nn.Identity,
        decoder_cfg=None,
        attention_class=nn.Identity,
        attention_cfg=None,
        segmentor_class=nn.Identity,
        segmentor_cfg=None,
    ) -> None:
        super().__init__()
        
        if encoder_cfg is None:
            encoder_cfg = {}
        if preattention_cfg is None:
            preattention_cfg = {}
        if decoder_cfg is None:
            decoder_cfg = {}
        if attention_cfg is None:
            attention_cfg = {}
        if segmentor_cfg is None:
            segmentor_cfg = {}
                
        self.encoder = encoder_class(**encoder_cfg)
        self.preattention =preattention_class(**preattention_cfg)
        self.decoder = decoder_class(**decoder_cfg)
        self.attention = attention_class(**attention_cfg)
        self.segmentor = segmentor_class(**segmentor_cfg)
        
    def forward(self, x):
        x_shape = x.shape[2:]
        einops_dims = ' '.join([x for i,x in enumerate(['h','w','d']) if i < len(x_shape)])
        channel_first_to_last = f"b c {einops_dims} -> b ({einops_dims}) c"
        channel_last_to_first = f"b ({einops_dims}) c -> b c {einops_dims}"
        
        x = self.encoder(x)
        
        x = rearrange(x, channel_first_to_last)
        x = self.preattention(x)
        x = self.decoder(x)
        x = self.attention(x)
        x = rearrange(x, channel_last_to_first)
        
        x = self.segmentor(x)
        return x