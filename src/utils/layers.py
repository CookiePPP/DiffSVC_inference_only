import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit as jit
from torch import Tensor
from typing import List, Tuple, Optional
from src.utils.utils import get_mask_from_lengths


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear', dropout=0.):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)
        self.dropout = dropout
        
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))
    
    def forward(self, x):
        x = self.linear_layer(x)
        if self.training and self.dropout > 0.:
            x = F.dropout(x, p=self.dropout)
        return x


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=None,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', dropout=0., act_func=None, act_func_params={},
                 causal=False, separable=False, ignore_separable_warning=False, channel_last_dim=False, LSUV_init=False, n_LSUV_passes=3, partial_padding=False,
                 weight_norm=False, instance_norm=False, layer_norm=False, batch_norm=False, affine_norm=False, ignore_norm_warning=False):
        super(ConvNorm, self).__init__()
        if separable and (not ignore_separable_warning):
            assert out_channels//in_channels==out_channels/in_channels, "in_channels must be equal to or a factor of out_channels to use separable Conv1d."
        if not ignore_norm_warning:
            assert bool(instance_norm)+bool(layer_norm)+bool(batch_norm) <= 1, 'only one of instance_norm, layer_norm or batch_norm is recommended to be used at a time. Use ignore_norm_warning=True if you know what you\'re doing'
        self.instance_norm = nn.InstanceNorm1d(out_channels,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_channels, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    = nn.   BatchNorm1d(out_channels,             affine=affine_norm) if    batch_norm else None
        
        self.channel_last_dim = channel_last_dim
        self.partial_padding = partial_padding
        self.weight_norm = weight_norm
        self.act_func        = act_func
        self.act_func_params = act_func_params
        
        if dilation is None:
            dilation = 1
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)
        
        self.padding = padding
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.separable = (separable and in_channels==out_channels)
        self.dropout = dropout
        
        conv_groups = groups or (in_channels if self.separable else 1)
        self.is_linear = kernel_size==1 and conv_groups==1
        self.in_channels = in_channels
        self.out_channels = out_channels
        if self.is_linear:
            self.conv = nn.Linear(in_channels, out_channels, bias=bias)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels,
                                  kernel_size=kernel_size, stride=stride,
                                  padding=0 if causal else padding,
                                  dilation=dilation, bias=bias, groups=conv_groups)
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain('linear' if self.separable else w_init_gain))
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if self.separable:
            self.conv_d = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=bias)
            torch.nn.init.xavier_uniform_(
                self.conv_d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
            if weight_norm:
                self.conv_d = nn.utils.weight_norm(self.conv_d, name='weight')
        self.causal_pad = (kernel_size-1)*dilation if causal else 0
        
        if LSUV_init:
            self.n_LSUV_passes = n_LSUV_passes
            self.register_buffer('LSUV_init_done', torch.tensor(False))
        self.squeeze_t_dim = False
    
    def maybe_pad(self, signal, pad_right=False):
        if self.causal_pad:
            if pad_right:
                signal = F.pad(signal, (0, self.causal_pad))
            else:
                signal = F.pad(signal, (self.causal_pad, 0))
        return signal
    
    def pre(self, signal):# [B, C, T] or ([B, T, C] or [B, C])
        self.squeeze_t_dim = False
        if self.channel_last_dim:
            if self.is_linear:
                assert len(signal.shape) in [2, 3], f"input has {len(signal.shape)} dims, should have 2 or 3 for Conv1d/Linear"
                if len(signal.shape) == 2:
                    self.squeeze_t_dim = True
                    signal = signal.unsqueeze(1)# -> [B, T, C]
            else:
                assert len(signal.shape) == 3, f"input has {len(signal.shape)} dims, should have 3 for Conv1d"
            signal = signal.transpose(1, 2)# -> [B, C, T]
        assert signal.shape[1] == self.in_channels, f"input has {signal.shape[1]} channels but expected {self.in_channels}"
        signal = self.maybe_pad(signal)
        return signal
    
    def conv1(self, signal):
        return self.conv(signal.transpose(1, 2)).transpose(1, 2) if self.is_linear else self.conv(signal)
    
    def main(self, signal, ignore_norm=False):# [B, C, T]
        conv_signal = self.conv1(signal)
        if self.separable:
            conv_signal = self.conv_d(conv_signal)
        if self.partial_padding and self.padding:
            # multiply values near the edge by (total edge n_elements/non-padded edge n_elements)
            pad = self.padding
            mask = signal.abs().sum(1, True)!=0.0
            signal_divisor = F.conv1d(mask.float(), signal.new_ones((1, 1, self.kernel_size),)/self.kernel_size, padding=pad, dilation=self.dilation).clamp_(min=0.0, max=1.0).masked_fill_(~mask, 1.0)
            
            if self.conv.bias is not None:
                bias = self.conv.bias.view(1, self.out_channels, 1)# [1, oC, 1]
                conv_signal = conv_signal.sub_(bias).div(signal_divisor).add_(bias).masked_fill_(~mask, 0.0)
            else:
                conv_signal = conv_signal.div(signal_divisor).masked_fill_(~mask, 0.0)
        
        if ignore_norm:
            conv_signal = self.instance_norm(conv_signal) if self.instance_norm is not None else conv_signal
            conv_signal = self.   batch_norm(conv_signal) if self.   batch_norm is not None else conv_signal
            conv_signal = self.   layer_norm(conv_signal
                        .transpose(1, 2)).transpose(1, 2) if self.   layer_norm is not None else conv_signal
        if self.act_func is not None:
            conv_signal = self.act_func(conv_signal, **self.act_func_params)
        if self.training and self.dropout > 0.:
            conv_signal = F.dropout(conv_signal, p=self.dropout, training=self.training)
        if self.channel_last_dim:
            conv_signal = conv_signal.transpose(1, 2)# -> original shape
        if self.squeeze_t_dim:
            conv_signal = conv_signal.squeeze(1)
        return conv_signal# [B, C, T] or ([B, T, C] or [B, C])
    
    def forward(self, signal):# [B, C, T] or [B, T, C]
        signal = self.pre(signal)# -> [B, C, T+maybe_causal_padding]
        if hasattr(self, 'LSUV_init_done') and not self.LSUV_init_done and self.training:
            training = self.training
            self.eval()
            for i in range(self.n_LSUV_passes):
                with torch.no_grad():
                    if self.separable:
                        z = self.conv1(signal)
                        self.conv.weight.data /= z.std()
                        if hasattr(self.conv, 'bias'): self.conv.bias.data -= z.mean()
                    z = self.main(signal, ignore_norm=True)
                    if self.separable:
                        self.conv_d.weight.data /= z.std()
                        if hasattr(self.conv_d, 'bias'): self.conv_d.bias.data -= z.mean()
                    else:
                        self.conv.weight.data /= z.std()
                        if hasattr(self.conv, 'bias'): self.conv.bias.data -= z.mean()
            del z
            self.train(training)
            self.LSUV_init_done += True
        
        conv_signal = self.main(signal)
        return conv_signal# [B, C, T] or [B, T, C]


class ConvNorm2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=None,
                 padding=None, dilation=1, bias=True, w_init_gain='linear', dropout=0., act_func=None, act_func_params={},
                 separable=False, ignore_separable_warning=False, channel_last_dim=False, LSUV_init=False, n_LSUV_passes=3,
                 weight_norm=False, instance_norm=False, layer_norm=False, batch_norm=False, affine_norm=False, ignore_norm_warning=False):
        super(ConvNorm2D, self).__init__()
        if separable and (not ignore_separable_warning):
            assert out_channels//in_channels==out_channels/in_channels, "in_channels must be equal to or a factor of out_channels to use separable Conv2d."
        if not ignore_norm_warning:
            assert bool(instance_norm)+bool(layer_norm)+bool(batch_norm) <= 1, 'only one of instance_norm, layer_norm or batch_norm is recommended to be used at a time. Use ignore_norm_warning=True if you know what you\'re doing'
        self.instance_norm = nn.InstanceNorm2d(out_channels,             affine=affine_norm) if instance_norm else None
        self.layer_norm    = nn.   LayerNorm  (out_channels, elementwise_affine=affine_norm) if    layer_norm else None
        self.batch_norm    = nn.   BatchNorm2d(out_channels,             affine=affine_norm) if    batch_norm else None
        
        self.channel_last_dim = channel_last_dim
        self.weight_norm = weight_norm
        self.act_func        = act_func
        self.act_func_params = act_func_params
        
        if dilation is None:
            dilation = 1
        if padding is None:
            assert(kernel_size % 2 == 1)
            kernel_size_tuple = (kernel_size, kernel_size) if type(kernel_size) is int else kernel_size
            dilation_tuple    = (   dilation,    dilation) if type(dilation)    is int else dilation
            padding = ( int(dilation_tuple[0]*(kernel_size_tuple[0]-1)/2), int(dilation_tuple[1]*(kernel_size_tuple[1]-1)/2) )
        
        self.separable = (separable and in_channels==out_channels)
        self.dropout = dropout
        
        self.in_channels = in_channels
        self.conv = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias, groups=groups or (in_channels if self.separable else 1))
        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain('linear' if self.separable else w_init_gain))
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv, name='weight')
        if self.separable:
            self.conv_d = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=bias)
            torch.nn.init.xavier_uniform_(
                self.conv_d.weight, gain=torch.nn.init.calculate_gain(w_init_gain))
            if weight_norm:
                self.conv_d = nn.utils.weight_norm(self.conv_d, name='weight')
        
        if LSUV_init:
            self.n_LSUV_passes = n_LSUV_passes
            self.register_buffer('LSUV_init_done', torch.tensor(False))
    
    def pre(self, signal):# [B, iC, H, W] or [B, H, W, iC]
        if self.channel_last_dim:
            assert len(signal.shape) == 4, f"input has {len(signal.shape)} dims, should have 4 for Conv2d"
            signal = signal.permute(0, 3, 1, 2)# [B, H, W, iC] -> [B, iC, H, W]
        assert signal.shape[1] == self.in_channels, f"input has {signal.shape[1]} channels but expected {self.in_channels}"
        return signal# [B, iC, H, W]
    
    def main(self, signal, ignore_norm=False):# [B, iC, H, W]
        conv_signal = self.conv(signal)# [B, oC, oH, oW]
        if self.separable:
            conv_signal = self.conv_d(conv_signal)# [B, oC, oH, oW]
        if ignore_norm:
            conv_signal = self.instance_norm(conv_signal) if self.instance_norm is not None else conv_signal
            conv_signal = self.   batch_norm(conv_signal) if self.   batch_norm is not None else conv_signal
            conv_signal = self.   layer_norm(conv_signal
                .permute(0, 2, 3, 1)).permute(0, 3, 1, 2) if self.   layer_norm is not None else conv_signal
        if self.training and self.dropout > 0.:
            conv_signal = F.dropout(conv_signal, p=self.dropout, training=self.training)# [B, oC, oH, oW]
        if self.act_func is not None:
            conv_signal = self.act_func(conv_signal, **self.act_func_params)
        if self.channel_last_dim:
            conv_signal = conv_signal.permute(0, 2, 3, 1)# [B, oC, oH, oW] -> [B, oH, oW, oC]
        return conv_signal# [B, iC, H, W] or [B, H, W, iC]
    
    def forward(self, signal):# [B, iC, H, W] or [B, H, W, iC]
        signal = self.pre(signal)
        if hasattr(self, 'LSUV_init_done') and not self.LSUV_init_done and self.training:
            training = self.training
            self.eval()
            for i in range(self.n_LSUV_passes):
                with torch.no_grad():
                    if self.separable:
                        z = self.conv1(signal)
                        self.conv.weight.data /= z.std()
                        if hasattr(self.conv, 'bias'): self.conv.bias.data -= z.mean()
                    z = self.main(signal, ignore_norm=True)
                    if self.separable:
                        self.conv_d.weight.data /= z.std()
                        if hasattr(self.conv_d, 'bias'): self.conv_d.bias.data -= z.mean()
                    else:
                        self.conv.weight.data /= z.std()
                        if hasattr(self.conv, 'bias'): self.conv.bias.data -= z.mean()
            del z
            self.train(training)
            self.LSUV_init_done += True
        
        conv_signal = self.main(signal)
        return conv_signal# [B, iC, H, W] or [B, H, W, iC]
