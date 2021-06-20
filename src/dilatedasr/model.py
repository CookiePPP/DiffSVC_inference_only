from math import sqrt, log
import random
import numpy as np
from numpy import finfo

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from torch import Tensor
from typing import List, Tuple, Optional
from collections import OrderedDict

from src.utils.layers import ConvNorm, ConvNorm2D
from src.utils.utils import get_mask_from_lengths, dropout_frame, freeze_grads, grad_scale, elapsed_timer
from src.utils.noncausal_wn import DilatedWN
from src.diffsvc.model import SmoothedBatchNorm1d

def load_model(h, device='cuda'):
    model = Model(h)
    if torch.cuda.is_available() or 'cuda' not in device:
        model = model.to(device)
    return model


class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.n_mel = h.n_mel_channels
        self.mel_dropout = h.mel_dropout
        
        if getattr(self, 'melbn', True):# batchnorm (unit mean-std normalization)
            self.melbn = SmoothedBatchNorm1d(self.n_mel, momentum=0.01, eval_only_momentum=False, affine=False)
        
        self.encoder = DilatedWN(self.n_mel, h.n_symbols, h.wn_dim, cond_channels=0, n_blocks=h.wn_n_blocks, n_layers=h.wn_n_layers,
                kernel_size=3, dropout=0.5, pre_kernel_size=3, separable=False, partial_padding=True, rezero=True, weight_norm=True, LSUV_init=True)
    
    def forward(self,# gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd):#FloatTensor[B, 2]
        out = {}
        
        mel_mask = get_mask_from_lengths(mel_lengths)
        gt_mel = self.melbn(gt_mel.transpose(1, 2), mel_mask )
        gt_mel = F.dropout(gt_mel, p=self.mel_dropout, training=self.training)
        
        x = self.encoder(gt_mel, None, mel_lengths)
        log_probs = x.log_softmax(dim=2)
        
        out['log_probs'] = log_probs# [B, mel_T, n_symbols]
        out['latents'  ] = x        # [B, mel_T, n_symbols]
        return out
    
    @torch.no_grad()
    def align(self, gt_mel, mel_lengths):# FloatTensor[B, n_mel, mel_T], LongTensor[B]
        mel_mask = get_mask_from_lengths(mel_lengths)
        gt_mel = self.melbn(gt_mel.transpose(1, 2), mel_mask)
        
        return self.encoder(gt_mel, None, mel_lengths).transpose(1, 2)


class Model(nn.Module):
    def __init__(self, h):
        super(Model, self).__init__()
        self.generator = Generator(h)
        self.fp16_run = h.fp16_run
    
    def parse_batch(self, batch, device='cuda'):
        if self.fp16_run:# convert data to half-precision before giving to GPU (to reduce wasted bandwidth)
            batch = {k: v.half() if type(v) is torch.Tensor and v.dtype is torch.float else v for k,v in batch.items()}
        batch = {k: v.to(device) if type(v) is torch.Tensor else v for k,v in batch.items()}
        return batch
    
    def forward(self,#gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                   text, text_lengths,#  LongTensor[B, txt_T],        LongTensor[B]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd):#FloatTensor[B, 2]
        
        out = self.generator(gt_mel, mel_lengths, text, text_lengths, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
        return out


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        loss += F.l1_loss(dr, dg)
    return loss*2


def masked_l1_loss(targ, pred, lengths):
    numel = lengths.sum()
    loss = F.l1_loss(*torch.broadcast_tensors(targ, pred), reduction='none').masked_fill_(~get_mask_from_lengths(lengths)[:, :pred.shape[1]].unsqueeze(2), 0.0).sum()/numel
    return loss


def masked_mse_loss(targ, pred, lengths):
    numel = lengths.sum()
    loss = F.mse_loss(*torch.broadcast_tensors(targ, pred), reduction='none').masked_fill_(~get_mask_from_lengths(lengths)[:, :pred.shape[1]].unsqueeze(2), 0.0).sum()/numel
    return loss


def discriminator_loss(dr, dg, mel_lengths):# [B, mel_T, 1], [B, mel_T, 1], [B]
    real_target = torch.tensor(1., device=dr.device, dtype=dr.dtype)
    fake_target = torch.tensor(0., device=dg.device, dtype=dg.dtype)
    
    r_loss = masked_mse_loss(real_target, dr, mel_lengths)# torch.mean((1-dr)**2)
    g_loss = masked_mse_loss(fake_target, dg, mel_lengths)# torch.mean(dg**2)
    return (r_loss + g_loss)


def generator_loss(dg, mel_lengths):
    real_target = torch.tensor(1., device=dg.device, dtype=dg.dtype)
    fake_target = torch.tensor(0., device=dg.device, dtype=dg.dtype)
    
    g_loss = masked_mse_loss(real_target, dg, mel_lengths)# torch.mean((1-dg)**2)
    return 2.*g_loss

