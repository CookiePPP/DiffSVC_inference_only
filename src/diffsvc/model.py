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
from src.diffsvc.diffusion import GaussianDiffusion

def load_model(h, device='cuda'):
    model = Model(h)
    if torch.cuda.is_available() or 'cuda' not in device:
        model = model.to(device)
    return model

class SmoothedBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, *args, eval_only_momentum=True, **kwargs):
        super(SmoothedBatchNorm1d, self).__init__(*args, **kwargs)
        self.register_buffer('iters_', torch.tensor(0).long())
        self.eval_only_momentum = eval_only_momentum # use momentum only for eval (set to True for hidden layers)
        self.momentum_eps = max(self.momentum, 0.01)
    
    def expand_vector(self, x, vec):
        if len(x.shape) == 3:
            if   len(vec.shape) == 1: vec = vec[None, None, :]# [1, 1, C]
            elif len(vec.shape) == 2: vec = vec[:, None, :]   # [1, 1, C]
        elif len(x.shape) == 2:
            if len(vec.shape) == 1: vec = vec[None, :]# [1, C]
        return vec
    
    def forward(self,
            x:             Tensor      ,# [B, T, C] or [B, C]
            mask: Optional[Tensor]=None,# [B, T]
            ):
        training = self.training
        x_dims = len(x.shape)
        assert x_dims in [2, 3], 'input must have 2/3 dims of shape [B, C] or [B, T, C]'
        assert mask is None or len(mask.shape) == 2, 'mask must have shape [B, T]'
        
        if mask is not None and x_dims == 3:# must be [B, C, T] and have mask
            x_selected = x[mask]# [B, T, C] -> [B*T, C]
            y_selected = super(SmoothedBatchNorm1d, self).forward(x_selected)# [B*T, C] -> [B*T, C]
            
            if not self.eval_only_momentum and self.training and ( self.iters_ > 2.0/self.momentum_eps ):
                self.eval(); y_selected = super(SmoothedBatchNorm1d, self).forward(x_selected); self.train()# [B*T, C] -> [B*T, C]
            
            y = x.clone()
            y[mask] = y_selected# [B *T, C]
        else:
            y = super(SmoothedBatchNorm1d, self).forward(x)
            if not self.eval_only_momentum and self.training and ( self.iters_ > 2.0/self.momentum_eps ):
                self.eval(); y = super(SmoothedBatchNorm1d, self).forward(x); self.train()
        
        self.iters_ += 1
        return y.to(x)# [B, T, C] or [B, C]
    
    def inverse(self,
            y:             Tensor      ,# [B, T, C]
            mask: Optional[Tensor]=None,# [B, T]
            ):
        y_shape = y.shape
        if len(y_shape) == 2:# if 2 dims, assume shape is [B, C]
            y = y.unsqueeze(1)# [B, C] -> [B, T, C]
            assert mask is None, "mask cannot be used without a time dimension on the input y"
        assert y.shape[2] == self.num_features, f"input must be shape [B, T, {self.num_features}], expected {self.num_features} input channels but found {y.shape[1]}"
        with torch.no_grad():
            mean = self.running_mean
            std  = self.running_var.sqrt()
            x = (y*std[None, None, :])+mean[None, None, :]
            if mask is not None:
                x.masked_fill_(~mask.unsqueeze(2), 0.0)
            if len(y_shape) == 2:
                x.squeeze(1)
        return x


class ConvStack(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, kernel_size, dropout=0.1, residual=False, act_func=nn.ReLU()):
        super(ConvStack, self).__init__()
        self.input_dim = input_dim
        self.output_dim= output_dim
        self.hidden_dim= hidden_dim
        self.residual = residual
        if self.residual:
            assert input_dim == output_dim
        self.conv = []
        for i in range(n_layers):
            input_dim  = self.input_dim  if   i==0        else self.hidden_dim
            output_dim = self.output_dim if 1+i==n_layers else self.hidden_dim
            self.conv.append(ConvNorm(input_dim, output_dim, kernel_size, act_func=act_func, dropout=dropout))
        self.conv = nn.ModuleList(self.conv)
    
    def forward(self, x, x_len=None):
        if x_len is not None:
            x_mask = get_mask_from_lengths(x_len).unsqueeze(1)# [B, 1, T]
            x = x*x_mask
        if self.residual:
            x_res = x
        
        for conv in self.conv:
            x = conv(x)
            if x_len is not None: x = x*x_mask
        
        if self.residual:
            x = x + x_res
        return x

class GLU(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GLU, self).__init__()
        self.proj = ConvNorm(input_dim, output_dim*2, channel_last_dim=True, LSUV_init=False)
    
    #@torch.jit.script
    def jit(self, x):
        x, y = x.chunk(2, dim=-1)
        x = x*y.sigmoid()
        return x
    
    def forward(self, x):
        x = self.proj(x)
        x = self.jit(x)
        return x

class SpeakerEncoder(nn.Module):
    def __init__(self, h):
        super(SpeakerEncoder, self).__init__()
        self.embedding = nn.Embedding(h.n_speakers, h.speaker_embed_dim)
        self.conv = ConvStack(h.speaker_embed_dim+4, h.speaker_embed_dim*2, h.speaker_embed_dim*2, n_layers=3, kernel_size=1, act_func=nn.ReLU(), residual=False, dropout=0.2)
        self.GLU = GLU(h.speaker_embed_dim*2, h.speaker_embed_dim)
    
    def forward(self, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd):
        speaker_embed = self.embedding(speaker_id)
        speaker_embed = self.conv(torch.cat((speaker_embed, speaker_f0_meanstd, speaker_slyps_meanstd), dim=1).unsqueeze(-1)).squeeze(-1)
        speaker_embed = self.GLU(speaker_embed)
        return speaker_embed# [B, embed_dim]


class Generator(nn.Module):
    def __init__(self, h):
        super(Generator, self).__init__()
        self.n_mel = h.n_mel_channels
        
        if 1:# peak normalization [-1, 1]
            self.mel_std  = (1.8-log(h.stft_clamp_val))*0.5# rough mel std prediction
            self.mel_mean = (1.8+log(h.stft_clamp_val))*0.5#(10**torch.tensor(h.target_lufs/10)).log() if getattr(h, 'target_lufs', None) else -5.0# and taken from target_lufs
        else:# batchnorm (unit mean-std normalization)
            self.melbn = SmoothedBatchNorm1d(self.n_mel, momentum=0.01, eval_only_momentum=False, affine=False)
        self.diffusion = GaussianDiffusion(self.n_mel, lin_start=1e-4, lin_end=0.06, lin_n_steps=100)
        LSUV_init = False; act_func = nn.SiLU()
        
        self.prefeat_dim = 128
        self.speaker_encoder  = SpeakerEncoder(h)
        self.loudness_encoder = nn.Sequential(
                                    ConvNorm(               1, self.prefeat_dim, act_func=act_func, dropout=0.2, channel_last_dim=True, LSUV_init=LSUV_init),
                                    ConvNorm(self.prefeat_dim, self.prefeat_dim, act_func=act_func, dropout=0.2, channel_last_dim=True, LSUV_init=LSUV_init),
                                )
        self.melody_encoder   = nn.Sequential(
                                    ConvNorm(               1, self.prefeat_dim, act_func=act_func, dropout=0.2, channel_last_dim=True, LSUV_init=LSUV_init),
                                    ConvNorm(self.prefeat_dim, self.prefeat_dim, act_func=act_func, dropout=0.2, channel_last_dim=True, LSUV_init=LSUV_init),
                                )
        self.step_encoder     = nn.Sequential(
                                    ConvNorm(               1, self.prefeat_dim, act_func=act_func, dropout=0.2, channel_last_dim=True, LSUV_init=LSUV_init),
                                    ConvNorm(self.prefeat_dim, self.prefeat_dim, act_func=act_func, dropout=0.2, channel_last_dim=True, LSUV_init=LSUV_init),
                                )
        self.cond_rezero = nn.Parameter(torch.tensor(1e-3))
        
        ppg_dim = h.n_symbols
        self.cond_dim = 2*self.prefeat_dim + ppg_dim + h.speaker_embed_dim
        self.denoiser = DilatedWN(self.n_mel+self.prefeat_dim, self.n_mel, h.wn_dim, cond_channels=self.cond_dim, n_blocks=h.wn_n_blocks, n_layers=h.wn_n_layers,
                kernel_size=3, dropout=getattr(h, 'wn_dropout', 0.1), pre_kernel_size=1, dilation_base=getattr(h, 'wn_dilation_base', 1), separable=False, partial_padding=True, rezero=True, weight_norm=True, LSUV_init=LSUV_init)
    
    def forward(self,# gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                     gt_perc_loudness,# FloatTensor[B]
                       gt_frame_logf0,# FloatTensor[B, mel_T]
                            frame_ppg,# FloatTensor[B, ppg_dim, mel_T]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd):#FloatTensor[B, 2]
        out = {}
        gt_mel = gt_mel.transpose(1, 2)# [B, n_mel, mel_T] -> [B, mel_T, n_mel]
        B, mel_T, n_mel = gt_mel.shape
        
        if hasattr(self, 'melbn'):
            mel_mask = get_mask_from_lengths(mel_lengths)
            gt_mel.data[:] = self.melbn(gt_mel, mel_mask)
        else:
            gt_mel.data[:] = (gt_mel-self.mel_mean)/self.mel_std
        
        randn_noise = torch.randn_like(gt_mel)# get randn_noise
        t = torch.randint(0, self.diffusion.num_timesteps, (B,), device=gt_mel.device).long()# [B] get random noise level for each item in batch
        noise_scalar = self.diffusion.get_noise_scalar(gt_mel, t)                     # [B, 1, 1] get noise scalar for MSE_Scaled and MAE_Scaled later
        noisy_gt_mel = self.diffusion.q_sample(x_start=gt_mel, t=t, noise=randn_noise)# add noise to mels
        
        speaker_embed = self.speaker_encoder (speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)# [B, embed_dim]
        step_feat     = self.step_encoder    (noise_scalar)                   # [B,     1, prefeat_dim]
        loudness_feat = self.loudness_encoder(gt_perc_loudness[:, None, None])# [B,     1, prefeat_dim]
        logf0_feat    = self.melody_encoder  (gt_frame_logf0.unsqueeze(-1))   # [B, mel_T, prefeat_dim]
        
        cond = torch.cat((
            speaker_embed[:, None, :].expand(B, mel_T, -1),# [B,     1,    spkr_dim]
#           step_feat                .expand(B, mel_T, -1),# [B,     1, prefeat_dim]
            loudness_feat            .expand(B, mel_T, -1),# [B,     1, prefeat_dim]
            logf0_feat               .expand(B, mel_T, -1),# [B, mel_T, prefeat_dim]
            frame_ppg.transpose(1, 2).expand(B, mel_T, -1),# [B, mel_T,   n_symbols]
        ), dim=2,)# -> [B, mel_T, cond_dim]
        cond = (self.cond_rezero + random.random()*1e-5) * cond
        
        noisy_gt_mel_with_step_feat = torch.cat((noisy_gt_mel, step_feat.expand(B, mel_T, -1)), dim=2)# [B, mel_T, n_mel], [B, 1, prefeat_dim] -> [B, mel_T, n_mel+prefeat_dim]
        predicted_randn_noise = self.denoiser(noisy_gt_mel_with_step_feat, cond, mel_lengths)# predict the noise from the noisy_mels
        
        #print(
        #    t.shape,                    # torch.Size([16])
        #    gt_mel.shape,               # torch.Size([16, 771, 128])
        #    noisy_gt_mel.shape,         # torch.Size([16, 771, 128])
        #    noise_scalar.shape,         # torch.Size([16,   1,   1])
        #    cond.shape,                 # torch.Size([16, 771, 819])
        #    randn_noise.shape,          # torch.Size([16, 771, 128])
        #    predicted_randn_noise.shape,# torch.Size([16, 771, 128])
        #)
        
        out['t'                    ] = t           # [B]
        out['noise_scalar'         ] = noise_scalar# [B, 1, 1]
        out['randn_noise'          ] = randn_noise          # [B, mel_T, n_mel]
        out['predicted_randn_noise'] = predicted_randn_noise# [B, mel_T, n_mel]
        out['noisy_gt_mel'         ] = noisy_gt_mel         # [B, mel_T, n_mel]
        
        if True:# take pred_randn_noise away from noisy_gt_mel to output pred_clean_mel
            pred_gt_mel = self.diffusion.predict_start_from_noise(noisy_gt_mel, t, predicted_randn_noise)
            out['pred_mel'] = pred_gt_mel
        
        return out
    
    @torch.no_grad()
    def voice_conversion_main(self,
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B] # take from reference/source
                     gt_perc_loudness,# FloatTensor[B]                              # take from reference/source
                       gt_frame_logf0,# FloatTensor[B, mel_T]                       # take from reference/source
                            frame_ppg,# FloatTensor[B, ppg_dim, mel_T]              # take from reference/source
                           speaker_id,#  LongTensor[B]                              # take from target speaker
                   speaker_f0_meanstd,# FloatTensor[B, 2]                           # take from target speaker
                speaker_slyps_meanstd,# FloatTensor[B, 2]                           # take from target speaker
                        t_step_size=1,
                        t_max_step=None,):
        gt_mel = gt_mel.transpose(1, 2)# [B, n_mel, mel_T] -> [B, mel_T, n_mel]
        B, mel_T, n_mel = gt_mel.shape
        
        # normalize mel
        if hasattr(self, 'melbn'):
            mel_mask = get_mask_from_lengths(mel_lengths)
            gt_mel.data[:] = self.melbn(gt_mel, mel_mask)
        else:
            gt_mel.data[:] = (gt_mel-self.mel_mean)/self.mel_std
        
        # get source features
        speaker_embed = self.speaker_encoder (speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)# [B, embed_dim]
        loudness_feat = self.loudness_encoder(gt_perc_loudness[:, None, None])# [B,     1, prefeat_dim]
        logf0_feat    = self.melody_encoder  (gt_frame_logf0.unsqueeze(-1))   # [B, mel_T, prefeat_dim]
        
        cond = torch.cat((
            speaker_embed[:, None, :].expand(B, mel_T, -1),# [B,     1,    spkr_dim]
            loudness_feat            .expand(B, mel_T, -1),# [B,     1, prefeat_dim]
            logf0_feat               .expand(B, mel_T, -1),# [B, mel_T, prefeat_dim]
            frame_ppg.transpose(1, 2).expand(B, mel_T, -1),# [B, mel_T,   n_symbols]
        ), dim=2,)# -> [B, mel_T, cond_dim]
        cond = self.cond_rezero * cond
        
        loop_mel = gt_mel
        max_t = min(t_max_step, self.diffusion.num_timesteps) if t_max_step is not None else self.diffusion.num_timesteps
        for t in reversed(range(0, max_t, t_step_size)):
            t = torch.tensor([t,], device=loop_mel.device, dtype=torch.long).expand(B)
            
            # run model and predict the noise
            noise_scalar = self.diffusion.get_noise_scalar(loop_mel, t)# [B, 1, 1] get noise scalar for MSE_Scaled and MAE_Scaled later
            step_feat = self.step_encoder(noise_scalar)# [B, 1, prefeat_dim]
            loop_mel_with_step_feat = torch.cat((loop_mel, step_feat.expand(B, mel_T, -1)), dim=2)# [B, mel_T, n_mel], [B, 1, prefeat_dim] -> [B, mel_T, n_mel+prefeat_dim]
            predicted_randn_noise = self.denoiser(loop_mel_with_step_feat, cond, mel_lengths)# predict the noise from the loop_mels
            
            # denoise mel
            pred_mel = self.diffusion.predict_start_from_noise(loop_mel, t, predicted_randn_noise)
            clamp_val = 1.0
            pred_mel.clamp_(min=-clamp_val, max=clamp_val)
            
            # add new noise
            randn_noise = torch.randn_like(loop_mel)
            pos_mean, _, pos_logvar = self.diffusion.q_posterior(x_start=pred_mel, x_t=loop_mel, t=t)
            model_std = pos_logvar.mul(0.5).exp()
            loop_mel = pos_mean + (0.0 if t==0 else randn_noise*model_std)
        pred_mel = loop_mel
        
        if hasattr(self, 'melbn'):
            mel_mask = get_mask_from_lengths(mel_lengths)
            pred_mel = self.melbn.inverse(pred_mel, mel_mask)
        else:
            pred_mel = (pred_mel*self.mel_std)+self.mel_mean# remove normalization
        
        return pred_mel

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
    
    def forward(self,# gt_frame_logf0s,# FloatTensor[B,     4, mel_T]
                 gt_mel,  mel_lengths,# FloatTensor[B, n_mel, mel_T], LongTensor[B]
                     gt_perc_loudness,# FloatTensor[B]
                       gt_frame_logf0,# FloatTensor[B, mel_T]
                            frame_ppg,# FloatTensor[B, ppg_dim, mel_T]
                           speaker_id,#  LongTensor[B]
                   speaker_f0_meanstd,# FloatTensor[B, 2]
                speaker_slyps_meanstd):#FloatTensor[B, 2]
        
        out = self.generator(gt_mel, mel_lengths, gt_perc_loudness, gt_frame_logf0, frame_ppg, speaker_id, speaker_f0_meanstd, speaker_slyps_meanstd)
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

