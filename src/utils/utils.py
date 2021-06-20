import numpy as np
from scipy.io.wavfile import read
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.autograd import Function

# Taken From https://github.com/janfreyberg/pytorch-revgrad/blob/master/src/pytorch_revgrad/functional.py
class GradScale(Function):
    @staticmethod
    def forward(ctx, input_, alpha_):
        ctx.save_for_backward(input_, alpha_ if type(alpha_) is torch.Tensor else torch.tensor(alpha_, requires_grad=False))
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        _, alpha_ = ctx.saved_tensors
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * alpha_
        return grad_input, None
grad_scale = GradScale.apply

class freeze_grads():
    def __init__(self, submodule):
        self.submodule = submodule
    
    def __enter__(self):
        self.require_grads = []
        for param in self.submodule.parameters():
            self.require_grads.append(param.requires_grad)
            param.requires_grad = False
    
    def __exit__(self, type, value, traceback):
        for i, param in enumerate(self.submodule.parameters()):
            param.requires_grad = self.require_grads[i]


@torch.jit.script
def get_mask_from_lengths(lengths: torch.Tensor, max_len:int = 0):
    if max_len == 0:
        max_len = int(torch.max(lengths).item())
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1))
    return mask

@torch.jit.script
def get_mask_3d(widths, heights, max_w: Optional[torch.Tensor] = None, max_h: Optional[torch.Tensor] = None):
    device = widths.device
    B = widths.shape[0]
    if max_w is None:
        max_w = torch.max(widths)
    if max_h is None:
        max_h = torch.max(heights)
    seq_w = torch.arange(0, max_w, device=device)# [max_w]
    seq_h = torch.arange(0, max_h, device=device)# [max_h]
    mask_w = (seq_w.unsqueeze(0) < widths.unsqueeze(1)).to(torch.bool) # [1, max_w] < [B, 1] -> [B, max_w]
    mask_h = (seq_h.unsqueeze(0) < heights.unsqueeze(1)).to(torch.bool)# [1, max_h] < [B, 1] -> [B, max_h]
    mask = (mask_w.unsqueeze(2) & mask_h.unsqueeze(1))# [B, max_w, 1] & [B, 1, max_h] -> [B, max_w, max_h]
    return mask# [B, max_w, max_h]


def get_drop_frame_mask_from_lengths(lengths, drop_frame_rate):
    batch_size = lengths.size(0)
    max_len = int(torch.max(lengths).item())
    mask = get_mask_from_lengths(lengths)
    drop_mask = torch.empty([batch_size, max_len], device=lengths.device).uniform_(0., 1.) < drop_frame_rate
    drop_mask = drop_mask * mask
    return drop_mask


def dropout_frame(mels, global_mean, mel_lengths, drop_frame_rate, soft_mask=False, local_mean=False, local_mean_range=5):
    drop_mask = get_drop_frame_mask_from_lengths(mel_lengths, drop_frame_rate).unsqueeze(1)# [B, 1, mel_T]
    
    if local_mean:
        def padidx(i):
            pad = (i+1)//2
            return (pad, -pad) if i%2==0 else (-pad, pad)
        mel_mean = sum([F.pad(mels.detach(), padidx(i), mode='replicate') for i in range(local_mean_range)])/local_mean_range# [B, n_mel, mel_T]
    else:
        if len(global_mean.shape) == 1:
            mel_mean = global_mean.unsqueeze(0) #    [n_mel] -> [B, n_mel]
        if len(mel_mean.shape) == 2:
            mel_mean = mel_mean.unsqueeze(-1)# [B, n_mel] -> [B, n_mel, mel_T]
    
    dropped_mels = (mels * ~drop_mask) + (mel_mean * drop_mask)
    if soft_mask:
        rand_mask = torch.rand(dropped_mels.shape[0], 1, dropped_mels.shape[2], device=mels.device, dtype=mels.dtype)
        rand_mask_inv = (1.-rand_mask)
        dropped_mels = (dropped_mels*rand_mask) + (mels*rand_mask_inv)
    return dropped_mels


# taken from https://stackoverflow.com/a/30024601
import time
class elapsed_timer(object):
    def __init__(self, msg=''):
        self.msg = msg
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, typ, value, traceback):
        print(f'{self.msg} took {time.time()-self.start}s')
