import os
import json
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from functools import partial
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)# get select a indexes along last dim using t
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class GaussianDiffusion(nn.Module):
    def __init__(self, n_mel_channels, lin_start=1e-4, lin_end=0.06, lin_n_steps=100, loss_type='l1'):
        super().__init__()
        self.n_mel_channels = n_mel_channels
        self.set_noise_schedule(lin_start, lin_end, lin_n_steps)
        self.loss_type = loss_type# Choice['l1','l2']
    
    def set_noise_schedule(self, lin_start, lin_end, lin_n_steps, device='cpu'):
        betas = np.linspace(lin_start, lin_end, lin_n_steps)# T == len(schedule_list)
                                                     # [0.0001, ..., 0.0301, ..., 0.0600]
        
        alphas = 1. - betas# -> [0.9999, ..., 0.9699, ..., 0.9400]
        alphas_cumprod = np.cumprod(alphas, axis=0)# [0.9999, ..., <0.9699, ..., <0.9400]
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])# [1.0000, 0.9999, ..., <0.9699, ...]
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        to_torch = partial(torch.tensor, device=device, dtype=torch.float32)# ??? numpy/list -> pytorch?
        
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))                 # sqrt(a_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))  # sqrt(1.0 - a_cumprod)
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))    #  log(1.0 - a_cumprod)
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))      #  log(1.0 / a_cumprod - 1)
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))# sqrt(1.0 - a_cumprod - 1)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20*self.num_timesteps))))
        
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))
    
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance
    
    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )
    
    def q_posterior(self, x_start, x_t, t):# x_start = pred_randn_noise, x_t = noisy_spectrogram, t = noise_level_idx
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped# ???, ???, ???
    
    def p_mean_variance(self, noisy_mel, t, cond, clip_denoised: bool):
        pred_noise = self.denoise_fn(noisy_mel, t, cond)# main model, predicts the noise from the noisy spectrogram
        pred_clean_mel = self.predict_start_from_noise(noisy_mel, t=t, noise=pred_noise)# use predicted noise and noisy spectrogram to guess the clean spectrogram
        
        if clip_denoised:# clamp [-1.0, 1.0] (only useful for audio)
            pred_clean_mel.clamp_(-1., 1.)
        
        model_mean, posterior_var, posterior_logvar = self.q_posterior(x_start=pred_clean_mel, x_t=noisy_mel, t=t)# pred_mel, noisy_mel, t
        return model_mean, posterior_var, posterior_logvar
    
    @torch.no_grad()
    def p_sample(self, noisy_mel, t, cond, clip_denoised=True, repeat_noise=False):# noisy_mel, t, cond
        b, *_, device = *noisy_mel.shape, noisy_mel.device
        model_mean, _, model_log_variance = self.p_mean_variance(noisy_mel=noisy_mel, t=t, cond=cond, clip_denoised=clip_denoised)
        
        noise = noise_like(noisy_mel.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(noisy_mel.shape) - 1)))
        model_std = (0.5 * model_log_variance).exp()
        return model_mean + nonzero_mask * model_std * noise
    
    @torch.no_grad()
    def interpolate(self, x1, x2, t, cond, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)
        
        assert x1.shape == x2.shape
        
        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))
        
        x = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long), cond)
        x = x[:, 0].transpose(1, 2)
        return self.denorm_spec(x)
    
    @torch.no_grad()
    def sampling(self):
        b, *_, device = *self.cond.shape, self.cond.device
        t = self.num_timesteps
        shape = (self.cond.shape[0], 1, self.n_mel_channels, self.cond.shape[2])
        noisy_mel = torch.randn(shape, device=device)
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            noisy_mel = self.p_sample(noisy_mel,
                              torch.full((b,), i, device=device, dtype=torch.long),
                              self.cond,)
        noisy_mel = noisy_mel[:, 0].transpose(1, 2)
        output = self.denorm_spec(noisy_mel)
        return output
    
    def q_sample(self, x_start, t, noise=None):# adds t scale noise to x_start. Called on the input before every denoiser pass
        noise = default(noise, lambda: torch.randn_like(x_start))
        
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
    
    def get_noise_scalar(self, x_start, t):
        return extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
    
    def p_losses(self, x_start, t, cond, noise=None, mask=None):# called by forward() during training
        noise = default(noise, lambda: torch.randn_like(x_start))# randn() if noise is None else noise
        
        noised_mel = self.q_sample(x_start=x_start, t=t, noise=noise)# add noise to spectrogram
        epsilon = self.denoise_fn(noised_mel, t, cond)               # guess the noise that was added
        
        if self.loss_type == 'l1':
            mask = mask.unsqueeze(-1).transpose(1, 2)
            loss = (noise - epsilon).abs().squeeze(1).masked_fill(mask, 0.0).mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, epsilon)
        
        noised_mel, epsilon = noised_mel.squeeze().transpose(1, 2), epsilon.squeeze().transpose(1, 2)# reshape for outputs
        return noised_mel, epsilon, loss
    
    def forward(self, mel, cond, mel_mask):# [B, mel_T, n_mel], [B, mel_T, C], [B, mel_T] returns training loss if mel is not None else predicted spectrogram
        b, *_, device = *cond.shape, cond.device
        output=epsilon = None
        loss=t = torch.tensor([0.], device=device, requires_grad=False)
        
        self.cond = cond.transpose(1, 2)
        if mel is None:
            output = self.sampling()
        else:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()# [B] get random noise level for each item in batch
            
            mel = mel.transpose(1, 2)[:, None, :, :]# [B, mel_T, n_mel] -> [B, 1, n_mel, mel_T]
            output, epsilon, loss = self.p_losses(x, t, self.cond, mask=mel_mask)
        return output, epsilon, loss, t
