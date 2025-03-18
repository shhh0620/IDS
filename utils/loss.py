# PatchNCE loss from https://github.com/taesungp/contrastive-unpaired-translation
# https://github.com/YSerin/ZeCon/blob/main/optimization/losses.py
from typing import Tuple, Union, Optional, List

from torch.nn import functional as F
import torch
import numpy as np
import torch.nn as nn

def unsqueeze_xdim(z, xdim):
    bc_dim= (...,) + (None,) * len(xdim)
    return z[bc_dim]

def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res.expand(broadcast_shape)

class CutLoss:
    def __init__(self, n_patches=256, patch_size=1):
        self.n_patches = n_patches
        self.patch_size = patch_size

    def get_attn_cut_loss(self, ref_noise, trg_noise):
        loss = 0

        if len(ref_noise.shape) == 3:
            bs, res2, c = ref_noise.shape
            res = int(np.sqrt(res2))
            ref_noise_reshape = ref_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2) 
            trg_noise_reshape = trg_noise.reshape(bs, res, res, c).permute(0, 3, 1, 2)

        elif len(ref_noise.shape) == 4:
            ref_noise_reshape = ref_noise
            trg_noise_reshape = trg_noise

        for ps in self.patch_size:
            if ps > 1:
                pooling = nn.AvgPool2d(kernel_size=(ps, ps))
                ref_noise_pooled = pooling(ref_noise_reshape)
                trg_noise_pooled = pooling(trg_noise_reshape)
            else:
                ref_noise_pooled = ref_noise_reshape
                trg_noise_pooled = trg_noise_reshape

            ref_noise_pooled = nn.functional.normalize(ref_noise_pooled, dim=1)
            trg_noise_pooled = nn.functional.normalize(trg_noise_pooled, dim=1)

            ref_noise_pooled = ref_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2)
            patch_ids = np.random.permutation(ref_noise_pooled.shape[1]) 
            patch_ids = patch_ids[:int(min(self.n_patches, ref_noise_pooled.shape[1]))]
            patch_ids = torch.tensor(patch_ids, dtype=torch.long, device=ref_noise.device)

            ref_sample = ref_noise_pooled[:1, patch_ids, :].flatten(0, 1)

            trg_noise_pooled = trg_noise_pooled.permute(0, 2, 3, 1).flatten(1, 2) 
            trg_sample = trg_noise_pooled[:1 , patch_ids, :].flatten(0, 1) 
            
            loss += self.PatchNCELoss(ref_sample, trg_sample).mean() 
        return loss

    def PatchNCELoss(self, ref_noise, trg_noise, batch_size=1, nce_T = 0.07):
        batch_size = batch_size
        nce_T = nce_T
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        mask_dtype = torch.bool

        num_patches = ref_noise.shape[0]
        dim = ref_noise.shape[1]
        ref_noise = ref_noise.detach()
        
        l_pos = torch.bmm(
            ref_noise.view(num_patches, 1, -1), trg_noise.view(num_patches, -1, 1))
        l_pos = l_pos.view(num_patches, 1) 

        # reshape features to batch size
        ref_noise = ref_noise.view(batch_size, -1, dim)
        trg_noise = trg_noise.view(batch_size, -1, dim) 
        npatches = ref_noise.shape[1]
        l_neg_curbatch = torch.bmm(ref_noise, trg_noise.transpose(2, 1))

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=ref_noise.device, dtype=mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0) 
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / nce_T

        loss = cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=ref_noise.device))

        return loss


class DDSLoss:
    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep
    
    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):

        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(latent_input, timestep, embedd, cross_attention_kwargs=cross_attention_kwargs,).sample
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        return e_t

    def predict_zstart_from_eps(self, z_t, timestep, eps):
        assert z_t.shape == eps.shape
        alphas = self.scheduler.alphas_cumprod.to(self.device)
        snr_inv = (1 - alphas[timestep]).sqrt() / alphas[timestep].sqrt()

        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, timestep, z_t.shape) * z_t 
                - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, timestep, z_t.shape) * eps
                ), snr_inv[0]

    def __init__(self, t_min, t_max, unet, scheduler, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scheduler = scheduler
        self.device = device

        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.scheduler.alphas_cumprod).to(self.device)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.scheduler.alphas_cumprod - 1).to(self.device)


class IDSLoss:
    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep

    def get_epsilon_prediction(self, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):
        latent_input = torch.cat([z_t] * 2)
        timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:])

        e_t = self.unet(latent_input, timestep, embedd, cross_attention_kwargs=cross_attention_kwargs,).sample
        e_t_uncond, e_t = e_t.chunk(2)
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)
        assert torch.isfinite(e_t).all()

        return e_t

    def update_zt(self, z_0, z_t, timestep, embedd, guidance_scale=7.5, cross_attention_kwargs=None):
        """
        Update z_t^(src) to obtain x_t^(src*) (line 4~9 in Algo. 1)
        by FPR loss (Eq. (6) & Eq. (7))
        param:
            z_0: clean source latent
            z_t: noisy source latent
            timestep: timestep used for generating z_t
            embedd: source prompt embedding 
        """
        norm_ = []
        for i in range(self.iter_fp):
            with torch.enable_grad():
                z_t = z_t.requires_grad_()
                # get score epsilon_phi^(src) [line 5]
                e_t = self.get_epsilon_prediction(z_t, timestep, embedd, guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs)
                # calculate posterior mean of z_t^(src) [line 6]
                z_0_pred = self.predict_z0(e_t, timestep, z_t)
                # get FPR loss [line 7]
                difference = z_0 - z_0_pred
                norm = torch.linalg.norm(difference)
                norm_grad = torch.autograd.grad(outputs=norm, inputs=z_t)[0]
                # update z_t^(src) [line 8]
                z_t = z_t - norm_grad * self.scale
                z_t = z_t.detach()
                norm_.append(norm)

        return z_t, norm_

    def predict_z0(self, model_output, timestep, sample, alpha_prod_t=None):
        # to get posterior mean from z_t
        if alpha_prod_t is None:
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timestep.device)
            alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod.to(timestep.device)

        beta_prod_t = 1 - alpha_prod_t
        z0_pred = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        return z0_pred

    def estimate_eps(self, z_0, z_t, timestep):
        # estimate epsilon* from z_t^(src*) and z^(src)
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=z_0.device)
        alphas_cumprod = alphas_cumprod.to(dtype=z_0.dtype)

        sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(z_0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(z_0.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        eps = (z_t - sqrt_alpha_prod * z_0) / sqrt_one_minus_alpha_prod
        return eps


    def __init__(self, t_min, t_max, unet, scale, scheduler, iter_fp, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scale = scale
        self.scheduler = scheduler
        self.device = device
        self.iter_fp = iter_fp
