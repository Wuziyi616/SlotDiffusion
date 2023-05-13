"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from .utils import noise_like, extract_to
from .cond_ddpm import CondDDPM


class LDM(CondDDPM):
    """Latent Diffusion Model."""

    def __init__(
        self,
        resolution,
        vae_dict,
        unet_dict,
        use_ema=True,
        diffusion_dict=dict(
            pred_target='eps',  # 'eps', 'x0', 'v'
            z_scale_factor=1.,
            timesteps=1000,
            beta_schedule="linear",
            linear_start=1e-4,
            linear_end=2e-2,
            cosine_s=8e-3,
            log_every_t=100,  # log every t steps in denoising sampling
            logvar_init=0.,
        ),
        conditioning_key='crossattn',  # 'concat'
        cond_stage_key='slots',
    ):
        vae_dict['scale_factor'] = diffusion_dict.pop('z_scale_factor', 1.)
        super().__init__(
            resolution=resolution,
            unet_dict=unet_dict,
            use_ema=use_ema,
            diffusion_dict=diffusion_dict,
            conditioning_key=conditioning_key,
            cond_stage_key=cond_stage_key,
        )

        vae_type = vae_dict.pop('vae_type')
        assert vae_type == 'VQVAE', f'VAE type {vae_type} not implemented.'
        from ..vqvae import VQVAEWrapper
        self.vae = VQVAEWrapper(**vae_dict)

        self.clip_denoised = False  # latent features are unbounded values
        self.vq_denoised = True  # LDM uses this by default

    def loss_function(self, data_dict):
        """VAE extract features --> add noise to feature map --> denoise."""
        img = data_dict['img']  # [B, C, H, W]
        # extract feature map
        with torch.no_grad():
            x0 = self.vae.encode(img).detach()  # [B, C, h, w]
        t = torch.randint(
            0, self.num_timesteps, (x0.shape[0], ), device=self.device).long()
        context = data_dict[self.cond_stage_key]  # [B, N, D]
        noise = torch.randn_like(x0)
        x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
        # condition!
        pred = self.forward(x_noisy, t, context=context)
        if self.pred_target == 'eps':
            gt = noise
        # follow https://arxiv.org/pdf/2202.00512.pdf in predicting v
        elif self.pred_target == 'v':
            alpha_t = extract_to(self.sqrt_alphas_bar, t, x0.shape)
            sigma_t = extract_to(self.sqrt_one_minus_alphas_bar, t, x0.shape)
            gt = alpha_t * noise - sigma_t * x0
        else:
            gt = x0
        gt = gt.detach()
        loss_dict = {'denoise_loss': F.mse_loss(pred, gt)}
        return loss_dict

    @torch.no_grad()
    def log_images(
        self,
        batch,
        ret_intermed=True,
        **kwargs,
    ):
        """Need to decode latent features to images."""
        log = {}
        img = batch['img']
        # extract features
        x = self.vae.encode(img)  # [B, C, h, w]
        cond = batch[self.cond_stage_key]
        B = x.shape[0]
        same_noise = kwargs.get('same_noise', False)

        # get diffusion row
        if ret_intermed:
            diffusion_row = [self.vae.decode(x.detach().clone())]
            x0 = x
            for t in range(self.num_timesteps):
                if (t + 1) % self.log_every_t == 0 or \
                        t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=B)
                    t = t.to(self.device).long()
                    noise = noise_like(x0.shape, x0.device, repeat=same_noise)
                    x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
                    diffusion_row.append(self.vae.decode(x_noisy))
            diffusion_row = torch.stack(diffusion_row, dim=0)  # [N, B, C, H,W]
            log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # get denoise row
        with self.ema_scope("Plotting"):
            ret = self.generate_imgs(
                cond=cond, batch_size=B, ret_intermed=ret_intermed, **kwargs)

        # decode patch tokens to images
        if isinstance(ret, tuple):
            log["samples"] = self.vae.decode(ret[0])
            if ret[1] is not None:
                denoise_row = torch.stack(
                    [self.vae.decode(feats) for feats in ret[1]], dim=0)
                log["denoise_row"] = self._get_rows_from_list(denoise_row)
        else:
            log["samples"] = self.vae.decode(ret)

        return log

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # fix the 1st stage VAE
        self.vae.eval()
        return self
