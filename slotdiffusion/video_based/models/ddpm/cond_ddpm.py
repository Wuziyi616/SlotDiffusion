"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

from tqdm import tqdm

import torch
import torch.nn.functional as F
from einops import repeat

from .dpm_solver import NoiseScheduleVP, model_wrapper, DPM_Solver
from .ddim import DDIMSampler
from .ddpm import DDPM
from .utils import noise_like, extract_to


class CondDDPM(DDPM):
    """Conditional DDPM in image space."""

    def __init__(
        self,
        resolution,
        unet_dict,
        use_ema=True,
        diffusion_dict=dict(
            pred_target='eps',  # 'eps', 'x0', 'v'
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
        super().__init__(
            resolution=resolution,
            unet_dict=unet_dict,
            use_ema=use_ema,
            diffusion_dict=diffusion_dict,
            conditioning_key=conditioning_key,
        )

        assert conditioning_key and cond_stage_key
        self.cond_stage_key = cond_stage_key

        self.vq_denoised = False  # useless for image space DDPM

    def _p_mean_variance(self, x, t, c, clip_denoised=True, vq_denoised=False):
        """
        Get the distribution p(x_{t-1} | x_t), conditioned on `c`.
        First estimate x0, then use x0 and x_t to compute x_{t-1}.
        """
        if self.pred_target == 'eps':
            eps = self.forward(x, t, context=c)
            x0_recon = self._predict_x0_from_xt(x, t=t, noise=eps)
        elif self.pred_target == 'v':
            v = self.forward(x, t, context=c)
            alpha_t = extract_to(self.sqrt_alphas_bar, t, v.shape)
            sigma_t = extract_to(self.sqrt_one_minus_alphas_bar, t, v.shape)
            x0_recon = alpha_t * x - sigma_t * v
        else:
            x0_recon = self.forward(x, t, context=c)
        # clip x0 to [-1., 1.]
        if clip_denoised:
            x0_recon.clamp_(-1., 1.)
        # use 1st stage VQVAE quantize to denoise
        if vq_denoised:
            x0_recon = self.vae.quantize(x0_recon)

        model_mean, posterior_variance, posterior_log_variance = \
            self._q_posterior(x0=x0_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def _p_sample(
        self,
        x,
        t,
        c,
        clip_denoised=True,
        vq_denoised=False,
        same_noise=False,
    ):
        """Sample x_{t-1} from p(x_{t-1} | x_t) conditioned on context `c`."""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self._p_mean_variance(
            x, t, c, clip_denoised=clip_denoised, vq_denoised=vq_denoised)
        noise = noise_like(x.shape, device, repeat=same_noise)
        # no noise when t == 0
        mask = (1 - (t == 0).float()).reshape(b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def _sample_x0_from_noise(
        self,
        cond,
        shape,
        ret_intermed=False,
        verbose=True,
        same_noise=False,
    ):
        """Generate x0 conditioned on `cond` by sampling from random noise."""
        device = self.betas.device
        b = shape[0]  # shape is [B, C, H, W]
        img = noise_like(shape, device, repeat=same_noise)
        intermediates = [img]
        for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc='Cond Sampling t',
                total=self.num_timesteps,
                disable=not verbose):
            img = self._p_sample(
                img,
                torch.full((b, ), i, device=device, dtype=torch.long),
                c=cond,
                clip_denoised=self.clip_denoised,
                vq_denoised=self.vq_denoised,
                same_noise=same_noise,
            )
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if ret_intermed:
            intermediates = torch.stack(intermediates, dim=0)  # [N, B, ...]
            return img, intermediates
        return img

    @torch.no_grad()
    def generate_imgs(
        self,
        cond,
        batch_size=16,
        ret_intermed=False,
        verbose=True,
        use_ddim=False,
        use_dpm=False,
        same_noise=False,
    ):
        """Generate `batch_size` images conditioned on `cond`."""
        assert cond is not None, 'Please provide conditioning data!'
        assert not (self.clip_denoised and self.vq_denoised), \
            'clip_denoised is image space DM, vq_denoised is LDM'
        # cond should be [B, N, C]
        # if no batch-dim, we duplicate it
        if len(cond.shape) == 2:
            cond = repeat(cond, 'n c -> b n c', b=batch_size)
        shape = (batch_size, self.channels, *self.resolution)
        # fastest sampling with DPMSolver
        if use_dpm:
            dpm_steps = max(20, self.num_timesteps // 50)
            noise_schedule = NoiseScheduleVP(betas=self.betas)
            # hack for DPM-Solver
            if hasattr(self, 'vae'):
                self.model.vae = self.vae
            if self.pred_target == 'eps':
                model_type = 'noise'
            elif self.pred_target == 'v':
                model_type = 'v'
            else:
                model_type = 'x_start'
            model_fn = model_wrapper(
                model=self.model,
                noise_schedule=noise_schedule,
                model_type=model_type,
                guidance_type='classifier-free',
                condition=cond,
            )
            dpm_sampler = DPM_Solver(
                model_fn,
                noise_schedule,
                algorithm_type='dpmsolver++',
                correcting_x0_fn=self.clip_denoised,
                vq_denoised=self.vq_denoised,
            )
            x0 = noise_like(shape, self.device, repeat=same_noise)
            ret = dpm_sampler.sample(
                x0,
                steps=dpm_steps,
                order=3,
                method='singlestep',
                return_intermediate=ret_intermed,
                verbose=verbose,
            )
            # hack for DPM-Solver
            if hasattr(self, 'vae'):
                self.model.vae = None
            return ret
        # faster sampling with DDIM
        if use_ddim:
            ddim_steps = max(200, self.num_timesteps // 5)
            ddim_sampler = DDIMSampler(self, schedule=self.beta_schedule)
            return ddim_sampler.generate_imgs(
                ddim_steps,
                shape,
                conditioning=cond,
                verbose=verbose,
                ret_intermed=ret_intermed,
                same_noise=same_noise,
            )
        return self._sample_x0_from_noise(
            cond=cond,
            shape=shape,
            ret_intermed=ret_intermed,
            verbose=verbose,
            same_noise=same_noise,
        )

    def loss_function(self, data_dict):
        """Make a noisy sample, then use model to predict the noise."""
        x0 = data_dict['img']  # [B, C, H, W]
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
        log = {}
        x = batch['img']
        cond = batch[self.cond_stage_key]
        B = x.shape[0]
        same_noise = kwargs.get('same_noise', False)

        # get diffusion row
        if ret_intermed:
            diffusion_row = [x.detach().clone()]
            x0 = x
            for t in range(self.num_timesteps):
                if (t + 1) % self.log_every_t == 0 or \
                        t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=B)
                    t = t.to(self.device).long()
                    noise = noise_like(x0.shape, x0.device, repeat=same_noise)
                    x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
                    diffusion_row.append(x_noisy)
            diffusion_row = torch.stack(diffusion_row, dim=0)  # [N, B, C, H,W]
            log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # get denoise row
        with self.ema_scope("Plotting"):
            ret = self.generate_imgs(
                cond=cond, batch_size=B, ret_intermed=ret_intermed, **kwargs)

        if isinstance(ret, tuple):
            log["samples"] = ret[0]
            if ret[1] is not None:
                log["denoise_row"] = self._get_rows_from_list(ret[1])
        else:
            log["samples"] = ret

        return log
