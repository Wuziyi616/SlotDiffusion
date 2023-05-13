"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import numpy as np
from tqdm import tqdm
from functools import partial
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid
from einops import rearrange, repeat

from .ddim import DDIMSampler
from .ema import LitEma
from .utils import default, is_list_tuple, make_beta_schedule, extract_to, \
    noise_like
from ..unet import UNetModel


class DDPM(nn.Module):
    """Classic DDPM with Gaussian diffusion, in image space."""

    def __init__(
            self,
            resolution,
            unet_dict,
            use_ema=True,
            diffusion_dict=dict(
                pred_target='eps',  # 'eps' or 'x0', predict noise or direct x0
                timesteps=1000,
                beta_schedule="linear",
                linear_start=1e-4,
                linear_end=2e-2,
                cosine_s=8e-3,
                log_every_t=100,  # log every t steps in denoising sampling
                logvar_init=0.,
            ),
            conditioning_key=None,  # 'concat', 'crossattn'
    ):
        super().__init__()

        self.channels = unet_dict.get('in_channels', 3)  # RGB image
        self.resolution = resolution

        self.clip_denoised = True
        self.log_every_t = diffusion_dict.pop('log_every_t')

        self.model = DiffusionWrapper(unet_dict, conditioning_key)
        # count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        # sampling schedules
        self.register_schedule(**diffusion_dict)

        if not conditioning_key:
            assert self.pred_target == 'eps', \
                'should predict noise in unconditional DDPM'

    def register_schedule(
        self,
        pred_target='eps',
        beta_schedule="linear",
        timesteps=1000,
        linear_start=1e-4,
        linear_end=2e-2,
        cosine_s=8e-3,
        logvar_init=0.,
    ):
        assert pred_target in ['eps', 'x0']
        self.pred_target = pred_target

        self.beta_schedule = beta_schedule
        betas = make_beta_schedule(
            beta_schedule,
            timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
            cosine_s=cosine_s,
        )
        alphas = 1. - betas
        alphas_bar = np.cumprod(alphas, axis=0)
        alphas_bar_prev = np.append(1., alphas_bar[:-1])
        self.num_timesteps = betas.shape[0]

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_bar', to_torch(alphas_bar))
        self.register_buffer('alphas_bar_prev', to_torch(alphas_bar_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar', to_torch(np.sqrt(alphas_bar)))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             to_torch(np.sqrt(1. - alphas_bar)))
        self.register_buffer('log_one_minus_alphas_bar',
                             to_torch(np.log(1. - alphas_bar)))
        self.register_buffer('sqrt_recip_alphas_bar',
                             to_torch(np.sqrt(1. / alphas_bar)))
        self.register_buffer('sqrt_recipm1_alphas_bar',
                             to_torch(np.sqrt(1. / alphas_bar - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_bar_prev) / \
            (1. - alphas_bar)
        # above: equal to 1. / (1. / (1. - alpha_bar_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_variance_clipped',
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer(
            'posterior_mean_coef1',
            to_torch(betas * np.sqrt(alphas_bar_prev) / (1. - alphas_bar)))
        self.register_buffer(
            'posterior_mean_coef2',
            to_torch(
                (1. - alphas_bar_prev) * np.sqrt(alphas) / (1. - alphas_bar)))

        self.logvar = torch.full(
            fill_value=logvar_init, size=(self.num_timesteps, ))

    @contextmanager
    def ema_scope(self, context=None):
        """Switch model to EMA weight, do something, then switch back."""
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def _q_mean_variance(self, x0, t):
        """
        Get the distribution q(x_t | x_0).
        :param x0: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x0's shape.
        """
        mean = extract_to(self.sqrt_alphas_bar, t, x0.shape) * x0
        variance = extract_to(1. - self.alphas_bar, t, x0.shape)
        log_variance = extract_to(self.log_one_minus_alphas_bar, t, x0.shape)
        return mean, variance, log_variance

    def _sample_xt_from_x0(self, x0, t, noise=None):
        """Sample x_t from q(x_t | x_0)."""
        noise = default(noise, lambda: torch.randn_like(x0))
        return extract_to(self.sqrt_alphas_bar, t, x0.shape) * x0 + \
            extract_to(self.sqrt_one_minus_alphas_bar, t, x0.shape) * noise

    def _predict_x0_from_xt(self, x_t, t, noise):
        """Predict x_0 given x_t and eps(x_t)."""
        return extract_to(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t - \
            extract_to(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * noise

    def _q_posterior(self, x0, x_t, t):
        """Estimate x_{t-1} given x_0 and x_t."""
        posterior_mean = (
            extract_to(self.posterior_mean_coef1, t, x_t.shape) * x0 +
            extract_to(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract_to(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_to(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _p_mean_variance(self, x, t, clip_denoised: bool):
        """
        Get the distribution p(x_{t-1} | x_t).
        First estimate x0, then use x0 and x_t to compute x_{t-1}.
        """
        eps = self.forward(x, t)
        x0_recon = self._predict_x0_from_xt(x, t=t, noise=eps)
        if clip_denoised:
            x0_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = \
            self._q_posterior(x0=x0_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def _p_sample(self, x, t, clip_denoised=True):
        """Sample x_{t-1} from p(x_{t-1} | x_t)."""
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self._p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device)
        # no noise when t == 0
        mask = (1 - (t == 0).float()).reshape(b, *((1, ) * (len(x.shape) - 1)))
        return model_mean + mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def _sample_x0_from_noise(self, shape, ret_intermed=False, verbose=True):
        """Generate x0 from random noise by iterative sampling."""
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc='Sampling t',
                total=self.num_timesteps,
                disable=not verbose):
            img = self._p_sample(
                img,
                torch.full((b, ), i, device=device, dtype=torch.long),
                clip_denoised=self.clip_denoised,
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
        batch_size=16,
        ret_intermed=False,
        verbose=True,
        use_ddim=False,
    ):
        """Generate `batch_size` images."""
        shape = (batch_size, self.channels, *self.resolution)
        # faster sampling with DDIM
        if use_ddim:
            ddim_steps = max(200, self.num_timesteps // 5)
            ddim_sampler = DDIMSampler(self, schedule=self.beta_schedule)
            return ddim_sampler.generate_imgs(
                ddim_steps,
                shape,
                conditioning=None,
                verbose=verbose,
                ret_intermed=ret_intermed,
            )
        return self._sample_x0_from_noise(
            shape,
            ret_intermed=ret_intermed,
            verbose=verbose,
        )

    def loss_function(self, data_dict):
        """Make a noisy sample, then use model to predict the noise."""
        x0 = data_dict['img']
        t = torch.randint(
            0, self.num_timesteps, (x0.shape[0], ), device=self.device).long()
        noise = torch.randn_like(x0)
        x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
        # conditional data
        eps = self.forward(x_noisy, t)
        loss_dict = {'denoise_loss': F.mse_loss(eps, noise)}
        return loss_dict

    def forward(self, x, t, **kwargs):
        return self.model(x, t, **kwargs)

    def _training_step_end(self, *args, **kwargs):
        # accumulate the moving average of model weights
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        """Put the input samples NxBxcxhxw into a grid of B x N grid."""
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(
            denoise_grid, nrow=n_imgs_per_row, pad_value=-1.)
        return denoise_grid

    @torch.no_grad()
    def log_images(
        self,
        batch,
        **kwargs,
    ):
        log = {}
        x = batch['img']
        B = x.shape[0]

        # get diffusion row
        diffusion_row = [x.detach().clone()]
        x0 = x

        for t in range(self.num_timesteps):
            if (t + 1) % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=B)
                t = t.to(self.device).long()
                noise = torch.randn_like(x0)
                x_noisy = self._sample_xt_from_x0(x0=x0, t=t, noise=noise)
                diffusion_row.append(x_noisy)
        diffusion_row = torch.stack(diffusion_row, dim=0)  # [N, B, C, H, W]
        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        # get denoise row
        with self.ema_scope("Plotting"):
            samples, denoise_row = self.generate_imgs(
                batch_size=B, ret_intermed=True, **kwargs)

        log["samples"] = samples
        log["denoise_row"] = self._get_rows_from_list(denoise_row)

        return log

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype


class DiffusionWrapper(nn.Module):
    """A wrapper for forwarding the diffusion model. Cond/Unconditional one."""

    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()

        self.diffusion_model = UNetModel(**diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn']

    def forward(self, x, t, context=None):
        if self.conditioning_key is None:
            return self.diffusion_model(x, t)

        assert context is not None, 'Please provide conditioning data'
        if self.conditioning_key == 'concat':
            if not is_list_tuple(context):
                context = [context]
            xc = torch.cat([x] + context, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            if is_list_tuple(context):
                context = torch.cat(context, 1)
            out = self.diffusion_model(x, t, context=context)
        else:
            raise NotImplementedError

        return out

    @property
    def device(self):
        return self.diffusion_model.device

    @property
    def dtype(self):
        return self.diffusion_model.dtype
