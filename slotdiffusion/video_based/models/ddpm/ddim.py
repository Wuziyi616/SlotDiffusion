"""SAMPLING ONLY."""

import numpy as np
from tqdm import tqdm

import torch

from .utils import make_ddim_sampling_parameters, make_ddim_timesteps, \
    noise_like


class DDIMSampler(object):

    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()

        self.model = model
        self.schedule = schedule

        self.ddpm_num_timesteps = model.num_timesteps
        self.pred_target = model.pred_target  # 'eps', 'x0', 'v'
        assert self.pred_target in ['eps', 'x0'], \
            f'pred_target {self.pred_target} not supported by DDIM'

        self.clip_denoised = self.model.clip_denoised
        self.vq_denoised = self.model.vq_denoised
        assert not (self.clip_denoised and self.vq_denoised), \
            'clip_denoised is image space DM, vq_denoised is LDM'

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(
        self,
        ddim_num_steps,
        ddim_discretize="uniform",
        ddim_eta=0.,
        verbose=True,
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_bar = self.model.alphas_bar
        to_torch = lambda x: x.clone().detach().\
            to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_bar', to_torch(alphas_bar))
        self.register_buffer('alphas_bar_prev',
                             to_torch(self.model.alphas_bar_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_bar',
                             to_torch(np.sqrt(alphas_bar.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_bar',
                             to_torch(np.sqrt(1. - alphas_bar.cpu())))
        self.register_buffer('log_one_minus_alphas_bar',
                             to_torch(np.log(1. - alphas_bar.cpu())))
        self.register_buffer('sqrt_recip_alphas_bar',
                             to_torch(np.sqrt(1. / alphas_bar.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_bar',
                             to_torch(np.sqrt(1. / alphas_bar.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = \
            make_ddim_sampling_parameters(
                alphacums=alphas_bar.cpu(),
                ddim_timesteps=self.ddim_timesteps,
                eta=ddim_eta,
                verbose=verbose,
            )
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas',
                             np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_bar_prev) / (1 - self.alphas_bar) *
            (1 - self.alphas_bar / self.alphas_bar_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps',
                             sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def generate_imgs(
        self,
        steps,
        shape,
        conditioning=None,
        eta=0.,
        verbose=True,
        log_every_t=100,
        ret_intermed=True,
        same_noise=False,
        **kwargs,
    ):
        if conditioning is not None:
            assert conditioning.shape[0] == shape[0]

        self.make_schedule(ddim_num_steps=steps, ddim_eta=eta, verbose=False)

        # sampling
        if verbose:
            print(f'Data shape for DDIM sampling is {shape}, eta {eta}')
        samples, intermediates = self._sample_x0_from_noise(
            conditioning,
            shape,
            use_original_steps=False,
            verbose=verbose,
            log_every_t=log_every_t,
            same_noise=same_noise,
        )
        if ret_intermed:
            return samples, intermediates
        return samples

    @torch.no_grad()
    def _sample_x0_from_noise(
        self,
        cond,
        shape,
        use_original_steps=False,
        verbose=True,
        log_every_t=100,
        same_noise=False,
    ):
        device = self.model.device
        timesteps = self.ddpm_num_timesteps if \
            use_original_steps else self.ddim_timesteps
        b = shape[0]
        img = noise_like(shape, device, repeat=same_noise)

        intermediates = [img]
        time_range = reversed(range(
            0, timesteps)) if use_original_steps else np.flip(timesteps)
        total_steps = timesteps if use_original_steps else timesteps.shape[0]

        if verbose:
            print(f"Running DDIM Sampling with {total_steps} timesteps")
        iterator = tqdm(
            time_range,
            desc='DDIM Sampler',
            total=total_steps,
            disable=not verbose)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b, ), step, device=device, dtype=torch.long)

            outs = self._p_sample_ddim(
                img,
                cond,
                ts,
                index=index,
                use_original_steps=use_original_steps,
                same_noise=same_noise,
            )
            img, pred_x0 = outs

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates.append(img)

        return img, torch.stack(intermediates, dim=0)  # [N, B, ...]

    @torch.no_grad()
    def _p_sample_ddim(
        self,
        x,
        c,
        t,
        index,
        use_original_steps=False,
        same_noise=False,
    ):
        b, *_, device = *x.shape, x.device

        alphas = self.model.alphas_bar if \
            use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_bar_prev if \
            use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_bar if \
            use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if \
            use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1),
                                       sqrt_one_minus_alphas[index],
                                       device=device)

        if self.pred_target == 'eps':
            # predict the noise
            e_t = self.model(x, t, context=c)
            # estimate x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            # predict x_0
            pred_x0 = self.model(x, t, context=c)
            # estimate the noise
            e_t = (x - a_t.sqrt() * pred_x0) / sqrt_one_minus_at
        # clip x0 to [-1., 1.]
        if self.clip_denoised:
            pred_x0.clamp_(-1., 1.)
        # use 1st stage VQVAE quantize to denoise
        if self.vq_denoised:
            pred_x0 = self.model.vae.quantize(pred_x0)
        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat=same_noise)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0
