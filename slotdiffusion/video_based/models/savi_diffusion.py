import torch
from torch import nn
from torch.nn import functional as F

from .savi import SlotAttention, SAVi
from .ddpm import CondDDPM
from .ddpm import LDM


class SlotAttentionWMask(SlotAttention):
    """Slot attention module that iteratively performs cross-attention.

    We return the last attention map from SA as the segmentation mask.
    """

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            slots (torch.Tensor): [B, num_slots, C] slot inits.
            masks (torch.Tensor): [B, num_slots, N] segmentation mask.
        """
        # `inputs` has shape [B, num_inputs, inputs_size].
        # `num_inputs` is actually the spatial dim of feature map (H*W)
        bs, num_inputs, inputs_size = inputs.shape
        inputs = self.norm_inputs(inputs)  # Apply layer norm to the input.
        # Shape: [B, num_inputs, slot_size].
        k = self.project_k(inputs)
        # Shape: [B, num_inputs, slot_size].
        v = self.project_v(inputs)

        # Initialize the slots. Shape: [B, num_slots, slot_size].
        assert len(slots.shape) == 3

        # Multiple rounds of attention.
        for attn_iter in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

            # attn_map normalized along slot-dim is treated as seg_mask
            if attn_iter == self.num_iterations - 1:
                seg_mask = attn.detach().clone().permute(0, 2, 1)

            # Normalize along spatial dim and do weighted mean.
            attn = attn + self.eps
            attn = attn / torch.sum(attn, dim=1, keepdim=True)
            updates = torch.einsum('bnm,bnc->bmc', attn, v)
            # `updates` has shape: [B, num_slots, slot_size].

            # Slot update.
            # GRU is expecting inputs of size (N, L)
            # so flatten batch and slots dimension
            slots = self.gru(
                updates.view(bs * self.num_slots, self.slot_size),
                slots_prev.view(bs * self.num_slots, self.slot_size),
            )
            slots = slots.view(bs, self.num_slots, self.slot_size)
            slots = slots + self.mlp(slots)

        # [B, num_slots, slot_size], [B, num_slots, num_inputs (H*W)]
        return slots, seg_mask


class SAViDiffusion(SAVi):
    """SAVi model that operates on videos."""

    def __init__(
            self,
            resolution,
            clip_len,
            slot_dict=dict(
                num_slots=7,
                slot_size=128,
                slot_mlp_size=256,
                num_iterations=2,
            ),
            enc_dict=dict(
                enc_channels=(3, 64, 64, 64, 64),
                enc_ks=5,
                enc_out_channels=128,
                enc_norm='',
            ),
            dec_dict=dict(
                resolution=(128, 128),
                vae_dict=dict(),
                unet_dict=dict(),
                use_ema=True,
                diffusion_dict=dict(
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
            ),
            pred_dict=dict(
                pred_type='transformer',
                pred_rnn=True,
                pred_norm_first=True,
                pred_num_layers=2,
                pred_num_heads=4,
                pred_ffn_dim=512,
                pred_sg_every=None,
            ),
            loss_dict=dict(use_denoise_loss=True, ),
            eps=1e-6,
    ):
        super().__init__(
            resolution=resolution,
            clip_len=clip_len,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            pred_dict=pred_dict,
            loss_dict=loss_dict,
            eps=eps,
        )

    def _build_slot_attention(self):
        # Build SlotAttention module
        # kernels x img_feats --> posterior_slots
        self.enc_out_channels = self.enc_dict['enc_out_channels']
        self.num_slots = self.slot_dict['num_slots']
        self.slot_size = self.slot_dict['slot_size']
        self.slot_mlp_size = self.slot_dict['slot_mlp_size']
        self.num_iterations = self.slot_dict['num_iterations']

        # directly use learnable embeddings for each slot
        self.init_latents = nn.Parameter(
            nn.init.normal_(torch.empty(1, self.num_slots, self.slot_size)))

        self.slot_attention = SlotAttentionWMask(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_decoder(self):
        # Build Decoder
        # if has a VAE config, then it's LDM
        if self.dec_dict.get('vae_dict', dict()):
            self.dm_decoder = LDM(**self.dec_dict)
        # Conditional DDPM
        else:
            self.dm_decoder = CondDDPM(**self.dec_dict)

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_denoise_loss = self.loss_dict['use_denoise_loss']
        assert self.use_denoise_loss

    def encode(self, img, prev_slots=None):
        """Encode from img to slots."""
        B, T, C, H, W = img.shape
        img = img.flatten(0, 1)

        encoder_out = self._get_encoder_out(img)
        encoder_out = encoder_out.unflatten(0, (B, T))
        # `encoder_out` has shape: [B, T, H*W, out_features]

        # init slots
        init_latents = self.init_latents.repeat(B, 1, 1)  # [B, N, C]

        # apply SlotAttn on video frames via reusing slots
        all_slots, all_masks = [], []
        for idx in range(T):
            # init
            if prev_slots is None:
                latents = init_latents  # [B, N, C]
            else:
                latents = self.predictor(prev_slots)  # [B, N, C]

            # perform SA to get `slots` and `masks`
            slots, masks = self.slot_attention(encoder_out[:, idx], latents)
            all_slots.append(slots)
            all_masks.append(masks)

            # next timestep
            prev_slots = slots

        # (B, T, self.num_slots, self.slot_size)
        slots = torch.stack(all_slots, dim=1)
        # (B, T, self.num_slots, h, w)
        masks = torch.stack(all_masks, dim=1)
        masks = masks.unflatten(-1, self.visual_resolution)

        # resize masks to the original resolution
        if not self.training and self.visual_resolution != self.resolution:
            with torch.no_grad():
                masks = masks.flatten(0, 2).unsqueeze(1)  # [BTN, 1, h, w]
                masks = F.interpolate(
                    masks,
                    self.resolution,
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(1).unflatten(0, (B, T, self.num_slots))  # back

        # [B, T, N, C], [B, T, N, H, W]
        return slots, masks

    def forward(self, data_dict, **kwargs):
        """A wrapper for model forward.
        If the input video is too long in testing, we manually cut it.
        """
        # a hack to do image reconstruction in DP mode
        if kwargs.pop('log_images', False):
            return self.log_images(data_dict, **kwargs)
        # normal SAVi forward
        assert kwargs == {}
        return super().forward(data_dict)

    def _forward(self, img, prev_slots=None):
        """Forward function.

        Args:
            img: [B, T, C, H, W]
            prev_slots: [B, num_slots, slot_size] or None,
                the `slots` from last timestep.
        """
        # reset RNN states if this is the first frame
        if prev_slots is None:
            self._reset_rnn()

        # encode
        slots, masks = self.encode(img, prev_slots=prev_slots)
        # `slots` has shape: [B, T, self.num_slots, self.slot_size]
        # `masks` has shape: [B, T, self.num_slots, H, W]

        out_dict = {'masks': masks, 'slots': slots}
        if self.testing:
            return out_dict

        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        loss_dict = {}
        # DDPM conditions on slots to denoise the image
        # DDPM only runs on single images
        # so we flatten the temporal dimension
        if self.use_denoise_loss:
            ddpm_dict = {
                'img': data_dict['img'].flatten(0, 1),
                'slots': out_dict['slots'].flatten(0, 1),
            }
            loss_dict.update(self.dm_decoder.loss_function(ddpm_dict))
        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = super().calc_eval_loss(data_dict, out_dict)

        if not self.dm_decoder.use_ema:
            return loss_dict

        # since SA --> masks part is not EMA, no need to eval again
        with self.dm_decoder.ema_scope():
            ema_loss_dict = self.calc_train_loss(data_dict, out_dict)
        ema_loss_dict = {f'{k}_ema': v for k, v in ema_loss_dict.items()}
        loss_dict.update(ema_loss_dict)
        return loss_dict

    @torch.no_grad()
    def log_images(self, data_dict, **kwargs):
        """Call the DDPM's `log_images()` function."""
        out_dict = self.forward(data_dict)
        B, T = data_dict['img'].shape[:2]
        ddpm_dict = {
            'img': data_dict['img'].flatten(0, 1),
            'slots': out_dict['slots'].flatten(0, 1),
        }
        '''All are outputs from `vutils.make_grid()`:
            - 'diffusion_row': from x0 (input imgs) to xT (random noise)
            - 'denoise_row': from xT denoises to x0
            - 'samples': the final denoised (reconstructed) imgs
        '''
        log_dict = self.dm_decoder.log_images(ddpm_dict, **kwargs)
        log_dict = {k: v.unflatten(0, (B, T)) for k, v in log_dict.items()}
        log_dict['masks'] = out_dict['masks']
        return log_dict

    def _training_step_end(self, method=None):
        """Potential EMA step."""
        self.dm_decoder._training_step_end()
