import torch
import torch.nn.functional as F

from nerv.training import BaseModel

from .slotformer import SlotFormer

from slotdiffusion.video_based.models.ddpm.ldm import LDM
from slotdiffusion.video_based.models.ddpm.cond_ddpm import CondDDPM


class LDMSlotFormer(SlotFormer):
    """SlotFormer with LDM-based slot decoder."""

    def __init__(
        self,
        resolution,
        clip_len,
        slot_dict=dict(  # from DM
            # 3-10 objects on MOVi-C
            num_slots=2,
            slot_size=10,
            slot_mlp_size=10 * 2,
            num_iterations=2,
        ),
        dec_dict=dict(
            resolution=(32, 32),
            vae_dict=dict(
                vae_type='VQVAE',
                enc_dec_dict=dict(
                    resolution=32,
                    in_channels=3,
                    z_channels=3,
                    ch=64,  # base_channel
                    ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
                    num_res_blocks=2,
                    attn_resolutions=[],
                    out_ch=3,
                    dropout=0.0,
                ),
                vq_dict=dict(
                    n_embed=12,  # vocab_size
                    embed_dim=3,
                    percept_loss_w=1.0,
                ),
                vqvae_ckp_path='',
            ),
            unet_dict=dict(
                in_channels=3,
                model_channels=64,
                out_channels=3,
                num_res_blocks=1,
                attention_resolutions=(8, 4, 2),
                dropout=0,
                channel_mult=(1, 2, 4, 8),
                conv_resample=True,
                dims=2,
                use_checkpoint=False,
                num_head_channels=32,
                resblock_updown=False,
                transformer_depth=1,  # custom transformer support
                context_dim=10,  # custom transformer support
                n_embed=
                None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            ),
            use_ema=True,
            diffusion_dict=dict(
                pred_target='eps',  # 'eps' or 'x0', predict noise or direct x0
                timesteps=2,
                beta_schedule="linear",
                linear_start=1e-4,
                linear_end=2e-2,
                cosine_s=8e-3,
                log_every_t=100,  # log every t steps in denoising sampling
                logvar_init=0.,
            ),
            conditioning_key='crossattn',  # 'concat'
            cond_stage_key='slots',
        ),  # decoder is latent diffusion model
        rollout_dict=dict(
            num_slots=2,
            slot_size=10,
            history_len=6,
            t_pe='sin',
            slots_pe='',
            d_model=10,
            num_layers=4,
            num_heads=5,
            ffn_dim=14,
            norm_first=True,
        ),
        loss_dict=dict(
            rollout_len=6,
            use_img_recon_loss=False,
            use_denoise_loss=False,  # the only line from DM configuration
        ),
        eps=1e-6,
    ):
        super().__init__(
            resolution=resolution,
            clip_len=clip_len,
            slot_dict=slot_dict,
            dec_dict=dec_dict,
            rollout_dict=rollout_dict,
            loss_dict=loss_dict,
            eps=eps,
        )

    def _build_decoder(self):
        """Build DM decoder."""
        weight_path = self.dec_dict.pop("dec_ckp_path")
        # LDM decoder
        if self.dec_dict.get('vae_dict', dict()):
            self.dm_decoder = LDM(**self.dec_dict)
        # Conditional DDPM
        else:
            self.dm_decoder = CondDDPM(**self.dec_dict)

        w = torch.load(weight_path, map_location='cpu')

        if 'state_dict' in w:
            w = w['state_dict']

        w = {k[11:]: v for k, v in w.items() if 'dm_decoder' in k}
        self.dm_decoder.load_state_dict(w)

        for p in self.dm_decoder.parameters():
            p.requires_grad = False

    def rollout(self, past_slots, pred_len, decode=False, with_gt=True):
        """Perform future rollout to `target_len` video."""
        pred_slots = self.rollouter(past_slots[:, -self.history_len:],
                                    pred_len)
        # `decode` is usually called from outside
        # used to visualize an entire video (burn-in + rollout)
        # i.e. `with_gt` is True
        if decode:
            if with_gt:
                slots = torch.cat([past_slots, pred_slots], dim=1)
            else:
                slots = pred_slots
            B, T = slots.shape[:2]
            log_dict = self.log_images(
                {'slots': slots.flatten(0, 1)}, use_dpm=True, same_noise=True
            )  # no flattening to slots because data_dict requires "B, T, N, C"
            out_dict = {
                'recon_combined': log_dict['samples'].unflatten(0, (B, T)),
                'slots': slots
            }
            return out_dict
        return pred_slots  # [B, pred_len, N, C]

    def forward(self, data_dict):
        """Forward pass."""
        slots = data_dict['slots']  # [B, T', N, C]
        assert self.rollout_len + self.history_len == slots.shape[1], \
            f'wrong SlotFormer training length {slots.shape[1]}'
        past_slots = slots[:, :self.history_len]
        gt_slots = slots[:, self.history_len:]
        pred_slots = self.rollout(past_slots, self.rollout_len)
        out_dict = {
            'gt_slots': gt_slots,  # both slots [B, pred_len, N, C]
            'pred_slots': pred_slots,
        }
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        gt_slots = out_dict['gt_slots']
        pred_slots = out_dict['pred_slots']
        slot_recon_loss = F.mse_loss(pred_slots, gt_slots)
        loss_dict = {'slot_recon_loss': slot_recon_loss}
        if self.use_denoise_loss:
            ddpm_dict = {
                'img': data_dict['img'].flatten(0, 1),
                'slots': out_dict['slots'].flatten(0, 1),
            }
            loss_dict.update(self.dm_decoder.loss_function(ddpm_dict))
        return loss_dict

    def train(self, mode=True):
        BaseModel.train(self, mode)
        # keep decoder part in eval mode
        self.dm_decoder.eval()
        return self

    @torch.no_grad()
    def log_images(self, data_dict, **kwargs):
        """Call the diffusion model's `log_images()` function."""
        slots = data_dict['slots']  # [T, N, C]
        # B, T = slots.shape[:2]
        T = slots.shape[0]

        if isinstance(self.dm_decoder, LDM):
            ret = self.dm_decoder.generate_imgs(
                slots, batch_size=T, **kwargs)  # [T, C, H, W]
            if isinstance(ret, tuple):
                generated_z = ret[0]
            else:
                generated_z = ret
            decoded_img = self.dm_decoder.vae.decode(
                generated_z)  # assume output is not a tuple
            log_dict = {'samples': decoded_img}
            return log_dict  # a dictionary with only "samples"
        else:
            log_imgs = self.dm_decoder.generate_imgs(
                slots, batch_size=T, **kwargs)
            log_dict = {'samples': log_imgs}
            return log_dict
