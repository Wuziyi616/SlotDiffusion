import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from nerv.training import BaseModel

from .modules import Encoder, Decoder
from .quantize import VectorQuantizer2 as VectorQuantizer
from .loss import VQLPIPSLoss


def temporal_wrapper(func):
    """A wrapper to make the model compatible with both 4D and 5D inputs."""

    def f(cls, x):
        """x is either [B, C, H, W] or [B, T, C, H, W]."""
        B = x.shape[0]
        if len(x.shape) == 5:
            unflatten = True
            x = x.flatten(0, 1)
        else:
            unflatten = False

        outs = func(cls, x)

        if unflatten:
            if isinstance(outs, tuple):
                outs = [o.unflatten(0, (B, -1)) if o.ndim else o for o in outs]
                return tuple(outs)
            else:
                return outs.unflatten(0, (B, -1))
        else:
            return outs

    return f


class VQVAE(BaseModel):
    """VQ-VAE consisting of Encoder, QuantizationLayer and Decoder."""

    def __init__(
        self,
        enc_dec_dict=dict(
            resolution=128,
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
            n_embed=4096,  # vocab_size
            embed_dim=3,  # same as `z_channels`
            percept_loss_w=1.0,
        ),
        use_loss=True,
    ):
        super().__init__()

        self.resolution = enc_dec_dict['resolution']
        self.embed_dim = vq_dict['embed_dim']
        self.n_embed = vq_dict['n_embed']
        self.z_ch = enc_dec_dict['z_channels']

        self.encoder = Encoder(**enc_dec_dict)
        self.decoder = Decoder(**enc_dec_dict)

        self.quantize = VectorQuantizer(
            self.n_embed,
            self.embed_dim,
            beta=0.25,
            sane_index_shape=True,
        )
        self.quant_conv = nn.Conv2d(self.z_ch, self.embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(self.embed_dim, self.z_ch, 1)

        if use_loss:
            self.loss = VQLPIPSLoss(percept_loss_w=vq_dict['percept_loss_w'])

    @temporal_wrapper
    def encode_quantize(self, x):
        """Encode image to pre-VQ features, then quantize."""
        h = self.encode(x)  # `embed_dim`
        quant, quant_loss, (_, _, quant_idx) = self.quantize(h)
        # [B, `embed_dim`, h, w], scalar, [B*h*w]
        return quant, quant_loss, quant_idx

    @temporal_wrapper
    def encode(self, x):
        """Encode image to pre-VQ features."""
        # this is the x0 in LDM!
        h = self.encoder(x)  # `z_ch`
        h = self.quant_conv(h)  # `embed_dim`
        return h

    @temporal_wrapper
    def quantize_decode(self, h):
        """Input pre-VQ features, quantize and decode to reconstruct."""
        # use this to reconstruct images from LDM's denoised output!
        quant, _, _ = self.quantize(h)
        dec = self.decode(quant)
        return dec

    @temporal_wrapper
    def decode(self, quant):
        """Input already quantized features, do reconstruction."""
        quant = self.post_quant_conv(quant)  # `z_ch`
        dec = self.decoder(quant)
        return dec

    def forward(self, data_dict):
        img = data_dict['img']
        quant, quant_loss, token_id = self.encode_quantize(img)
        recon = self.decode(quant)
        out_dict = {
            'recon': recon,
            'token_id': token_id,
            'quant_loss': quant_loss,
        }
        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute training loss."""
        img = data_dict['img']
        recon = out_dict['recon']
        quant_loss = out_dict['quant_loss']

        loss_dict = self.loss(quant_loss, img, recon)

        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = self.calc_train_loss(data_dict, out_dict)
        img = data_dict['img']
        recon = out_dict['recon']
        loss_dict['recon_mse'] = F.mse_loss(recon, img)
        return loss_dict

    @property
    def dtype(self):
        return self.quant_conv.weight.dtype

    @property
    def device(self):
        return self.quant_conv.weight.device


class VQVAEWrapper(nn.Module):
    """A wrapper for LDM with VQVAE."""

    def __init__(self, enc_dec_dict, vq_dict, vqvae_ckp_path, scale_factor=1.):
        super().__init__()

        self.vqvae = VQVAE(
            enc_dec_dict=enc_dec_dict,
            vq_dict=vq_dict,
            use_loss=False,
        ).eval()
        # load pre-trained weight
        if os.path.exists(vqvae_ckp_path):
            ckp = torch.load(vqvae_ckp_path, map_location='cpu')
            if 'state_dict' in ckp:
                ckp = ckp['state_dict']
            # remove 'loss.' keys
            ckp = {k: v for k, v in ckp.items() if not k.startswith('loss.')}
            self.vqvae.load_state_dict(ckp)
        else:
            print(f'Warning: VQ-VAE weight not found at {vqvae_ckp_path}!!!')
        # freeze the VQVAE
        for p in self.vqvae.parameters():
            p.requires_grad = False

        # scale the Z by dividing this value
        self.scale_factor = scale_factor

    def encode(self, x):
        return self.vqvae.encode(x) / self.scale_factor

    def decode(self, h, quantize=True):
        h = h * self.scale_factor
        if quantize:
            return self.vqvae.quantize_decode(h)
        return self.vqvae.decode(h)

    def quantize(self, h):
        h = h * self.scale_factor
        return self.vqvae.quantize(h)[0] / self.scale_factor
