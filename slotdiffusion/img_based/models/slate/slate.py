import os

import torch
from torch.nn import functional as F

from nerv.training import BaseModel

from ..utils import torch_cat
from ..slot_attention import SA
from ..sa_diffusion import SADiffusion
from .dVAE import dVAE
from .slate_transformer import SLATETransformerDecoder
from .slate_utils import gumbel_softmax, make_one_hot


class SLATE(SADiffusion):
    """SA model with TransformerDecoder predicting patch tokens."""

    def __init__(
        self,
        resolution,
        slot_dict=dict(
            num_slots=7,
            slot_size=128,
            slot_mlp_size=256,
            num_iterations=2,
        ),
        dvae_dict=dict(
            down_factor=4,
            vocab_size=4096,
            dvae_ckp_path='',
        ),
        enc_dict=dict(
            enc_channels=(3, 64, 64, 64, 64),
            enc_ks=5,
            enc_out_channels=128,
            enc_norm='',
        ),
        dec_dict=dict(
            dec_type='slate',
            dec_num_layers=4,
            dec_num_heads=4,
            dec_d_model=128,
        ),
        loss_dict=dict(
            use_img_recon_loss=False,  # dVAE decoded img recon loss
        ),
        eps=1e-6,
    ):
        self.dvae_dict = dvae_dict

        super().__init__(
            resolution=resolution,
            slot_dict=slot_dict,
            enc_dict=enc_dict,
            dec_dict=dec_dict,
            loss_dict=loss_dict,
            eps=eps,
        )

    def _build_dvae(self):
        # Build dVAE module
        self.vocab_size = self.dvae_dict['vocab_size']
        self.down_factor = self.dvae_dict['down_factor']
        self.dvae = dVAE(vocab_size=self.vocab_size, img_channels=3)
        ckp_path = self.dvae_dict['dvae_ckp_path']
        if os.path.exists(ckp_path):
            self.dvae.load_weight(ckp_path)
            print(f'Loading dVAE from {ckp_path}')
        else:
            print(f'Warning: dVAE weight not found at {ckp_path}!!!')
        # fix dVAE
        for p in self.dvae.parameters():
            p.requires_grad = False
        self.dvae.eval()

    def _build_decoder(self):
        # an ugly hack since we need some vars to build the TransformerDecoder
        self._build_dvae()
        # Build Decoder
        # GPT-style causal masking Transformer decoder
        H, W = self.resolution
        self.h, self.w = H // self.down_factor, W // self.down_factor
        self.num_patches = self.h * self.w
        max_len = self.num_patches - 1
        self.trans_decoder = SLATETransformerDecoder(
            vocab_size=self.vocab_size,
            d_model=self.dec_dict['dec_d_model'],
            n_head=self.dec_dict['dec_num_heads'],
            max_len=max_len,
            num_slots=self.num_slots,
            num_layers=self.dec_dict['dec_num_layers'],
        )

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_img_recon_loss = self.loss_dict['use_img_recon_loss']

    def forward(self, data_dict, **kwargs):
        """Forward function.

        Args:
            img: [B, C, H, W]
            img_token_id: [B, h*w], pre-computed dVAE tokenized img ids
        """
        # a hack to do image reconstruction in DP mode
        if kwargs.pop('recon_img', False):
            return self.recon_img(slots=data_dict['slots'], **kwargs)

        img, img_token_id = data_dict['img'], data_dict.get('token_id', None)

        # encode
        slots, masks = self.encode(img)
        # `slots` has shape: [B, self.num_slots, self.slot_size]
        # `masks` has shape: [B, self.num_slots, H, W]

        out_dict = {'slots': slots, 'masks': masks}
        if self.testing:
            return out_dict

        # tokenize the images to patches as Transformer inputs/targets
        if img_token_id is None:
            with torch.no_grad():
                img_token_id = self.dvae.tokenize(
                    img, one_hot=False).flatten(1, 2).detach().long()
        # [B, h*w]
        h, w = self.h, self.w

        # TransformerDecoder token prediction loss
        in_token_id = img_token_id[:, :-1]
        pred_token_id = self.trans_decoder(slots, in_token_id)[:, -(h * w):]
        # [B, h*w, vocab_size]
        out_dict.update({
            'pred_token_id': pred_token_id,
            'target_token_id': img_token_id,
        })

        # decode image for loss computing
        if self.use_img_recon_loss:
            out_dict['gt_img'] = img  # [B, C, H, W]
            logits = pred_token_id.transpose(2, 1).\
                unflatten(-1, (self.h, self.w)).contiguous()  # [B, S, h, w]
            z_logits = F.log_softmax(logits, dim=1)
            z = gumbel_softmax(z_logits, tau=0.1, hard=False, dim=1)
            recon_img = self.dvae.detokenize(z)  # [B, C, H, W]
            out_dict['recon_img'] = recon_img

        return out_dict

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        pred_token_id = out_dict['pred_token_id'].flatten(0, 1)
        target_token_id = out_dict['target_token_id'].flatten(0, 1)
        token_recon_loss = F.cross_entropy(pred_token_id, target_token_id)
        loss_dict = {'token_recon_loss': token_recon_loss}
        if self.use_img_recon_loss:
            gt_img = out_dict['gt_img']
            recon_img = out_dict['recon_img']
            recon_loss = F.mse_loss(recon_img, gt_img)
            loss_dict['img_recon_loss'] = recon_loss
        return loss_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        return SA.calc_eval_loss(self, data_dict, out_dict)

    @torch.no_grad()
    def recon_img(self, slots, bs=16, verbose=False):
        """Autoregressively reconstruct images from slots."""
        # slots: [B, num_slots, slot_size]
        imgs = []
        for batch_idx in range(0, slots.shape[0], bs):
            _, logits = self.trans_decoder.generate(
                slots[batch_idx:batch_idx + bs],
                steps=self.num_patches,
                sample=False,
                verbose=verbose,
            )
            # [T, patch_size**2, vocab_size] --> [T, vocab_size, h, w]
            logits = logits.transpose(2, 1).unflatten(
                -1, (self.h, self.w)).contiguous().cuda()
            # SLATE directly use ont-hot token (argmax) as input
            z_hard = make_one_hot(logits, dim=1)
            img = self.dvae.detokenize(z_hard)
            imgs.append(img.cpu())
            del logits, z_hard
            torch.cuda.empty_cache()

        imgs = torch.cat(imgs, dim=0)
        return imgs.to(self.device)  # DDP needs cuda for collection

    def train(self, mode=True):
        BaseModel.train(self, mode)
        # fix the dVAE
        self.dvae.eval()
        return self

    def _training_step_end(self, method=None):
        pass
