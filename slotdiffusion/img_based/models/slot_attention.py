import torch
from torch import nn
from torch.nn import functional as F

from nerv.training import BaseModel
from nerv.models import deconv_out_shape, conv_norm_act, deconv_norm_act

from .resnet import resnet18, resnet34
from .dino import dino
from .utils import assert_shape, SoftPositionEmbed
from .eval_utils import ARI_metric, fARI_metric, miou_metric, fmiou_metric, \
    mbo_metric


class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size**-0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

        # Slot update functions.
        self.gru = nn.GRUCell(self.slot_size, self.slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size),
        )

    def forward(self, inputs, slots):
        """Forward function.

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            updated slots, same shape as `slots`.
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
        for _ in range(self.num_iterations):
            slots_prev = slots

            # Attention. Shape: [B, num_slots, slot_size].
            q = self.project_q(slots)

            attn_logits = self.attn_scale * torch.einsum('bnc,bmc->bnm', k, q)
            attn = F.softmax(attn_logits, dim=-1)
            # `attn` has shape: [B, num_inputs, num_slots].

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

        return slots

    @property
    def dtype(self):
        return self.project_k.weight.dtype

    @property
    def device(self):
        return self.project_k.weight.device


class SA(BaseModel):
    """SA model that operates on images."""

    def __init__(
            self,
            resolution,
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
                dec_channels=(128, 64, 64, 64, 64),
                dec_resolution=(8, 8),
                dec_ks=5,
                dec_norm='',
            ),
            loss_dict=dict(use_img_recon_loss=True, ),
            eps=1e-6,
    ):
        super().__init__()

        self.resolution = resolution
        self.eps = eps

        self.slot_dict = slot_dict
        self.enc_dict = enc_dict
        self.dec_dict = dec_dict
        self.loss_dict = loss_dict

        self._build_slot_attention()
        self._build_encoder()
        self._build_decoder()
        self._build_loss()

        # a hack for only extracting slots
        self.testing = False

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

        self.slot_attention = SlotAttention(
            in_features=self.enc_out_channels,
            num_iterations=self.num_iterations,
            num_slots=self.num_slots,
            slot_size=self.slot_size,
            mlp_hidden_size=self.slot_mlp_size,
            eps=self.eps,
        )

    def _build_encoder(self):
        # Build Encoder
        # ResNet with less downsampling
        if self.enc_dict.get('resnet', False):
            use_layer4 = self.enc_dict['use_layer4']
            self.encoder = eval(self.enc_dict['resnet'])(
                small_inputs=True, use_layer4=use_layer4)
            if use_layer4:  # downsample by 8
                self.visual_resolution = tuple(i // 8 for i in self.resolution)
                self.visual_channels = 512
            else:  # downsample by 4
                self.visual_resolution = tuple(i // 4 for i in self.resolution)
                self.visual_channels = 256
        # using SSL pre-trained DINO encoder
        elif self.enc_dict.get('dino', False):
            print('Using DINO encoder!')
            patch_size = self.enc_dict['patch_size']
            small_size = self.enc_dict['small_size']

            if patch_size == 8:
                self.visual_resolution = tuple(i // 8 for i in self.resolution)
            elif patch_size == 16:
                self.visual_resolution = tuple(i // 16 for i in self.resolution)
            else:
                raise NotImplementedError('Patch size not supported! (8, 16)')
            if small_size:
                self.visual_channels = 384
            else:
                self.visual_channels = 768
            self.encoder = dino(patch_size=patch_size, small_size=small_size)
        # Conv CNN --> PosEnc --> MLP
        else:
            # CNN out visual resolution
            if self.resolution[0] > 64:
                self.visual_resolution = tuple(i // 2 for i in self.resolution)
                downsample = True
            else:
                self.visual_resolution = self.resolution
                downsample = False
            enc_channels = list(self.enc_dict['enc_channels'])  # CNN channels
            self.visual_channels = enc_channels[-1]  # CNN out visual channels

            enc_layers = len(enc_channels) - 1
            self.encoder = nn.Sequential(*[
                conv_norm_act(
                    enc_channels[i],
                    enc_channels[i + 1],
                    kernel_size=self.enc_dict['enc_ks'],
                    # 2x downsampling for 128x128 image
                    stride=2 if (i == 0 and downsample) else 1,
                    norm=self.enc_dict['enc_norm'],
                    act='relu' if i != (enc_layers - 1) else '',
                ) for i in range(enc_layers)
            ])  # relu except for the last layer

        # Build Encoder related modules
        self.encoder_pos_embedding = SoftPositionEmbed(self.visual_channels,
                                                       self.visual_resolution)
        self.encoder_out_layer = nn.Sequential(
            nn.LayerNorm(self.visual_channels),
            nn.Linear(self.visual_channels, self.enc_out_channels),
            nn.ReLU(),
            nn.Linear(self.enc_out_channels, self.enc_out_channels),
        )

    def _build_decoder(self):
        # Build Decoder
        # Spatial broadcast --> PosEnc --> DeConv CNN
        dec_channels = self.dec_dict['dec_channels']  # CNN channels
        self.dec_resolution = self.dec_dict['dec_resolution']  # broadcast size
        dec_ks = self.dec_dict['dec_ks']  # kernel size
        assert dec_channels[0] == self.slot_size, \
            'wrong in_channels for Decoder'
        modules = []
        in_size = self.dec_resolution
        out_size = in_size
        stride = 2
        for i in range(len(dec_channels) - 1):
            if out_size == self.resolution:
                stride = 1
            modules.append(
                deconv_norm_act(
                    dec_channels[i],
                    dec_channels[i + 1],
                    kernel_size=dec_ks,
                    stride=stride,
                    norm=self.dec_dict['dec_norm'],
                    act='relu',
                ))
            out_size = deconv_out_shape(out_size, stride, dec_ks // 2, dec_ks,
                                        stride - 1)

        assert_shape(
            self.resolution,
            out_size,
            message="Output shape of decoder did not match input resolution. "
                    "Try changing `decoder_resolution`.",
        )

        # out Conv for RGB and seg mask
        modules.append(
            nn.Conv2d(dec_channels[-1], 4, kernel_size=1, stride=1, padding=0))

        self.decoder = nn.Sequential(*modules)
        self.decoder_pos_embedding = SoftPositionEmbed(self.slot_size,
                                                       self.dec_resolution)

    def _build_loss(self):
        """Loss calculation settings."""
        self.use_img_recon_loss = self.loss_dict['use_img_recon_loss']
        assert self.use_img_recon_loss

    def _get_encoder_out(self, img):
        """Encode image, potentially add pos enc, apply MLP."""
        encoder_out = self.encoder(img).type(self.dtype)
        encoder_out = self.encoder_pos_embedding(encoder_out)
        # `encoder_out` has shape: [B, C, H, W]
        encoder_out = torch.flatten(encoder_out, start_dim=2, end_dim=3)
        # `encoder_out` has shape: [B, C, H*W]
        encoder_out = encoder_out.permute(0, 2, 1).contiguous()
        encoder_out = self.encoder_out_layer(encoder_out)
        # `encoder_out` has shape: [B, H*W, enc_out_channels]
        return encoder_out

    def encode(self, img, init_slots=None):
        """Encode from img to slots."""
        B, C, H, W = img.shape

        encoder_out = self._get_encoder_out(img)
        # `encoder_out` has shape: [B, H*W, out_features]

        # init slots
        if init_slots is None:
            init_slots = self.init_latents.repeat(B, 1, 1)

        # perform slot attention operation
        slots = self.slot_attention(encoder_out, init_slots)

        return slots  # [B, num_slots, slot_size]

    def forward(self, data_dict):
        """Forward function."""
        img = data_dict['img']

        # encode
        slots = self.encode(img)

        if self.testing:
            return {'slots': slots}

        # decode
        recon_img, recons, masks, _ = self.decode(slots)

        out_dict = {
            'recon_img': recon_img,
            'recons': recons,
            'masks': masks,
            'slots': slots,
        }
        return out_dict

    def decode(self, slots):
        """Decode from slots to reconstructed images and masks."""
        # `slots` has shape: [B, self.num_slots, self.slot_size].
        bs, num_slots, slot_size = slots.shape
        height, width = self.resolution
        num_channels = 3

        # spatial broadcast
        decoder_in = slots.view(bs * num_slots, slot_size, 1, 1)
        decoder_in = decoder_in.repeat(1, 1, self.dec_resolution[0],
                                       self.dec_resolution[1])

        out = self.decoder_pos_embedding(decoder_in)
        out = self.decoder(out)
        # `out` has shape: [B*num_slots, 4, H, W].

        out = out.view(bs, num_slots, num_channels + 1, height, width)
        recons = out[:, :, :num_channels, :, :]  # [B, num_slots, 3, H, W]
        masks = out[:, :, -1:, :, :]
        masks = F.softmax(masks, dim=1)  # [B, num_slots, 1, H, W]
        recon_combined = torch.sum(recons * masks, dim=1)  # [B, 3, H, W]
        return recon_combined, recons, masks, slots

    def calc_train_loss(self, data_dict, out_dict):
        """Compute loss that are general for SlotAttn models."""
        loss_dict = {}
        if self.use_img_recon_loss:
            recon_img = out_dict['recon_img']
            img = data_dict['img']
            loss_dict['img_recon_loss'] = F.mse_loss(recon_img, img)
        return loss_dict

    @torch.no_grad()
    def _eval_masks(self, pred_masks, gt_masks):
        """Calculate seg_mask related metrics.

        Args:
            pred_masks: [B, H, W], argmaxed predicted masks.
            gt_masks: [B, H, W], argmaxed GT
        """
        ari = ARI_metric(gt_masks, pred_masks)
        fari = fARI_metric(gt_masks, pred_masks)
        miou = miou_metric(gt_masks, pred_masks)
        fmiou = fmiou_metric(gt_masks, pred_masks)
        mbo = mbo_metric(gt_masks, pred_masks)
        ret_dict = {
            'ari': ari,
            'fari': fari,
            'miou': miou,
            'fmiou': fmiou,
            'mbo': mbo,
        }
        ret_dict = {
            k: torch.tensor(v).to(self.dtype).to(self.device)
            for k, v in ret_dict.items()
        }
        return ret_dict

    @torch.no_grad()
    def calc_eval_loss(self, data_dict, out_dict):
        """Loss computation in eval."""
        loss_dict = self.calc_train_loss(data_dict, out_dict)

        # in case GT masks are provided, calculate metrics
        if 'masks' in data_dict:
            pred_masks = out_dict['masks']  # [B, N(, 1), H, W]
            if len(pred_masks.shape) == 5:
                pred_masks = pred_masks.squeeze(-3)
            # proc_pred_masks = postproc_mask(pred_masks)  # [B, H, W]
            pred_masks = pred_masks.argmax(dim=-3)  # [B, H, W]
            gt_masks = data_dict['masks']  # [B, H, W], argmaxed GT sem_mask
            mask_dict = self._eval_masks(pred_masks, gt_masks.long())
            loss_dict.update(mask_dict)

        return loss_dict

    @property
    def dtype(self):
        return self.slot_attention.dtype

    @property
    def device(self):
        return self.slot_attention.device
