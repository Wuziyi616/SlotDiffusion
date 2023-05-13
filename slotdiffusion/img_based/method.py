import wandb
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts

from slotdiffusion.img_based.models import to_rgb_from_tensor, get_lr, \
    cosine_anneal
from slotdiffusion.img_based.datasets import draw_coco_bbox
from slotdiffusion.img_based.vis import torch_draw_rgb_mask


def build_method(**kwargs):
    params = kwargs['params']
    if params.model in ['SADiffusion', 'SA', 'SLATE']:
        return eval(f'{params.model}Method')(**kwargs)
    elif 'VAE' in params.model:
        return VAEMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


class SlotBaseMethod(BaseMethod):
    """Base method in this project."""

    def _convert_img(self, img, caption=None):
        """Convert torch.tensor image in [0, 1] to wandb.Image."""
        # rescale to [0., 1.]
        dst, dst_name = self.val_loader.dataset, self.params.dataset
        if dst_name in ['coco', 'voc']:
            norm = dst.__getattribute__(f'{dst_name}_transforms').normalize
            img = norm.denormalize_image(img)
        else:
            img = to_rgb_from_tensor(img)
        # [3, H, W] --> [H, W, 3]
        img = img.permute(1, 2, 0)
        # to [0, 255], np.uint8
        img = (img * 255.).cpu().numpy().astype(np.uint8)
        return wandb.Image(img, caption=caption)

    def _get_sample_idx(self, N, dst):
        """Load samples uniformly from the dataset."""
        dst_len = len(dst)
        if N > dst_len:
            N = dst_len
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N).numpy()
        return sampled_idx

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_img=True):
        """Validate one epoch.
        We aggregate the avg of all statistics and only log once.
        """
        self._is_epoch_end = (san_check_step <= 0)
        self._is_last_epoch = (self.epoch == self.max_epochs)
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_img:
            self._sample_img(model)

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        optimizer = super()._configure_optimizers()[0]

        lr = self.params.lr
        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')


class SAMethod(SlotBaseMethod):
    """SlotAttention model training method."""

    def _make_img_grid(self, imgs, recon_combined, recons, masks):
        """Make grid images showing slot decomposition."""
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = torch.cat(
            [
                imgs.unsqueeze(1),  # original images
                recon_combined.unsqueeze(1),  # reconstructions
                recons * masks + (1. - masks),  # each slot
            ],
            dim=1,
        ).cpu()  # [B, num_slots+2, C, H, W]
        images = vutils.make_grid(
            out.flatten(0, 1),
            nrow=out.shape[1],
            pad_value=-1.,
        )  # [C, B*H, (num_slots+2)*W]
        return images

    @torch.no_grad()
    def _sample_img(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        collate_fn = self.val_loader.collate_fn
        data_dict = collate_fn([dst[i] for i in sampled_idx])
        data_dict = {k: v.to(model.device) for k, v in data_dict.items()}

        out_dict = model(data_dict)
        recon_img, recons, masks = \
            out_dict['recon_img'], out_dict['recons'], out_dict['masks']
        imgs = data_dict['img']
        images = self._make_img_grid(imgs, recon_img, recons, masks)

        wandb.log({'val/imgs': self._convert_img(images)}, step=self.it)
        torch.cuda.empty_cache()


class SLATEMethod(SlotBaseMethod):
    """SLATE model training method."""

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        assert self.params.optimizer.lower() == 'adam'
        assert self.params.weight_decay <= 0.
        lr = self.params.lr
        dec_lr = self.params.dec_lr

        # SLATE uses different lr for its Transformer decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'trans_decoder' not in kv[0] and kv[1].
                requires_grad, self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'trans_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
            },
        ]

        optimizer = optim.Adam(params_list, lr=lr, weight_decay=0.)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @staticmethod
    def _make_img(img, other_img):
        """imgs are of shape [C, H, W]"""
        return VAEMethod._make_img(img, other_img)

    @staticmethod
    def _make_slots_img(imgs, recon_imgs, masks, gt_bboxs, gt_masks):
        """[3, H, W]x2, [N, H, W]"""
        return SADiffusionMethod._make_slots_img(imgs, recon_imgs, masks,
                                                 gt_bboxs, gt_masks)

    @torch.no_grad()
    def _sample_img(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots & masks
        dst = self.val_loader.dataset

        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        collate_fn = self.val_loader.collate_fn
        data_dict = collate_fn([dst[i] for i in sampled_idx])
        data_dict = {k: v.to(model.device) for k, v in data_dict.items()}
        out_dict = model(data_dict)
        imgs = data_dict['img']
        masks = out_dict['masks']
        wandb_dict = {}

        slots = out_dict['slots']  # [B, num_slots, slot_size]
        recon_imgs = model.recon_img(slots, bs=16, verbose=True)
        wandb_dict['val/sample_img_recon_loss'] = F.mse_loss(recon_imgs, imgs)

        # in case GT masks & bboxs are provided
        gt_bboxs = data_dict.get('annos', None)
        gt_masks = data_dict.get('masks', None)

        masked_img = self._make_slots_img(imgs, recon_imgs, masks, gt_bboxs,
                                          gt_masks)
        wandb_dict['val/recon'] = self._convert_img(masked_img)

        wandb.log(wandb_dict, step=self.it)
        torch.cuda.empty_cache()
        model.testing = False

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)


class SADiffusionMethod(SlotBaseMethod):
    """SlotAttention with Diffusion decoder model training method."""

    def _clip_model_grad(self):
        """Clip model weights' gradients."""
        sa_clip_grad = self.clip_grad
        sa_params = self.optimizer.param_groups[0]['params']
        nn.utils.clip_grad_norm_(sa_params, sa_clip_grad)
        # usually DM papers don't do gradient clipping
        dec_clip_grad = self.params.get('dec_clip_grad', sa_clip_grad)
        if dec_clip_grad > 0.:
            dec_params = self.optimizer.param_groups[1]['params']
            nn.utils.clip_grad_norm_(dec_params, dec_clip_grad)

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        if self.params.optimizer.lower() == 'adam':
            opt = optim.Adam
        elif self.params.optimizer.lower() == 'adamw':
            opt = optim.AdamW
        else:
            raise ValueError('Should use Adam or AdamW optimizer!')
        if self.params.weight_decay > 0.:
            assert self.params.optimizer.lower() == 'adamw', \
                'Should use AdamW optimizer for weight decay!'

        lr = self.params.lr
        dec_lr = self.params.get('dec_lr', lr)

        # may need different lr for the DM decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'dm_decoder' not in kv[0] and kv[1].requires_grad,
                self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'dm_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
                'lr': lr,
                'weight_decay': 0.,
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
                'weight_decay': self.params.weight_decay,
            },
        ]

        optimizer = opt(params_list)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @staticmethod
    def _make_slots_img(imgs, recon_imgs, masks, gt_bboxs=None, gt_masks=None):
        """[B, 3, H, W]x2, [B, N, H, W], [B, N, 5], [B, H, W]"""
        # merge masks together, and map it to the imgs
        # [B, 3, H, W], [B, 3, H, W]
        rgb_mask, colored_mask = torch_draw_rgb_mask(imgs, masks.argmax(1))
        # make the masked image, [B, N, 3, H, W]
        masked_imgs = imgs.unsqueeze(1) * masks.unsqueeze(2) + \
            (1. - masks.unsqueeze(2))
        img_lst = [
            imgs.unsqueeze(1),  # [B, 1, 3, H, W]
            # recon_imgs.unsqueeze(1),  # [B, 1, 3, H, W]
            colored_mask.unsqueeze(1),  # [B, 1, 3, H, W]
            rgb_mask.unsqueeze(1),  # [B, 1, 3, H, W]
            masked_imgs,  # [B, N, 3, H, W]
        ]
        if recon_imgs is not None:
            img_lst.insert(1, recon_imgs.unsqueeze(1))
        img_lst = [img.cpu() for img in img_lst]
        if gt_masks is not None:
            _, gt_colored_mask = torch_draw_rgb_mask(imgs, gt_masks)
            img_lst.insert(2, gt_colored_mask.unsqueeze(1).cpu())
        if gt_bboxs is not None:
            bbox_img = draw_coco_bbox(imgs.cpu(), gt_bboxs.cpu())
            img_lst.insert(2, bbox_img.unsqueeze(1))
        out = torch.cat(img_lst, dim=1).cpu()  # [B, N + 4, 3, H, W]
        out = vutils.make_grid(
            out.flatten(0, 1),
            nrow=out.shape[1],
            pad_value=-1.,
        )  # [3, B*H, (N+4)*W]
        return out

    @torch.no_grad()
    def _sample_img(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots & masks
        dst = self.val_loader.dataset
        cls_free_guidance = self.params.get('cls_free_guidance', False)

        # 1. sample a batch of images, evaluate recon loss
        # slow, only do this at the end of each epoch
        if self._is_epoch_end:
            sampled_idx = self._get_sample_idx(self.params.val_batch_size, dst)
            collate_fn = self.val_loader.collate_fn
            data_dict = collate_fn([dst[i] for i in sampled_idx])
            data_dict = {k: v.to(model.device) for k, v in data_dict.items()}
            imgs = data_dict['img']
            dpm_log_dict = model.log_images(
                data_dict,
                use_dpm=True,
                cls_free_guidance=cls_free_guidance,
            )
            dpm_recon_imgs = dpm_log_dict['samples']
            dpm_recon_loss = F.mse_loss(dpm_recon_imgs, imgs)
            del dpm_log_dict
            torch.cuda.empty_cache()
            wandb_dict = {
                'val/dpm_sample_img_recon_loss': dpm_recon_loss.cpu().item()
            }
        else:
            wandb_dict = {}

        # 2. sample a few images for visualization
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        collate_fn = self.val_loader.collate_fn
        data_dict = collate_fn([dst[i] for i in sampled_idx])
        data_dict = {k: v.to(model.device) for k, v in data_dict.items()}

        log_dict = model.log_images(
            data_dict, use_dpm=True, cls_free_guidance=cls_free_guidance)
        '''
            - 'diffusion_row': from x0 (input imgs) to xT (random noise)
            - 'denoise_row': from xT denoises to x0
            - 'samples': the final denoised (reconstructed) imgs
            - 'masks': [B, N, H, W]
        '''
        imgs, masks, recon_imgs = \
            data_dict['img'], log_dict.pop('masks'), log_dict.pop('samples')
        gt_bboxs = data_dict.get('annos', None)
        gt_masks = data_dict.get('masks', None)
        log_dict['recon'] = self._make_slots_img(imgs, recon_imgs, masks,
                                                 gt_bboxs, gt_masks)
        wandb_dict.update(
            {f'val/{k}': self._convert_img(v)
             for k, v in log_dict.items()})
        wandb.log(wandb_dict, step=self.it)
        torch.cuda.empty_cache()
        model.testing = False

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)


class VAEMethod(SlotBaseMethod):
    """dVAE and VQVAE model training method."""

    @staticmethod
    def _make_img(imgs, pred_imgs):
        """imgs are of shape [N, C, H, W]"""
        out = torch.stack([
            imgs.cpu(),
            pred_imgs.cpu(),
        ], dim=1).cpu()  # [N, 2, 3, H, W]
        out = vutils.make_grid(
            out.flatten(0, 1),
            nrow=out.shape[1],
            pad_value=-1.,
        )  # [T, 3, H, 2*W]
        return out

    @torch.no_grad()
    def _sample_img(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        collate_fn = self.val_loader.collate_fn
        data_dict = collate_fn([dst[i] for i in sampled_idx])
        data_dict = {k: v.to(model.device) for k, v in data_dict.items()}

        out_dict = model(data_dict)
        recon_imgs = out_dict['recon']
        imgs = data_dict['img']
        images = self._make_img(imgs, recon_imgs)

        wandb.log({'val/imgs': self._convert_img(images)}, step=self.it)
        torch.cuda.empty_cache()

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        super()._training_step_start()

        if self.params.model != 'dVAE':
            return

        # dVAE: update the tau (gumbel softmax temperature)
        total_steps = self.params.max_epochs * len(self.train_loader)
        decay_steps = self.params.tau_decay_pct * total_steps

        # decay tau
        self.model.module.tau = cosine_anneal(
            self.it,
            start_value=self.params.init_tau,
            final_value=self.params.final_tau,
            start_step=0,
            final_step=decay_steps,
        )

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.params.model != 'dVAE':
            return

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # also log the tau
        wandb.log({'train/gumbel_tau': self.model.module.tau}, step=self.it)
