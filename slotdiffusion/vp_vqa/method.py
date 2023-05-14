import os
import wandb
import numpy as np

import torch
import torchvision.utils as vutils

from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts

from torch.utils.data._utils.collate import default_collate


def build_method(**kwargs):
    params = kwargs['params']
    if params.model == 'LDMSlotFormer':
        return LDMSlotFormerMethod(**kwargs)
    elif params.model == 'PhysionReadout':
        return PhysionReadoutMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


def to_rgb_from_tensor(x):
    """Reverse the Normalize operation in torchvision."""
    return (x * 0.5 + 0.5).clamp(0, 1)


class SlotBaseMethod(BaseMethod):
    """Base method in this project."""

    @staticmethod
    def _pad_frame(video, target_T):
        """Pad the video to a target length at the end"""
        if video.shape[0] >= target_T:
            return video
        dup_video = torch.stack(
            [video[-1]] * (target_T - video.shape[0]), dim=0)
        return torch.cat([video, dup_video], dim=0)

    @staticmethod
    def _pause_frame(video, N=4):
        """Pause the video on the first frame by duplicating it"""
        dup_video = torch.stack([video[0]] * N, dim=0)
        return torch.cat([dup_video, video], dim=0)

    def _convert_video(self, video, caption=None):
        video = torch.cat(video, dim=2)  # [T, 3, B*H, L*W]
        video = (video * 255.).numpy().astype(np.uint8)
        return wandb.Video(video, fps=self.vis_fps, caption=caption)

    @staticmethod
    def _get_sample_idx(N, dst):
        """Load videos uniformly from the dataset."""
        dst_len = len(dst.files)  # treat each video as a sample
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N)
        return sampled_idx

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_video:
            self._sample_video(model)

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

    @property
    def vis_fps(self):
        return 8


class SAViMethod(SlotBaseMethod):
    """SAVi model training method."""

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1. - masks),  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                pad_value=0.,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, labels = [], []
        for i in sampled_idx:
            data_dict = dst.get_video(i.item())
            video, label = data_dict['video'].float().to(self.device), \
                data_dict.get('label', None)
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            recon_combined, recons, masks = out_dict['post_recon_combined'], \
                out_dict['post_recons'], out_dict['post_masks']
            imgs = video.type_as(recon_combined)
            save_video = self._make_video_grid(imgs, recon_combined, recons,
                                               masks)
            results.append(save_video)
            labels.append(label)

        if all(lbl is not None for lbl in labels):
            caption = '\n'.join(
                ['Success' if lbl == 1 else 'Fail' for lbl in labels])
        else:
            caption = None
        wandb.log({'val/video': self._convert_video(results, caption=caption)},
                  step=self.it)
        torch.cuda.empty_cache()


class PhysionReadoutMethod(SlotBaseMethod):
    """PhysionReadout model training method."""

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        save_load_img = dst.load_img
        dst.load_img = True
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst).numpy()
        batch = default_collate([dst.__getitem__(i) for i in sampled_idx])
        batch = {k: v.cuda() for k, v in batch.items()}
        dst.load_img = save_load_img
        out_dict = model(batch)

        videos = to_rgb_from_tensor(batch['img'])  # [B, T, C, H, W]
        videos = (videos * 255.).cpu().numpy().astype(np.uint8)
        gts = batch['label'].flatten()  # [B]
        preds = torch.sigmoid(out_dict['logits'].flatten())  # [B]
        texts = [
            f'GT: {int(gt.item())}. Pred: {pred.item():.3f}'
            for gt, pred in zip(gts, preds)
        ]
        log_dict = {
            f'val/video{i}': wandb.Video(videos[i], fps=16, caption=texts[i])
            for i in range(videos.shape[0])
        }
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()


class SlotFormerMethod(SAViMethod):
    """SlotFormer model training method."""

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        super()._training_step_start()

        if not hasattr(self.params, 'use_loss_decay'):
            return

        # decay the temporal weighting linearly
        if not self.params.use_loss_decay:
            self.model.module.loss_decay_factor = 1.
            return

        cur_steps = self.it
        total_steps = self.params.max_epochs * len(self.train_loader)
        decay_steps = self.params.loss_decay_pct * total_steps

        if cur_steps >= decay_steps:
            self.model.module.loss_decay_factor = 1.
            return

        # increase tau linearly from 0.01 to 1
        self.model.module.loss_decay_factor = \
            0.01 + cur_steps / decay_steps * 0.99

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        if not hasattr(self.params, 'use_loss_decay'):
            return

        # also log the loss_decay_factor
        x = self.model.module.loss_decay_factor
        wandb.log({'train/loss_decay_factor': x}, step=self.it)

    def _compare_videos(self, img, recon_combined, rollout_combined):
        """Stack 3 videos to compare them."""
        # pad to the length of rollout video
        T = rollout_combined.shape[0]
        img = self._pad_frame(img, T)
        recon_combined = self._pad_frame(recon_combined, T)
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    img,  # original images
                    recon_combined,  # reconstructions
                    rollout_combined,  # rollouts
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                # pad white if using black background
                pad_value=1 if self.params.get('reverse_color', False) else 0,
            ) for i in range(img.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    def _read_video_and_slots(self, dst, idx):
        """Read the video and slots from the dataset."""
        # read video
        video = dst.get_video(idx)['video']
        # read slots
        video_path = dst.files[idx]
        slots = dst.video_slots[os.path.basename(video_path)]  # [T, N, C]
        # slots = np.zeros((self.params.video_len, 8, 192))
        if self.params.frame_offset > 1:
            slots = np.ascontiguousarray(slots[::self.params.frame_offset])
        slots = torch.from_numpy(slots).float().to(self.device)
        T = min(video.shape[0], slots.shape[0])
        # video: [T, 3, H, W], slots: [T, N, C]
        return video[:T], slots[:T]

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        BaseMethod.validation_epoch(self, model, san_check_step=san_check_step)

        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_video:
            self._sample_video(model)

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, rollout_results, compare_results = [], [], []
        for i in sampled_idx:
            video, slots = self._read_video_and_slots(dst, i.item())
            T = video.shape[0]
            # reconstruct gt_slots as sanity-check
            # i.e. if the pre-trained weights are loaded correctly
            recon_combined, recons, masks, _ = model.decode(slots)
            img = video.type_as(recon_combined)
            save_video = self._make_video_grid(img, recon_combined, recons,
                                               masks)
            results.append(save_video)
            # rollout
            past_steps = self.params.input_frames
            past_slots = slots[:past_steps][None]  # [1, t, N, C]
            out_dict = model.rollout(
                past_slots, T - past_steps, decode=True, with_gt=True)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            rollout_combined, recons, masks = \
                out_dict['recon_combined'], out_dict['recons'], out_dict['masks']
            img = video.type_as(rollout_combined)
            pred_video = self._make_video_grid(img, rollout_combined, recons,
                                               masks)
            rollout_results.append(pred_video)  # per-slot rollout results
            # stack (gt video, gt slots recon video, slot_0 rollout video)
            # horizontally to better compare the 3 videos
            compare_video = self._compare_videos(img, recon_combined,
                                                 rollout_combined)
            compare_results.append(compare_video)

        log_dict = {
            'val/video': self._convert_video(results),
            'val/rollout_video': self._convert_video(rollout_results),
            'val/compare_video': self._convert_video(compare_results),
        }
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()


class LDMSlotFormerMethod(SlotFormerMethod):
    """SlotFormer + SlotDiffusion training method."""

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_video:
            self._sample_video(model)

    def _make_video_grid(self, imgs, recon_combined, recons=None, masks=None):
        """Make a video of grid images showing slot decomposition."""
        scale = 0. if self.params.get('reverse_color', False) else 1.
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    # recons * masks + (1. - masks) * scale,  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                pad_value=1. - scale,
                # ) for i in range(recons.shape[0])
            ) for i in range(recon_combined.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    def _slots2video(self, model, slots):
        """Decode slots to videos."""
        T = slots.shape[0]
        bs = 32
        all_hard_recon = []
        for idx in range(0, T, bs):
            slots_dict = {"slots": slots[idx:idx + bs]}
            hard_recon = model.log_images(
                slots_dict, use_dpm=True, same_noise=True)
            all_hard_recon.append(hard_recon["samples"])
        return torch.cat(all_hard_recon)

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, rollout_results, compare_results = [], [], []
        for i in sampled_idx:
            video, slots = self._read_video_and_slots(dst, i.item())
            T = video.shape[0]
            # reconstruct gt_slots as sanity-check
            # i.e. if the pre-trained weights are loaded correctly
            log_samples = self._slots2video(model, slots)
            img = video.type_as(log_samples)
            recon_combined = log_samples
            save_video = self._make_video_grid(img, recon_combined)
            results.append(save_video)

            # rollout
            past_steps = self.params.input_frames
            past_slots = slots[:past_steps][None]  # [1, t, N, C]
            out_dict = model.rollout(
                past_slots, T - past_steps, decode=True, with_gt=True)
            out_dict = {k: v[0] for k, v in out_dict.items()}

            rollout_combined = out_dict['recon_combined']
            img = video.type_as(rollout_combined)
            pred_video = self._make_video_grid(img, rollout_combined)
            rollout_results.append(pred_video)  # per-slot rollout results
            # stack (gt video, gt slots recon video, slot_0 rollout video)
            # horizontally to better compare the 3 videos
            compare_video = self._compare_videos(img, recon_combined,
                                                 rollout_combined)
            compare_results.append(compare_video)

        log_dict = {
            'val/video': self._convert_video(results),
            'val/rollout_video': self._convert_video(rollout_results),
            'val/compare_video': self._convert_video(compare_results),
        }
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
