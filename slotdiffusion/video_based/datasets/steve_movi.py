import cv2
import os.path as osp
import numpy as np
from PIL import Image, ImageFile

import torch

from .movi import MOViDataset
from .utils import BaseTransforms, suppress_mask_idx

ImageFile.LOAD_TRUNCATED_IMAGES = True


class STEVEMOViDataset(MOViDataset):
    """Dataset for loading MOVi-Tex and MOVi-Solid videos."""

    def __init__(
        self,
        level,
        data_root,
        movi_transform,
        split='train',
        n_sample_frames=6,
        frame_offset=None,
        video_len=24,
        load_mask=False,
    ):

        assert level in ['Tex', 'Solid']
        assert split in ['train', 'test'], 'Val set does not have GT masks'

        self.dataset = 'STEVEMOVi'
        self.level = level
        self.data_root = osp.join(data_root, f'MOVi-{level}', split)
        self.split = split
        self.movi_transform = movi_transform
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self.video_len = video_len
        self.load_mask = load_mask
        self.num_masks = 10  # hard-code this number

        # Get all numbers
        self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False

    def _read_frames(self, idx, is_video=False):
        # a hack for reading the entire video
        if is_video:
            num = self.video_len // self.frame_offset
            folder, start_idx = self.files[idx], 0
        else:
            num = self.n_sample_frames
            folder, start_idx = self._get_video_start_idx(idx)
        filename = osp.join(folder, '{:08d}_image.png')
        frames = [
            Image.open(filename.format(start_idx +
                                       n * self.frame_offset)).convert('RGB')
            for n in range(num)
        ]
        # raise error if any frame is corrupted
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [self.movi_transform(img) for img in frames]
        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def _merge_masks(self, mask_name):
        # each frame has `num_masks` binary masks
        # each mask's name is e.g. '00000011_mask_06.png'
        # input `mask_name` is the 'prefix' '00000011_mask.png'
        # we need to merge all masks into one by taking argmax
        mask_name = mask_name.replace('_mask.png', '_mask_{:02d}.png')
        masks = [
            cv2.imread(mask_name.format(i), cv2.IMREAD_GRAYSCALE)
            for i in range(self.num_masks)
        ]
        # raise error if any mask is corrupted
        if any(mask is None for mask in masks):
            raise ValueError
        # the loaded masks are all object masks
        # so we need to insert a background mask to position 0th
        # so that bg label after argmax is 0
        masks.insert(0, np.ones_like(masks[0]))
        mask = np.stack(masks, axis=0).argmax(0).astype(np.uint8)
        return mask

    def _read_masks(self, idx, is_video=False):
        # a hack for reading the entire video
        if is_video:
            num = self.video_len // self.frame_offset
            folder, start_idx = self.files[idx], 0
        else:
            num = self.n_sample_frames
            folder, start_idx = self._get_video_start_idx(idx)
        filename = osp.join(folder, '{:08d}_mask.png')
        masks = [
            self._merge_masks(
                filename.format(start_idx + n * self.frame_offset))
            for n in range(num)
        ]
        masks = [self.movi_transform.process_mask(mask) for mask in masks]
        masks = torch.stack(masks, dim=0)  # [T, H, W]
        # convert obj_idx to [0, 1, 2, ...]
        masks = suppress_mask_idx(masks)
        return masks


def build_steve_movi_dataset(params, val_only=False):
    """Build MOVi video dataset from STEVE paper."""
    movi_transform = BaseTransforms(params.resolution)
    args = dict(
        level=params.movi_level,
        data_root=params.data_root,
        movi_transform=movi_transform,
        split='test',
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
        load_mask=params.load_mask,
    )
    if val_only:
        print(f'Using MOVi-{params.movi_level} test set!')
        args['split'] = 'test'
    val_dataset = STEVEMOViDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['load_mask'] = False  # no need mask for training
    train_dataset = STEVEMOViDataset(**args)
    return train_dataset, val_dataset
