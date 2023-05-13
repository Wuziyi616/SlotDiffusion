import cv2
import os
import os.path as osp
import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

from nerv.utils import glob_all, load_obj, dump_obj

from .utils import BaseTransforms, suppress_mask_idx

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MOViDataset(Dataset):
    """Dataset for loading MOVi-[A~E] videos."""

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

        level = level.upper()
        assert level in ['A', 'B', 'C', 'D', 'E']
        if split == 'val':
            split = 'validation'
        assert split in ['train', 'validation', 'test']

        self.dataset = 'MOVi'
        self.level = level
        self.data_root = osp.join(data_root, f'MOVi-{level}', split)
        self.split = split
        self.movi_transform = movi_transform
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self.video_len = video_len
        self.load_mask = load_mask

        # Get all numbers
        self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False

    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

    def _read_frames(self, idx, is_video=False):
        # a hack for reading the entire video
        if is_video:
            num = self.video_len // self.frame_offset
            folder, start_idx = self.files[idx], 0
        else:
            num = self.n_sample_frames
            folder, start_idx = self._get_video_start_idx(idx)
        filename = osp.join(folder, '{:06d}.jpg')
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

    def _read_masks(self, idx, is_video=False):
        # a hack for reading the entire video
        if is_video:
            num = self.video_len // self.frame_offset
            folder, start_idx = self.files[idx], 0
        else:
            num = self.n_sample_frames
            folder, start_idx = self._get_video_start_idx(idx)
        filename = osp.join(folder, '{:06d}_mask.png')
        masks = [
            cv2.imread(
                filename.format(start_idx + n * self.frame_offset),
                cv2.IMREAD_GRAYSCALE) for n in range(num)
        ]
        # raise error if any mask is corrupted
        if any(mask is None for mask in masks):
            raise ValueError
        masks = [self.movi_transform.process_mask(mask) for mask in masks]
        masks = torch.stack(masks, dim=0)  # [T, H, W]
        # convert obj_idx to [0, 1, 2, ...]
        masks = suppress_mask_idx(masks)
        return masks

    def get_video(self, video_idx):
        try:
            frames = self._read_frames(video_idx, is_video=True)
            if self.load_mask:
                masks = self._read_masks(video_idx, is_video=True)
        except ValueError:
            data_dict = self._rand_another(is_video=True)
            data_dict['error_flag'] = True
        data_dict = {
            'video': frames,
            'data_idx': video_idx,
        }
        if self.load_mask:
            data_dict['masks'] = masks
        return data_dict

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
            - mask (optional): [T, H, W]
        """
        if self.load_video:
            return self.get_video(idx)

        try:
            frames = self._read_frames(idx)
            if self.load_mask:
                masks = self._read_masks(idx)
        except ValueError:
            return self._rand_another()
        data_dict = {
            'data_idx': idx,
            'img': frames,
        }
        if self.load_mask:
            data_dict['masks'] = masks
        return data_dict

    def _get_sample_idx(self):
        valid_idx = []  # (video_folder, start_idx)
        cur_dir = osp.dirname(osp.realpath(__file__))
        file_path = osp.join(cur_dir, f'splits/{self.dataset}',
                             f'{self.level}/{self.split}.json')
        if not osp.exists(file_path):
            self.files = glob_all(self.data_root, only_dir=True)
            os.makedirs(osp.dirname(file_path), exist_ok=True)
            dump_obj(self.files, file_path)
        else:
            self.files = load_obj(file_path)
        self.num_videos = len(self.files)
        for folder in self.files:
            # simply use random uniform sampling
            if self.split == 'train':
                max_start_idx = self.video_len - \
                    (self.n_sample_frames - 1) * self.frame_offset
                valid_idx += [(folder, idx) for idx in range(max_start_idx)]
            # only test once per video
            # since MOVi videos are not time-equivalent, i.e. objects are
            #     thrown to the ground at the beginning of each video, where
            #     such motions make seg easier, while during the later part
            #     part of the video, there is rarely object movements
            # therefore, we should only test one video once, [0, T]
            elif self.split == 'test':
                valid_idx += [(folder, 0)]
            # test one frame once
            else:
                size = self.n_sample_frames * self.frame_offset
                start_idx = []
                for idx in range(0, self.video_len - size + 1, size):
                    start_idx += [i + idx for i in range(self.frame_offset)]
                valid_idx += [(folder, idx) for idx in start_idx]
        return valid_idx

    def __len__(self):
        if self.load_video:
            return len(self.files)
        return len(self.valid_idx)


def build_movi_dataset(params, val_only=False):
    """Build MOVi video dataset."""
    movi_transform = BaseTransforms(params.resolution)
    args = dict(
        level=params.movi_level,
        data_root=params.data_root,
        movi_transform=movi_transform,
        split='val',
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
        load_mask=params.load_mask,
    )
    if val_only:
        print(f'Using MOVi-{params.movi_level} test set!')
        args['split'] = 'test'
    val_dataset = MOViDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['load_mask'] = False  # no need mask for training
    train_dataset = MOViDataset(**args)
    return train_dataset, val_dataset
