import os.path as osp
import numpy as np
from PIL import Image, ImageFile

import torch
from torch.utils.data import Dataset

from nerv.utils import load_obj

try:
    from .utils import BaseTransforms
except ImportError:
    from utils import BaseTransforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


class PhysionDataset(Dataset):
    """Dataset for loading Physion videos."""

    def __init__(
        self,
        data_root,
        physion_transform,
        split='train',
        tasks=['all'],
        n_sample_frames=6,
        frame_offset=None,
        video_len=150,
        subset='training',
    ):

        if subset in ['training', 'readout']:
            assert split in ['train', 'val']
        elif subset == 'test':
            assert split == 'test'
        else:
            raise NotImplementedError(f'Unknown subset: {subset}')

        self.data_root = data_root
        self.split = split
        self.tasks = tasks
        self.physion_transform = physion_transform
        self.n_sample_frames = n_sample_frames
        self.frame_offset = frame_offset
        self.video_len = video_len
        self.subset = subset

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

    def _read_frames(self, idx):
        folder, start_idx = self._get_video_start_idx(idx)
        filename = osp.join(folder, '{:06d}.jpg')
        frames = [
            Image.open(filename.format(start_idx +
                                       n * self.frame_offset)).convert('RGB')
            for n in range(self.n_sample_frames)
        ]
        # raise error if any frame is corrupted
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [self.physion_transform(img) for img in frames]
        return torch.stack(frames, dim=0)  # [T, C, H, W]

    def get_video(self, video_idx):
        folder = self.files[video_idx]
        # read frames saved from videos
        assert osp.exists(folder), "Please extract frames from videos first."
        num_frames = self.video_len // self.frame_offset
        frames = [
            Image.open(osp.join(
                folder, f'{n * self.frame_offset:06d}.jpg')).convert('RGB')
            for n in range(num_frames)
        ]
        # corrupted video
        if any(frame is None for frame in frames):
            return self._rand_another(is_video=True)
        frames = [self.physion_transform(img) for img in frames]
        return {
            'video': torch.stack(frames, dim=0),
            'data_idx': video_idx,
        }

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [T, 3, H, W]
        """
        if self.load_video:
            return self.get_video(idx)

        try:
            frames = self._read_frames(idx)
        except ValueError:
            return self._rand_another()
        data_dict = {
            'data_idx': idx,
            'img': frames,
        }
        return data_dict

    def _get_sample_idx(self):
        valid_idx = []  # (video_folder, start_idx)
        # get the train-val split file for each subset
        # by default we store it under './splits/Physion'
        cur_dir = osp.dirname(osp.realpath(__file__))
        json_fn = osp.join(cur_dir,
                           f'splits/Physion/{self.subset}_{self.split}.json')
        json_file = load_obj(json_fn)
        # count the number of each task
        # ['Collide', 'Contain', 'Dominoes', 'Drape', 'Drop', 'Link', 'Roll', 'Support']
        self.all_tasks = sorted(list(json_file.keys()))
        self.task2num = {task: len(json_file[task]) for task in self.all_tasks}
        self.video_idx2task_idx = {}  # mapping from video index to task index
        # load video paths
        self.files = []
        if self.tasks[0].lower() == 'all':
            print('Loading all tasks from Physion...')
            self.tasks = list(json_file.keys())
        for task in self.tasks:
            idx1 = len(self.files)
            task_files = json_file[task]  # 'xxx.mp4'
            task_files = [osp.join(self.data_root, f[:-4]) for f in task_files]
            self.files.extend(task_files)
            idx2 = len(self.files)
            self.video_idx2task_idx.update(
                {idx: self.all_tasks.index(task)
                 for idx in range(idx1, idx2)})
        self.num_videos = len(self.files)
        for folder in self.files:
            # simply use random uniform sampling
            if self.split == 'train':
                max_start_idx = self.video_len - \
                    (self.n_sample_frames - 1) * self.frame_offset
                valid_idx += [(folder, idx) for idx in range(max_start_idx)]
            # only test once per video
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


def build_physion_dataset(params, val_only=False):
    """Build Physion video dataset."""
    subset = params.dataset.split('_')[-1]
    physion_transform = BaseTransforms(params.resolution)
    args = dict(
        data_root=params.data_root,
        physion_transform=physion_transform,
        split='val',
        tasks=params.tasks,
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        video_len=params.video_len,
        subset=subset,
    )
    if subset == 'test':
        args['split'] = 'test'
        val_only = True
    val_dataset = PhysionDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = PhysionDataset(**args)
    return train_dataset, val_dataset
