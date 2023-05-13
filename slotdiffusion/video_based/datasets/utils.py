import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F


def compact(l):
    return list(filter(None, l))


def first(x):
    return next(iter(x))


def only(x):
    materialized_x = list(x)
    assert len(materialized_x) == 1
    return materialized_x[0]


def suppress_mask_idx(masks):
    """Make the mask index 0, 1, 2, ..."""
    # the original mask could have not continuous index, 0, 3, 4, 6, 9, 13, ...
    # we make them 0, 1, 2, 3, 4, 5, ...
    obj_idx = torch.unique(masks)
    idx_mapping = torch.arange(obj_idx.max() + 1)
    idx_mapping[obj_idx] = torch.arange(len(obj_idx))
    masks = idx_mapping[masks]
    return masks


class BaseTransforms(object):
    """Data pre-processing steps."""

    def __init__(self, resolution, mean=(0.5, ), std=(0.5, ), flip=False):
        trans = [
            transforms.ToTensor(),  # [3, H, W]
            transforms.Normalize(mean, std),  # [-1, 1]
            transforms.Resize(resolution),
        ]
        if flip:
            trans.append(transforms.RandomHorizontalFlip())
        self.transforms = transforms.Compose(trans)
        self.resolution = resolution
        self.flip = flip

    def process_mask(self, mask):
        assert not self.flip
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)[0]
        else:
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)
        return mask

    def __call__(self, input):
        return self.transforms(input)
