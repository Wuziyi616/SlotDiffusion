#
# Authors: Wouter Van Gansbeke & Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import os
import cv2
import copy

import numpy as np
from PIL import Image

import torch.utils.data as data

from .voc_transforms import VOCTransforms, VOCCollater


class VOC12Dataset(data.Dataset):

    VOC_CATEGORY_NAMES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
        'tvmonitor'
    ]

    def __init__(
        self,
        data_root,
        split='val',
        voc_transforms=None,
        load_anno=True,
        ignore_classes=[],
    ):
        # Set paths
        self.data_root = data_root
        valid_splits = ['trainaug', 'train', 'val']
        assert (split in valid_splits)
        self.split = split
        self.load_anno = load_anno

        if split == 'trainaug':
            _semseg_dir = os.path.join(self.data_root, 'SegmentationClassAug')
        else:
            _semseg_dir = os.path.join(self.data_root, 'SegmentationClass')

        _instseg_dir = os.path.join(self.data_root, 'SegmentationObject')
        _image_dir = os.path.join(self.data_root, 'images')

        # Transform
        self.voc_transforms = voc_transforms

        # Splits are pre-cut
        print("Initializing dataloader for PASCAL VOC12 {} set".format(''.join(
            self.split)))
        split_file = os.path.join(self.data_root, 'sets', self.split + '.txt')
        self.images, self.semsegs, self.instsegs = [], [], []

        with open(split_file, "r") as f:
            lines = f.read().splitlines()

        for ii, line in enumerate(lines):
            # Images
            _image = os.path.join(_image_dir, line + ".jpg")
            assert os.path.isfile(_image)
            self.images.append(_image)

            # Semantic Segmentation
            _semseg = os.path.join(_semseg_dir, line + '.png')
            assert os.path.isfile(_semseg)
            self.semsegs.append(_semseg)

            # Instance Segmentation
            # only available for val set
            if self.split == 'val':
                _instseg = os.path.join(_instseg_dir, line + '.png')
                assert os.path.isfile(_instseg)
            else:
                _instseg = _semseg
            self.instsegs.append(_instseg)

        assert (len(self.images) == len(self.semsegs) == len(self.instsegs))

        # Display stats
        print('Number of dataset images: {:d}'.format(len(self.images)))

        # List of classes which are remapped to ignore index.
        # This option is used for comparing with other works that consider only a subset of the pascal classes.
        self.ignore_classes = [
            self.VOC_CATEGORY_NAMES.index(class_name)
            for class_name in ignore_classes
        ]

    def __getitem__(self, index):
        sample = {}

        # Load image
        _img = self._load_img(index)  # [H, W, 3]
        sample['image'] = _img
        if not self.load_anno:
            H, W = _img.shape[:2]
            sample['masks'] = np.zeros((H, W), dtype=np.int32)
            return self.voc_transforms(sample)

        # Load pixel-level annotations
        _semseg = self._load_semseg(index)  # [H, W]
        if _semseg.shape != _img.shape[:2]:
            _semseg = cv2.resize(
                _semseg, _img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
        inst_overlap_masks = (_semseg == 255)

        # Load inst_seg mask
        # we don't have it for training set! Only load for val set
        if self.split == 'val':
            _instseg = self._load_instseg(index)  # [H, W]
            inst_overlap_masks = (_instseg == 255) | inst_overlap_masks
        else:
            _instseg = copy.deepcopy(_semseg)  # [H, W], fake it
        _semseg[inst_overlap_masks] = 0
        _instseg[inst_overlap_masks] = 0
        inst_overlap_masks = inst_overlap_masks.astype(np.uint8)
        masks = [_instseg, _semseg, inst_overlap_masks]
        sample['masks'] = np.stack(masks, axis=-1)  # [H, W, 3]

        return self.voc_transforms(sample)

    def __len__(self):
        return len(self.images)

    def _load_img(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB'))
        return _img.astype(np.uint8)

    def _load_semseg(self, index):
        # background is 0, 255 is ignored
        _semseg = np.array(Image.open(self.semsegs[index]))  # [H, W]
        for ignore_class in self.ignore_classes:
            _semseg[_semseg == ignore_class] = 255
        return _semseg

    def _load_instseg(self, index):
        # background is 0, 255 is ignored
        _instseg = np.array(Image.open(self.instsegs[index]))  # [H, W]
        return _instseg


def build_voc_dataset(params, val_only=False):
    """Build VOC12 dataset that load images."""
    norm_mean = params.get('norm_mean', 0.5)
    norm_std = params.get('norm_std', 0.5)
    val_transforms = VOCTransforms(
        params.resolution,
        norm_mean=norm_mean,
        norm_std=norm_std,
        val=True,
    )
    args = dict(
        data_root=params.data_root,
        voc_transforms=val_transforms,
        split='val',
        load_anno=params.load_anno,
    )
    val_dataset = VOC12Dataset(**args)
    if val_only:
        return val_dataset, VOCCollater()
    args['split'] = 'trainaug'
    args['load_anno'] = False
    args['voc_transforms'] = VOCTransforms(
        params.resolution,
        norm_mean=norm_mean,
        norm_std=norm_std,
        val=False,
    )
    train_dataset = VOC12Dataset(**args)
    return train_dataset, val_dataset, VOCCollater()
