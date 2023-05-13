import cv2
import copy
import numpy as np

import torch
import torchvision.transforms as transforms

from .utils import suppress_mask_idx


class RandomHorizontalFlip:
    """Flip the image and bbox horizontally."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, sample):
        # [H, W, 3], [H, W(, 2)], [N, 5]
        image, masks = sample['image'], sample['masks']

        if np.random.uniform(0, 1) < self.prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            masks = np.ascontiguousarray(masks[:, ::-1])

        return {
            'image': image,
            'masks': masks,
        }


class ResizeMinShape:
    """Resize for later center crop."""

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']
        h, w, _ = image.shape
        # resize so that the h' is at lease resolution[0]
        # and the w' is at lease resolution[1]
        factor = max(self.resolution[0] / h, self.resolution[1] / w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(
            image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        masks = cv2.resize(
            masks, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        return {
            'image': image,
            'masks': masks,
        }


class CenterCrop:

    def __init__(self, resolution=(224, 224), random=False):
        self.resolution = resolution
        self.random = random  # if True, offset the center crop

    def __call__(self, sample):
        image, masks = sample['image'], sample['masks']

        h, w, _ = image.shape
        assert h >= self.resolution[0] and w >= self.resolution[1]
        assert h == self.resolution[0] or w == self.resolution[1]

        if h == self.resolution[0]:
            crop_ymin = 0
            crop_ymax = h
            if self.random:
                crop_xmin = np.random.randint(0, w - self.resolution[1] + 1)
                crop_xmax = crop_xmin + self.resolution[1]
            else:
                crop_xmin = (w - self.resolution[0]) // 2
                crop_xmax = crop_xmin + self.resolution[0]
        else:
            crop_xmin = 0
            crop_xmax = w
            if self.random:
                crop_ymin = np.random.randint(0, h - self.resolution[1] + 1)
                crop_ymax = crop_ymin + self.resolution[1]
            else:
                crop_ymin = (h - self.resolution[1]) // 2
                crop_ymax = crop_ymin + self.resolution[1]
        image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        masks = masks[crop_ymin:crop_ymax, crop_xmin:crop_xmax]

        return {
            'image': image,
            'masks': masks,
        }


class Normalize:

    def __init__(self, mean=0.5, std=0.5):
        if isinstance(mean, (list, tuple)):
            mean = np.array(mean, dtype=np.float32)[None, None]  # [1, 1, 3]
        if isinstance(std, (list, tuple)):
            std = np.array(std, dtype=np.float32)[None, None]  # [1, 1, 3]
        self.mean = mean
        self.std = std

    def normalize_image(self, image):
        image = image.astype(np.float32) / 255.
        image = (image - self.mean) / self.std
        return image

    def denormalize_image(self, image):
        # simple numbers
        if isinstance(self.mean, (int, float)) and \
                isinstance(self.std, (int, float)):
            image = image * self.std + self.mean
            return image.clamp(0, 1)
        # need to convert the shapes
        mean = image.new_tensor(self.mean.squeeze())  # [3]
        std = image.new_tensor(self.std.squeeze())  # [3]
        if image.shape[-1] == 3:  # C last
            mean = mean[None, None]  # [1, 1, 3]
            std = std[None, None]  # [1, 1, 3]
        else:  # C first
            mean = mean[:, None, None]  # [3, 1, 1]
            std = std[:, None, None]  # [3, 1, 1]
        if len(image.shape) == 4:  # [B, C, H, W] or [B, H, W, C], batch dim
            mean = mean[None]
            std = std[None]
        image = image * self.std + self.mean
        return image.clamp(0, 1)

    def __call__(self, sample):
        # [H, W, C]
        image, masks = sample['image'], sample['masks']
        image = self.normalize_image(image)
        # make mask index start from 0 and continuous
        # `masks` is [H, W(, 2 or 3)]
        if len(masks.shape) == 3:
            assert masks.shape[-1] in [2, 3]
            # we don't suppress the last mask since it is the overlapping mask
            # i.e. regions with overlapping instances
            for i in range(masks.shape[-1] - 1):
                masks[:, :, i] = suppress_mask_idx(masks[:, :, i])
        else:
            masks = suppress_mask_idx(masks)
        return {
            'image': image,
            'masks': masks,
        }


class VOCCollater:

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        if 'clip_image' in data[0]:
            clip_images = [s['clip_image'] for s in data]
        if 'dino_image' in data[0]:
            dino_images = [s['dino_image'] for s in data]
        masks = [s['masks'] for s in data]

        images = np.stack(images, axis=0)  # [B, H, W, C]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [B, C, H, W]
        if 'clip_image' in data[0]:
            clip_images = np.stack(clip_images, axis=0)
            clip_images = torch.from_numpy(clip_images).permute(0, 3, 1, 2)
        if 'dino_image' in data[0]:
            dino_images = np.stack(dino_images, axis=0)
            dino_images = torch.from_numpy(dino_images).permute(0, 3, 1, 2)

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)  # [B, H, W(, 2 or 3)]

        data_dict = {
            'img': images.contiguous().float(),
            'masks': masks.contiguous().long(),
        }
        if 'clip_image' in data[0]:
            data_dict['clip_img'] = clip_images.contiguous().float()
        if 'dino_image' in data[0]:
            data_dict['dino_img'] = dino_images.contiguous().float()
        if len(masks.shape) == 4:
            assert masks.shape[-1] in [2, 3]
            if masks.shape[-1] == 3:
                data_dict['masks'] = masks[:, :, :, 0]
                data_dict['sem_masks'] = masks[:, :, :, 1]
                data_dict['inst_overlap_masks'] = masks[:, :, :, 2]
            else:
                data_dict['masks'] = masks[:, :, :, 0]
                data_dict['inst_overlap_masks'] = masks[:, :, :, 1]
        return data_dict


class VOCTransforms(object):
    """Data pre-processing steps."""

    def __init__(
        self,
        resolution,
        norm_mean=0.5,
        norm_std=0.5,
        val=False,
        load_clip=False,
        load_dino=False,
    ):
        self.common_transforms = transforms.Compose([
            RandomHorizontalFlip(0.5 if not val else 0),
            ResizeMinShape(resolution),
            CenterCrop(resolution, random=(not val)),
        ])
        self.normalize = Normalize(norm_mean, norm_std)
        self.resolution = resolution

        # potentially perform CLIP preprocessing
        self.load_clip = load_clip
        if load_clip:
            self.clip_normalize = Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            )
        # potentially perform DINO preprocessing
        self.load_dino = load_dino
        if load_dino:
            assert not load_clip, 'Cannot load both CLIP and DINO'
            self.dino_normalize = Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )

    def __call__(self, input):
        common_data = self.common_transforms(input)
        img = copy.deepcopy(common_data['image'])  # [H, W, C], numpy.ndarray
        data = self.normalize(common_data)
        if not self.load_clip and not self.load_dino:
            return data

        # perform CLIP/DINO preprocessing
        if self.load_clip:
            data['clip_image'] = self.clip_normalize.normalize_image(img)
        if self.load_dino:
            data['dino_image'] = self.dino_normalize.normalize_image(img)

        return data
