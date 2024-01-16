import cv2
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
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']

        if np.random.uniform(0, 1) < self.prob:
            image = np.ascontiguousarray(image[:, ::-1, :])
            masks = np.ascontiguousarray(masks[:, ::-1])
            _, w, _ = image.shape
            # adjust annos
            if annos.shape[0] > 0:
                x1 = annos[:, 0].copy()
                x2 = annos[:, 2].copy()
                annos[:, 0] = w - x2
                annos[:, 2] = w - x1

        return {
            'image': image,
            'masks': masks,
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class ResizeMinShape:
    """Resize for later center crop."""

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def __call__(self, sample):
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']
        h, w, _ = image.shape
        # resize so that the h' is at lease resolution[0]
        # and the w' is at lease resolution[1]
        factor = max(self.resolution[0] / h, self.resolution[1] / w)
        resize_h, resize_w = int(round(h * factor)), int(round(w * factor))
        image = cv2.resize(
            image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        masks = cv2.resize(
            masks, (resize_w, resize_h), interpolation=cv2.INTER_NEAREST)
        # adjust annos
        factor = float(factor)
        annos[:, :4] *= factor
        scale *= factor
        return {
            'image': image,
            'masks': masks,
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class CenterCrop:
    """Crop the center square of the image."""

    def __init__(self, resolution=(224, 224)):
        self.resolution = resolution

    def __call__(self, sample):
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']

        h, w, _ = image.shape
        assert h >= self.resolution[0] and w >= self.resolution[1]
        assert h == self.resolution[0] or w == self.resolution[1]

        if h == self.resolution[0]:
            crop_ymin = 0
            crop_ymax = h
            crop_xmin = (w - self.resolution[0]) // 2
            crop_xmax = crop_xmin + self.resolution[0]
        else:
            crop_xmin = 0
            crop_xmax = w
            crop_ymin = (h - self.resolution[1]) // 2
            crop_ymax = crop_ymin + self.resolution[1]
        image = image[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        masks = masks[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
        # adjust annos
        if annos.shape[0] > 0:
            annos[:, [0, 2]] = annos[:, [0, 2]] - crop_xmin
            annos[:, [1, 3]] = annos[:, [1, 3]] - crop_ymin
            # filter out annos that are out of the image
            mask1 = (annos[:, 2] > 0) & (annos[:, 3] > 0)
            mask2 = (annos[:, 0] < self.resolution[0]) & \
                (annos[:, 1] < self.resolution[1])
            annos = annos[mask1 & mask2]
            annos[:, [0, 2]] = np.clip(annos[:, [0, 2]], 0, self.resolution[0])
            annos[:, [1, 3]] = np.clip(annos[:, [1, 3]], 0, self.resolution[1])

        return {
            'image': image,
            'masks': masks,
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class Normalize:
    """Normalize the image with mean and std."""

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
        image, masks, annos, scale, size = sample['image'], sample['masks'], \
            sample['annos'], sample['scale'], sample['size']
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
            'annos': annos,
            'scale': scale,
            'size': size,
        }


class COCOCollater:
    """Collect images, annotations, etc. into a batch."""

    def __init__(self):
        pass

    def __call__(self, data):
        images = [s['image'] for s in data]
        masks = [s['masks'] for s in data]
        annos = [s['annos'] for s in data]
        scales = [s['scale'] for s in data]
        sizes = [s['size'] for s in data]

        images = np.stack(images, axis=0)  # [B, H, W, C]
        images = torch.from_numpy(images).permute(0, 3, 1, 2)  # [B, C, H, W]

        masks = np.stack(masks, axis=0)
        masks = torch.from_numpy(masks)  # [B, H, W(, 2 or 3)]

        max_annos_num = max(anno.shape[0] for anno in annos)
        if max_annos_num > 0:
            input_annos = np.ones(
                (len(annos), max_annos_num, 5), dtype=np.float32) * (-1)
            for i, anno in enumerate(annos):
                if anno.shape[0] > 0:
                    input_annos[i, :anno.shape[0], :] = anno
        else:
            input_annos = np.ones((len(annos), 1, 5), dtype=np.float32) * (-1)
        input_annos = torch.from_numpy(input_annos).float()

        scales = torch.from_numpy(np.array(scales)).float()
        sizes = torch.from_numpy(np.array(sizes)).float()

        data_dict = {
            'img': images.contiguous().float(),
            'masks': masks.contiguous().long(),
            'annos': input_annos,
            'scale': scales,
            'size': sizes,
        }
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


class COCOTransforms(object):
    """Data pre-processing steps."""

    def __init__(
        self,
        resolution,
        norm_mean=0.5,
        norm_std=0.5,
        val=False,
    ):
        self.normalize = Normalize(norm_mean, norm_std)
        self.transforms = transforms.Compose([
            RandomHorizontalFlip(0.5 if not val else 0),
            ResizeMinShape(resolution),
            CenterCrop(resolution),
            self.normalize,
        ])
        self.resolution = resolution

    def __call__(self, input):
        return self.transforms(input)
