import numpy as np

from torchvision.datasets import CelebA

from .utils import BaseTransforms


class CelebADataset(CelebA):
    """Dataset for loading CelebA images."""

    def __init__(
        self,
        data_root,
        celeba_transform,
        split='train',
    ):
        if split == 'val':
            split = 'valid'
        super().__init__(
            root=data_root,
            split=split,
            target_type=[],
            transform=celeba_transform,
            target_transform=None,
            download=False,
        )

    def _rand_another(self):
        """Random get another sample when encountering loading error."""
        another_idx = np.random.choice(len(self))
        data_dict = self.__getitem__(another_idx)
        data_dict['error_flag'] = True
        return data_dict

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - img: [C, H, W]
            - error_flag: whether loading `idx` causes error and _rand_another
        """
        try:
            img, _ = super().__getitem__(idx)
        except FileNotFoundError:
            return self._rand_another()
        return {
            'data_idx': idx,
            'img': img,
            'error_flag': False,
        }


def build_celeba_dataset(params, val_only=False):
    """Build CelebA dataset that load images."""
    args = dict(
        data_root=params.data_root,
        celeba_transform=BaseTransforms(params.resolution),
        split='val',
    )
    if val_only:
        print('Using CelebA test set!')
        args['split'] = 'test'
    val_dataset = CelebADataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = CelebADataset(**args)
    return train_dataset, val_dataset
