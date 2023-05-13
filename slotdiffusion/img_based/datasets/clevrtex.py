import os
import os.path as osp
from pathlib import Path

import numpy as np
from PIL import Image

from torch.utils.data import Dataset

from nerv.utils import load_obj, dump_obj

from .utils import suppress_mask_idx, BaseTransforms


class DatasetReadError(ValueError):
    pass


class _CLEVRTEX(Dataset):
    """https://github.com/karazijal/clevrtex-generation/blob/main/clevrtex_eval.py"""

    splits = {'test': (0., 0.1), 'val': (0.1, 0.2), 'train': (0.2, 1.)}
    shape = (3, 240, 320)
    variants = {'full', 'pbg', 'vbg', 'grassbg', 'camo', 'outd'}

    def __init__(
        self,
        path,
        dataset_variant='full',
        split='train',
        crop=192,
        max_obj=-1,
    ):
        self.crop = crop
        if dataset_variant not in self.variants:
            raise DatasetReadError(
                f"Unknown variant {dataset_variant}; [{', '.join(self.variants)}] available "
            )

        if split not in self.splits:
            raise DatasetReadError(
                f"Unknown split {split}; [{', '.join(self.splits)}] available "
            )
        if dataset_variant == 'outd':
            # No dataset splits in
            split = None

        self.dataset_variant = dataset_variant
        self.split = split
        self.max_obj = max_obj

        self.basepath = Path(path)
        if not self.basepath.exists():
            raise DatasetReadError()
        sub_fold = self._variant_subfolder()
        if self.basepath.name != sub_fold:
            self.basepath = self.basepath / sub_fold
        self.index, self.mask_index = self._reindex()

        print(f"Sourced {dataset_variant} ({split}) from {self.basepath}")

        bias, limit = self.splits.get(split, (0., 1.))
        if isinstance(bias, float):
            bias = int(bias * len(self.index))
        if isinstance(limit, float):
            limit = int(limit * len(self.index))
        self.limit = limit
        self.bias = bias

    def _index_with_bias_and_limit(self, idx):
        if idx >= 0:
            idx += self.bias
            if idx >= self.limit:
                raise IndexError()
        else:
            idx = self.limit + idx
            if idx < self.bias:
                raise IndexError()
        return idx

    def _reindex(self):
        # try to read from files
        cur_dir = osp.dirname(osp.realpath(__file__))
        img_file_path = osp.join(cur_dir, 'splits/CLEVRTex',
                                 f'{self.dataset_variant}/img.pkl')
        msk_file_path = osp.join(cur_dir, 'splits/CLEVRTex',
                                 f'{self.dataset_variant}/msk.pkl')
        if self.max_obj > 0:
            img_file_path = img_file_path.replace('.pkl', f'-max_{self.max_obj}.pkl')
            msk_file_path = msk_file_path.replace('.pkl', f'-max_{self.max_obj}.pkl')
        if osp.exists(img_file_path) and osp.exists(msk_file_path):
            print('Loading index from json files')
            img_index = load_obj(img_file_path)
            msk_index = load_obj(msk_file_path)
            return img_index, msk_index

        print(f'Indexing {self.basepath}')

        img_index = {}
        msk_index = {}

        prefix = f"CLEVRTEX_{self.dataset_variant}_"

        img_suffix = ".png"
        msk_suffix = "_flat.png"

        _max = 0
        for img_path in self.basepath.glob(f'**/{prefix}??????{img_suffix}'):
            indstr = img_path.name.replace(prefix, '').replace(img_suffix, '')
            msk_path = img_path.parent / f"{prefix}{indstr}{msk_suffix}"
            indstr_stripped = indstr.lstrip('0')
            if indstr_stripped:
                ind = int(indstr)
            else:
                ind = 0
            if ind > _max:
                _max = ind

            if not msk_path.exists():
                raise DatasetReadError(f"Missing {msk_suffix.name}")

            if ind in img_index:
                raise DatasetReadError(f"Duplica {ind}")

            # filter out images with too many objects
            if self.max_obj > 0:
                msk = Image.open(msk_path)
                if self.crop > 0:
                    msk = self._center_crop(msk, int(self.crop))
                msk = np.array(msk)
                if np.unique(msk).shape[0] > self.max_obj + 1:
                    continue

            img_index[ind] = img_path
            msk_index[ind] = msk_path

        if len(img_index) == 0:
            raise DatasetReadError("No values found")
        missing = [i for i in range(0, _max) if i not in img_index]
        if missing:
            # raise DatasetReadError(f"Missing images numbers {missing}")
            print(f'Filter out {len(missing)} images')

        # save to file
        os.makedirs(osp.dirname(img_file_path), exist_ok=True)
        img_index = {i: str(v) for i, v in enumerate(img_index.values())}
        msk_index = {i: str(v) for i, v in enumerate(msk_index.values())}
        dump_obj(img_index, img_file_path)
        dump_obj(msk_index, msk_file_path)

        return img_index, msk_index

    def _variant_subfolder(self):
        return f"clevrtex_{self.dataset_variant.lower()}"

    def __len__(self):
        return self.limit - self.bias

    @staticmethod
    def _center_crop(img, crop_size):
        W, H = img.width, img.height
        img = img.crop((
            (W - crop_size) // 2,
            (H - crop_size) // 2,
            (W + crop_size) // 2,
            (H + crop_size) // 2,
        ))
        return img

    def _read_img(self, idx):
        img = Image.open(self.index[idx]).convert('RGB')
        if self.crop > 0:
            img = self._center_crop(img, int(self.crop))
        return img

    def _read_mask(self, idx):
        msk = Image.open(self.mask_index[idx])
        if self.crop > 0:
            msk = self._center_crop(msk, int(self.crop))
        return np.array(msk)

    def __getitem__(self, idx):
        raise NotImplementedError


class CLEVRTexDataset(_CLEVRTEX):

    def __init__(
        self,
        data_root,
        clevrtex_transforms,
        split='train',
        load_mask=True,
        max_obj=-1,
    ):
        super().__init__(
            path=data_root,
            dataset_variant='full',
            split=split,
            crop=192,
            max_obj=max_obj,
        )

        self.clevrtex_transforms = clevrtex_transforms
        self.load_mask = load_mask

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
            - masks (optional): [H, W]
            - error_flag: whether loading `idx` causes error and _rand_another
        """
        idx = self._index_with_bias_and_limit(idx)
        try:
            img = self._read_img(idx)
            img = self.clevrtex_transforms(img)
            if self.load_mask:
                mask = self._read_mask(idx)
                mask = self.clevrtex_transforms.process_mask(mask)
                mask = suppress_mask_idx(mask)
        except FileNotFoundError:
            return self._rand_another()
        data_dict = {
            'data_idx': idx,
            'img': img,
            'error_flag': False,
        }
        if self.load_mask:
            data_dict['masks'] = mask
        return data_dict


def build_clevrtex_dataset(params, val_only=False):
    """Build CelebA dataset that load images."""
    args = dict(
        data_root=params.data_root,
        clevrtex_transforms=BaseTransforms(params.resolution),
        split='val',
        load_mask=params.load_mask,
        max_obj=params.get('max_obj', -1),
    )
    if val_only:
        print('Using CLEVRTex test set!')
        args['split'] = 'test'
    val_dataset = CLEVRTexDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = CLEVRTexDataset(**args)
    return train_dataset, val_dataset
