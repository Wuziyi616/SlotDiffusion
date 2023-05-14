import numpy as np

import torch

PALETTE = [[174, 199, 232], [152, 223, 138], [31, 119, 180], [255, 187, 120],
           [188, 189, 34], [140, 86, 75], [255, 152, 150], [214, 39, 40],
           [197, 176, 213], [148, 103, 189], [196, 156, 148], [23, 190, 207],
           [247, 182, 210], [219, 219, 141], [255, 127, 14], [158, 218, 229],
           [44, 160, 44], [112, 128, 144], [227, 119, 194], [82, 84, 163]]

# np palette in [0, 1]
np_palette = np.array(PALETTE) / 255.0

# torch palette in [-1, 1]
torch_palette = torch.from_numpy(np_palette).float() * 2. - 1.


def torch_draw_mask(mask):
    """Draw mask with palette.

    Args:
        mask (torch.Tensor): shape [B(, T), H, W]

    Returns:
        torch.Tensor: shape [B(, T), 3, H, W]
    """
    mask = mask.long()
    if len(mask.shape) == 4:
        unflatten = True
        B, T = mask.shape[:2]
        mask = mask.flatten(0, 1)
    else:
        unflatten = False
    colored_mask = torch_palette.to(mask.device)[mask]  # [B', H, W, 3]
    colored_mask = colored_mask.permute(0, 3, 1, 2)  # [B', 3, H, W]
    if unflatten:
        colored_mask = colored_mask.unflatten(0, (B, T))
    return colored_mask


def torch_draw_rgb_mask(rgb, mask, alpha=0.7):
    """Map mask to rgb image."""
    assert 0. < alpha < 1.
    colored_mask = torch_draw_mask(mask)
    return rgb * alpha + colored_mask * (1. - alpha), colored_mask
