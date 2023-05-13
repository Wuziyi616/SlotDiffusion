import os
import cv2
import numpy as np

import torch

from nerv.utils import convert4save


def save_video(video, save_path, fps=30, codec='mp4v'):
    """video: np.ndarray of shape [M, H, W, 3]"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    video = convert4save(video, is_video=True)
    video = video[..., [2, 1, 0]]
    H, W = video.shape[-3:-1]
    assert save_path.split('.')[-1] == 'mp4'  # save as mp4 file
    # opencv has opposite dimension definition as numpy
    size = [W, H]
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*codec), fps, size)
    for i in range(video.shape[0]):
        out.write(video[i])
    out.release()
