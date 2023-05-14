from PIL import Image
import requests

import torch.nn as nn

from transformers import ViTFeatureExtractor, ViTModel


def dino(patch_size: int, small_size: bool = True):
    if patch_size == 8:
        resolution = 28
    elif patch_size == 16:
        resolution = 14
    else:
        raise ValueError(f'patch_size {patch_size} not supported')
    model = DINOEncoder(
        resolution=resolution, patch_size=patch_size, small_size=small_size)
    return model


class DINOEncoder(nn.Module):
    """A wrapper for DINO."""

    def __init__(self,
                 resolution: int,
                 patch_size: int = 8,
                 small_size: bool = True):
        super().__init__()

        self.resolution = resolution
        self.patch_size = patch_size
        self.small_size = small_size
        if self.small_size:
            self.version = 's'
        else:
            self.version = 'b'

        self.dino = ViTModel.from_pretrained(
            f'facebook/dino-vit{self.version}{patch_size}')
        for p in self.dino.parameters():
            p.requires_grad = False

    def forward(self, x):
        # Reduce the CLS token
        out = self.dino(x).last_hidden_state[:, 1:, :]
        # out has shape: [B, H*W, C]

        out = out.reshape(out.shape[0], self.resolution, self.resolution,
                          out.shape[-1])
        # out has shape: [B, H, W, C]

        out = out.permute(0, 3, 1, 2)
        # out has shape: [B, C, H, W]
        return out

    def train(self, mode=True):
        nn.Module.train(self, mode)
        # freeze the DINO model
        self.dino.eval()
        return self


if __name__ == '__main__':
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw)

    feature_extractor = ViTFeatureExtractor.from_pretrained(
        'facebook/dino-vits8')
    encoder = eval('dino_vits8')()
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = encoder(inputs)
    last_hidden_states = outputs.last_hidden_state[:, 1:, :]
    print(last_hidden_states.shape)
