from .slot_attention import SA
from .slate import build_model as build_slate_model
from .sa_diffusion import SADiffusion
from .vqvae import VQVAE

from .utils import get_lr, to_rgb_from_tensor
from .eval_utils import ARI_metric, fARI_metric, miou_metric, fmiou_metric, \
    mbo_metric, mse_metric, psnr_metric, ssim_metric, perceptual_dist
from .slate import cosine_anneal, gumbel_softmax, make_one_hot


def build_model(params):
    """Build model."""
    if params.model == 'VQVAE':
        model = VQVAE(
            enc_dec_dict=params.enc_dec_dict,
            vq_dict=params.vq_dict,
        )
    elif params.model == 'SA':
        model = SA(
            resolution=params.resolution,
            slot_dict=params.slot_dict,
            enc_dict=params.enc_dict,
            dec_dict=params.dec_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model in ['dVAE', 'SLATE']:
        model = build_slate_model(params)
    elif params.model == 'SADiffusion':
        model = SADiffusion(
            resolution=params.resolution,
            slot_dict=params.slot_dict,
            enc_dict=params.enc_dict,
            dec_dict=params.dec_dict,
            loss_dict=params.loss_dict,
        )
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
    return model
