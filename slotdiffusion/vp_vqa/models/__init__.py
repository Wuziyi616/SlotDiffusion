from .ldm_slotformer import LDMSlotFormer
from .readout import PhysionReadout

from .utils import get_lr, to_rgb_from_tensor


def build_model(params):
    if params.model == 'LDMSlotFormer':  # stands for latent diffusion model
        return LDMSlotFormer(
            resolution=params.resolution,
            clip_len=params.input_frames,
            slot_dict=params.slot_dict,
            dec_dict=params.dec_dict,
            rollout_dict=params.rollout_dict,
            loss_dict=params.loss_dict,
        )
    elif params.model == 'PhysionReadout':
        return PhysionReadout(params.readout_dict)
    else:
        raise NotImplementedError(f'{params.model} is not implemented.')
