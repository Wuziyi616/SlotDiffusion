from .physion import PhysionDataset, build_physion_dataset
from .movi import build_movi_dataset
from .steve_movi import build_steve_movi_dataset  # MOVi-Solid/Tex


def build_dataset(params, val_only=False):
    dst = params.dataset
    if 'physion' in dst:
        return build_physion_dataset(params, val_only=val_only)
    return eval(f"build_{dst}_dataset")(params, val_only=val_only)
