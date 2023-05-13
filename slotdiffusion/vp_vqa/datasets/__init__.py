from .physion import build_physion_dataset, build_physion_slots_dataset, \
    build_physion_slots_label_dataset, build_physion_tokens_dataset


def build_dataset(params, val_only=False):
    dst = params.dataset
    if 'physion' not in dst:
        return eval(f'build_{dst}_dataset')(params, val_only=val_only)
    # physion dataset looks like 'physion_xxx_$SUBSET'
    return eval(f"build_{dst[:dst.rindex('_')]}_dataset")(
        params, val_only=val_only)
