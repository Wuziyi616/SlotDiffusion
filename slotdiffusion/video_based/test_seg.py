import os
import sys
import importlib
import argparse
from tqdm import tqdm

import torch

from nerv.utils import AverageMeter
from nerv.training import BaseDataModule

from models import build_model
from models import ARI_metric, fARI_metric, miou_metric, fmiou_metric, mbo_metric
from datasets import build_dataset


@torch.no_grad()
def eval_masks(pred_masks, gt_masks):
    """Calculate seg_mask related metrics.

    Args:
        pred_masks: [B(, T), H, W], argmaxed predicted masks.
        gt_masks: [B(, T), H, W], argmaxed GT
    """
    # need to consider the temporal consistency of seg_masks
    # absorb `T` into spatial dims so that object ID should be the same
    if len(pred_masks.shape) == 4:
        pred_masks = pred_masks.flatten(1, 2)
    if len(gt_masks.shape) == 4:
        gt_masks = gt_masks.flatten(1, 2)
    # in the paper, we report FG-ARI, mIoU and mBO
    # we also provide other metrics to evaluate
    metrics = [
        # 'ARI',
        'fARI',
        'miou',
        'mbo',
        # 'fmiou',
    ]
    ret_dict = {m: eval(f'{m}_metric')(gt_masks, pred_masks) for m in metrics}
    return ret_dict


@torch.no_grad()
def main(params):
    val_set = build_dataset(params, val_only=True)
    datamodule = BaseDataModule(
        params, train_set=None, val_set=val_set, use_ddp=False)
    val_loader = datamodule.val_loader

    results = {}
    for data_dict in tqdm(val_loader):
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        out_dict = model(data_dict)
        pred_masks = out_dict['masks']  # [B, T, N(, 1), H, W]
        if len(pred_masks.shape) == 6:
            pred_masks = pred_masks.squeeze(-3)
        pred_masks = pred_masks.argmax(dim=-3)  # [B, T, H, W]
        gt_masks = data_dict['masks'].long()  # [B, T, H, W]
        mask_dict = eval_masks(pred_masks, gt_masks)
        B = pred_masks.shape[0]

        if not results:
            results = {k: AverageMeter() for k in mask_dict}
        for k, v in mask_dict.items():
            results[k].update(v, B)

        torch.cuda.empty_cache()

    print(f'{os.path.basename(args.params)}, L={params.n_sample_frames}')
    for k, v in results.items():
        print(f'{k}: {(v.avg * 100.):.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test video segmentation')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--seq_len', nargs='+', default=[-1], type=int)
    parser.add_argument('--bs', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotAttentionParams()

    # build model
    model = build_model(params).eval().cuda()
    model.load_weight(args.weight)
    print(f'Loading weight from {args.weight}')

    # for SlotDiffusion and STEVE, masks are obtained without calling decoder
    # but for SA, decoding slots to masks is done by the decoder
    if params.model != 'SAVi':
        model.testing = True

    torch.backends.cudnn.benchmark = True
    if args.bs > 0:
        params.val_batch_size = args.bs

    # test over multiple seq_len
    all_seq_lens = args.seq_len
    for seq_len in all_seq_lens:
        # adjust video clip length to evaluate on
        if seq_len > 0:
            params.n_sample_frames = seq_len
        # use full video length by default
        else:
            params.n_sample_frames = params.video_len  # entire video

        main(params)
