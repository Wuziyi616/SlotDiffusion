import os
import sys
import importlib
import argparse
from tqdm import tqdm

import torch
from torch.utils.data._utils.collate import default_collate

from nerv.utils import AverageMeter
from nerv.training import BaseDataModule

from models import build_model
from models import ARI_metric, fARI_metric, miou_metric, fmiou_metric, \
    mbo_metric
from datasets import build_dataset


@torch.no_grad()
def eval_masks(pred_masks, gt_masks, inst_overlap_masks=None):
    """Calculate seg_mask related metrics.

    Args:
        pred_masks: [B, N, H, W], argmaxed predicted masks.
        gt_masks: [B, H, W], argmaxed GT
    """
    pred_masks = pred_masks.argmax(dim=-3)  # [B, N, H, W] --> [B, H, W]
    gt_masks = gt_masks.long()
    # in the paper, we report FG-ARI, mIoU and mBO
    # we also provide other metrics to evaluate
    metrics = [
        # 'ARI',
        'fARI',
        'miou',
        # 'fmiou',
        'mbo',
    ]
    ret_dict = {
        m: eval(f'{m}_metric')(gt_masks, pred_masks, inst_overlap_masks)
        for m in metrics
    }
    return ret_dict


@torch.no_grad()
def main(params):
    if params.dataset.lower() in ['coco', 'voc']:
        val_set, collate_fn = build_dataset(params, val_only=True)
        val_set.load_sem_mask = True  # also load semantic seg masks
    else:
        val_set = build_dataset(params, val_only=True)
        collate_fn = default_collate
    datamodule = BaseDataModule(
        params,
        train_set=None,
        val_set=val_set,
        use_ddp=False,
        collate_fn=collate_fn,
    )
    val_loader = datamodule.val_loader

    model = build_model(params).eval().cuda()
    model.load_weight(args.weight)
    print(f'Loading weight from {args.weight}')

    # for SlotDiffusion and SLATE, masks are obtained without calling decoder
    # but for SA, decoding slots to masks is done by the decoder
    if params.model != 'SA':
        model.testing = True

    results = {}
    for data_dict in tqdm(val_loader):
        data_dict = {k: v.cuda() for k, v in data_dict.items()}
        out_dict = model(data_dict)
        pred_masks = out_dict['masks']  # [B, N(, 1), H, W]
        if len(pred_masks.shape) == 5:
            pred_masks = pred_masks.squeeze(-3)
        gt_masks = data_dict['masks'].long()  # [B, H, W]
        # on VOC and COCO, some pixels are ignored during evaluation
        inst_overlap_masks = data_dict.get('inst_overlap_masks', None)
        mask_dict = eval_masks(pred_masks, gt_masks, inst_overlap_masks)
        # on VOC and COCO, we compute metrics with semantic and instance masks
        if params.dataset.lower() in ['coco', 'voc']:
            mask_dict = {f'inst/{k}': v for k, v in mask_dict.items()}
            gt_sem_masks = data_dict['sem_masks']
            sem_mask_dict = eval_masks(pred_masks, gt_sem_masks,
                                       inst_overlap_masks)
            sem_mask_dict = {f'sem/{k}': v for k, v in sem_mask_dict.items()}
            mask_dict.update(sem_mask_dict)

        B = pred_masks.shape[0]
        if not results:
            results = {k: AverageMeter() for k in mask_dict}
        for k, v in mask_dict.items():
            results[k].update(v, B)

        torch.cuda.empty_cache()

    print(os.path.basename(args.params))
    for k, v in results.items():
        print(f'{k}: {(v.avg * 100.):.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test image segmentation')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--bs', type=int, default=-1)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotAttentionParams()

    if args.bs > 0:
        params.val_batch_size = args.bs

    torch.backends.cudnn.benchmark = True
    main(params)
