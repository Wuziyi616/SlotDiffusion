import os
import sys
import importlib
import argparse

import numpy as np
from PIL import Image

import torch

from nerv.training import BaseDataModule
from nerv.utils import load_obj, dump_obj

from models import build_model, to_rgb_from_tensor
from datasets import build_dataset
from method import build_method


@torch.no_grad()
def gen_imgs(model, gt_imgs, model_type):
    """Encode imgs to slots and decode them back to imgs."""
    data_dict = {'img': gt_imgs}
    slots = model(data_dict)['slots']  # [B, N, C]
    # combine slots from different images to generate new images
    # switching the slots, we will shift the slots by 1 data item
    # i.e. data0 will have slot[0, 0], slot[1, 1], slot[2, 2], ...
    #      data1 will have slot[1, 0], slot[2, 1], slot[3, 2], ...
    #      data2 will have slot[2, 0], slot[3, 1], slot[4, 2], ...
    N = slots.shape[1]
    for i in range(N):
        slots[:, i] = torch.cat([slots[i:, i], slots[:i, i]])
    if model_type == 'SA':
        pred_imgs, _, _, _ = model.decode(slots)
    elif model_type == 'SLATE':
        data_dict = {'slots': slots}
        pred_imgs = model(data_dict, recon_img=True, bs=24, verbose=False)
    else:
        ddpm_dict = {'slots': slots, 'img': gt_imgs}
        pred_imgs = model.dm_decoder.log_images(
            ddpm_dict,
            use_ddim=False,
            use_dpm=True,  # we use DPM-Solver by default
            ret_intermed=False,
            verbose=False,
        )['samples']
    return pred_imgs.cuda()


@torch.no_grad()
def test_gen_imgs(model, data_dict, model_type, save_dir):
    """Encode images to slots and then decode back to images."""
    torch.cuda.empty_cache()
    data_idx = data_dict['data_idx'].cpu().numpy()  # [B]
    # similar to reconstruction, compositional generation can also be
    #     time-consuming, so we save the results to disk
    metric_fn = os.path.join(save_dir, f'metrics/{data_idx[0]}_metric.pkl')
    if os.path.exists(metric_fn):
        recon_dict = load_obj(metric_fn)
    else:
        gt_imgs = data_dict['img']
        pred_imgs = gen_imgs(model, gt_imgs, model_type)
        recon_dict = {'mse': 0.}  # fake a return dict for compatibility
        # save all generated images to compute FID later
        save_results(data_idx, pred_imgs, gt_imgs, recon_dict, save_dir)
    recon_dict = {k: torch.tensor(v) for k, v in recon_dict.items()}
    return recon_dict


def save_results(data_idx, pred_imgs, gt_imgs, metrics_dict, save_dir):
    """Save metrics and images."""
    # metrics
    metric_fn = os.path.join(save_dir, f'metrics/{data_idx[0]}_metric.pkl')
    os.makedirs(os.path.dirname(metric_fn), exist_ok=True)
    dump_obj(metrics_dict, metric_fn)
    # images
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, 3, H, W]
    pred_imgs = np.round(pred_imgs * 255.).astype(np.uint8)
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()  # [B, 3, H, W]
    gt_imgs = np.round(gt_imgs * 255.).astype(np.uint8)
    pred_dir = os.path.join(save_dir, 'comp_imgs')
    gt_dir = os.path.join(save_dir, 'gt_imgs')
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    for i, (pred, gt) in enumerate(zip(pred_imgs, gt_imgs)):
        idx = data_idx[i]
        pred = pred.transpose(1, 2, 0)
        gt = gt.transpose(1, 2, 0)
        Image.fromarray(pred).save(os.path.join(pred_dir, f'{idx}.png'))
        Image.fromarray(gt).save(os.path.join(gt_dir, f'{idx}.png'))


@torch.no_grad()
def main(params):
    val_set = build_dataset(params, val_only=True)  # actually test set
    datamodule = BaseDataModule(
        params, train_set=None, val_set=val_set, use_ddp=params.ddp)

    model = build_model(params)
    model.load_weight(args.weight)
    print(f'Loading weight from {args.weight}')

    # we only want slots since we'll compose them manually
    model.testing = True

    # DDP to speed up testing
    method = build_method(
        model=model,
        datamodule=datamodule,
        params=params,
        ckp_path=os.path.dirname(args.weight),
        local_rank=args.local_rank,
        use_ddp=params.ddp,
        use_fp16=False,
        val_only=True,
    )

    save_dir = os.path.join(os.path.dirname(args.weight), 'eval')
    if args.local_rank == 0:
        os.makedirs(save_dir, exist_ok=True)
    method._test_step = lambda model, batch_data: test_gen_imgs(
        model, batch_data, model_type=params.model, save_dir=save_dir)
    method.test()
    print(f'{os.path.basename(args.params)} testing done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test image generation')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotAttentionParams()
    assert params.model in ['SA', 'SADiffusion', 'SLATE'], \
        f'Unknown model {params.model}'

    # adjust bs & GPU settings
    if args.bs > 0:
        params.val_batch_size = args.bs  # DDP
    params.gpus = torch.cuda.device_count()
    params.ddp = (params.gpus > 1)

    torch.backends.cudnn.benchmark = True
    main(params)
