import os
import sys
import importlib
import argparse

import numpy as np
from PIL import Image

import torch
import lpips

from nerv.training import BaseDataModule
from nerv.utils import load_obj, dump_obj

from models import build_model, to_rgb_from_tensor
from models import mse_metric, psnr_metric, ssim_metric, perceptual_dist
from datasets import build_dataset
from method import build_method

vgg_fn = lpips.LPIPS(net='vgg')
vgg_fn = torch.nn.DataParallel(vgg_fn).cuda().eval()


@torch.no_grad()
def recon_imgs(model, gt_imgs, model_type):
    """Encode imgs to slots and decode them back to imgs."""
    data_dict = {'img': gt_imgs}
    if model_type == 'SA':
        out_dict = model(data_dict)
        pred_imgs = out_dict['recon_img']
    elif model_type == 'SLATE':
        out_dict = model(data_dict)
        pred_imgs = model(out_dict, recon_img=True, bs=24, verbose=False)
    else:
        pred_imgs = model(
            data_dict,
            log_images=True,
            use_ddim=False,
            use_dpm=True,  # we use DPM-Solver by default
            ret_intermed=False,
            verbose=False,
        )['samples']
    return pred_imgs.cuda()


@torch.no_grad()
def eval_imgs(pred_imgs, gt_imgs):
    """Calculate image quality related metrics.

    Args:
        pred_imgs: [B, 3, H, W], [-1, 1]
        gt_imgs: [B, 3, H, W], [-1, 1]
    """
    lpips = perceptual_dist(pred_imgs, gt_imgs, vgg_fn)
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()
    mse = mse_metric(pred_imgs, gt_imgs)
    psnr = psnr_metric(pred_imgs, gt_imgs)
    ssim = ssim_metric(pred_imgs, gt_imgs)
    ret_dict = {'lpips': lpips, 'mse': mse, 'psnr': psnr, 'ssim': ssim}
    return ret_dict


@torch.no_grad()
def test_recon_imgs(model, data_dict, model_type, save_dir):
    """Encode images to slots and then decode back to images."""
    torch.cuda.empty_cache()
    data_idx = data_dict['data_idx'].cpu().numpy()  # [B]
    # reconstruction evaluation can be very time-consuming
    # so we save the results for every sample, so that we don't need to
    #     recompute them when we re-start the evaluation next time
    # load metrics if already computed
    metric_fn = os.path.join(save_dir, f'metrics/{data_idx[0]}_metric.pkl')
    if os.path.exists(metric_fn):
        recon_dict = load_obj(metric_fn)
    else:
        gt_imgs = data_dict['img']
        pred_imgs = recon_imgs(model, gt_imgs, model_type)
        recon_dict = eval_imgs(pred_imgs, gt_imgs)  # np.array
        # save metrics and images for computing FID later
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
    # this takes some space, if you don't need FID, you can comment it out
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, 3, H, W]
    pred_imgs = np.round(pred_imgs * 255.).astype(np.uint8)
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()  # [B, 3, H, W]
    gt_imgs = np.round(gt_imgs * 255.).astype(np.uint8)
    pred_dir = os.path.join(save_dir, 'recon_imgs')
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
    val_set = build_dataset(params, val_only=True)  # actually the test set
    datamodule = BaseDataModule(
        params, train_set=None, val_set=val_set, use_ddp=params.ddp)

    model = build_model(params)
    model.load_weight(args.weight)
    print(f'Loading weight from {args.weight}')

    # for SlotDiffusion and SLATE, we only want slots
    # and then we will call additional functions to reconstruct the imgs
    # but for SA the slot2img decoding is already included in the forward pass
    if params.model != 'SA':
        model.testing = True

    # reconstruction evaluation can be very time-consuming
    # so we leverage DDP to speed up
    # in order to do that, we build `method` which implements DDP
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
    method._test_step = lambda model, batch_data: test_recon_imgs(
        model, batch_data, model_type=params.model, save_dir=save_dir)
    method.test()
    print(f'{os.path.basename(args.params)} testing done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test image reconstruction')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--bs', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--local-rank', type=int, default=0)
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
