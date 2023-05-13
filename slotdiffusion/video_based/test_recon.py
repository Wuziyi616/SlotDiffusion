import os
import sys
import importlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import lpips

from nerv.training import BaseDataModule
from nerv.utils import load_obj, dump_obj, convert4save

from models import build_model, to_rgb_from_tensor
from models import mse_metric, psnr_metric, ssim_metric, perceptual_dist
from datasets import build_dataset
from method import build_method
from vis import save_video

vgg_fn = lpips.LPIPS(net='vgg')
vgg_fn = torch.nn.DataParallel(vgg_fn).cuda().eval()


@torch.no_grad()
def recon_imgs(model, gt_imgs, model_type):
    """Encode videos to slots and decode them back to videos."""
    data_dict = {'img': gt_imgs}
    if model_type == 'SAVi':
        out_dict = model(data_dict)
        pred_imgs = out_dict['recon_img']
    elif model_type == 'STEVE':
        out_dict = model(data_dict)
        pred_imgs = model(out_dict, recon_img=True, bs=24, verbose=False)
    else:
        pred_imgs = model(
            data_dict,
            log_images=True,
            use_ddim=False,
            use_dpm=True,  # we use DPM-Solver by default
            same_noise=True,
            ret_intermed=False,
            verbose=False,
        )['samples']
    return pred_imgs.cuda()


@torch.no_grad()
def eval_imgs(pred_imgs, gt_imgs):
    """Calculate image quality related metrics.

    Args:
        pred_imgs: [B(, T), 3, H, W], [-1, 1]
        gt_imgs: [B(, T), 3, H, W], [-1, 1]
    """
    if len(pred_imgs.shape) == 5:
        pred_imgs = pred_imgs.flatten(0, 1)
    if len(gt_imgs.shape) == 5:
        gt_imgs = gt_imgs.flatten(0, 1)
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
        # save metrics and images for computing FVD later
        save_results(data_idx, pred_imgs, gt_imgs, recon_dict, save_dir)
    recon_dict = {k: torch.tensor(v) for k, v in recon_dict.items()}
    return recon_dict


def save_results(data_idx, pred_imgs, gt_imgs, metrics_dict, save_dir):
    """Save metrics and videos to individual frames."""
    # metrics
    metric_fn = os.path.join(save_dir, f'metrics/{data_idx[0]}_metric.pkl')
    os.makedirs(os.path.dirname(metric_fn), exist_ok=True)
    dump_obj(metrics_dict, metric_fn)
    # save videos as separate frames, required by FVD
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, T, 3, H, W]
    pred_imgs = np.round(pred_imgs * 255.).astype(np.uint8)
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()  # [B, T, 3, H, W]
    gt_imgs = np.round(gt_imgs * 255.).astype(np.uint8)
    for i, (pred_vid, gt_vid) in enumerate(zip(pred_imgs, gt_imgs)):
        idx = data_idx[i]
        # [T, 3, H, W] -> [T, H, W, 3]
        pred_vid = pred_vid.transpose(0, 2, 3, 1)
        gt_vid = gt_vid.transpose(0, 2, 3, 1)
        pred_dir = os.path.join(save_dir, 'recon_vids', str(idx))
        gt_dir = os.path.join(save_dir, 'gt_vids', str(idx))
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        for t, (pred, gt) in enumerate(zip(pred_vid, gt_vid)):
            plt.imsave(os.path.join(pred_dir, f'{t}.png'), pred)
            plt.imsave(os.path.join(gt_dir, f'{t}.png'), gt)


@torch.no_grad()
def save_recon_imgs(model, data_dict, model_type, save_dir):
    """Save the recon video instead of computing metrics."""
    gt_imgs = data_dict['img']
    # fake a metric_dict for compatibility
    recon_dict = {'mse': torch.tensor(0.).type_as(gt_imgs)}
    pred_imgs = recon_imgs(model, gt_imgs, model_type)
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, T, 3, H, W]
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()  # [B, T, 3, H, W]
    data_idx = data_dict['data_idx']  # [B]
    for i, (pred, gt) in enumerate(zip(pred_imgs, gt_imgs)):
        idx = data_idx[i].cpu().item()
        pred, gt = convert4save(pred), convert4save(gt)
        save_video(pred, os.path.join(save_dir, f'{idx}_recon.mp4'), fps=8)
        save_video(gt, os.path.join(save_dir, f'{idx}_gt.mp4'), fps=8)
    # exit program when saving enough many samples
    if len(os.listdir(save_dir)) > 200:
        exit(-1)
    return recon_dict


@torch.no_grad()
def main(params):
    val_set = build_dataset(params, val_only=True)  # actually test set
    datamodule = BaseDataModule(
        params, train_set=None, val_set=val_set, use_ddp=params.ddp)

    model = build_model(params)
    model.load_weight(args.weight)
    print(f'Loading weight from {args.weight}')

    # for SlotDiffusion and STEVE, we only want slots
    # and then we will call additional functions to reconstruct the imgs
    # but for SAVi the slot2img decoding is already included in the forward pass
    if params.model != 'SAVi':
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

    if args.save_video:  # save to mp4 for visualization
        save_dir = os.path.join(os.path.dirname(args.weight), 'vis')
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        method._test_step = lambda model, batch_data: \
            save_recon_imgs(model, batch_data, model_type=params.model,
                            save_dir=save_dir)
    else:  # save to individual frames for FVD computation
        save_dir = os.path.join(os.path.dirname(args.weight), 'eval')
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        method._test_step = lambda model, batch_data: \
            test_recon_imgs(model, batch_data, model_type=params.model,
                            save_dir=save_dir)
    method.test()
    print(f'{os.path.basename(args.params)} testing done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test video reconstruction')
    parser.add_argument('--params', type=str, required=True)
    parser.add_argument('--weight', type=str, default='', help='load weight')
    parser.add_argument('--save_video', action='store_true', help='vis videos')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()

    if args.params.endswith('.py'):
        args.params = args.params[:-3]
    sys.path.append(os.path.dirname(args.params))
    params = importlib.import_module(os.path.basename(args.params))
    params = params.SlotAttentionParams()
    assert params.model in ['SAVi', 'SAViDiffusion', 'STEVE'], \
        f'Unknown model {params.model}'

    # load full video for eval
    params.n_sample_frames = params.video_len  # entire video

    # adjust bs & GPU settings
    params.val_batch_size = args.bs  # DDP
    params.gpus = torch.cuda.device_count()
    params.ddp = (params.gpus > 1)

    torch.backends.cudnn.benchmark = True
    main(params)
