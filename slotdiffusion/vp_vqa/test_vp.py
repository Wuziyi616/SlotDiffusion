import os
import sys
import importlib
import argparse

import numpy as np
from PIL import Image

import torch
import lpips

from nerv.training import BaseDataModule
from nerv.utils import load_obj, dump_obj, save_video

from models import build_model, to_rgb_from_tensor
from models import mse_metric, psnr_metric, ssim_metric, perceptual_dist
from datasets import build_dataset
from method import build_method

vgg_fn = lpips.LPIPS(net='vgg')
vgg_fn = torch.nn.DataParallel(vgg_fn).cuda().eval()


@torch.no_grad()
def vp_imgs(model, gt_inputs, pred_len, model_type):
    """Encode imgs to slots and decode them back to imgs."""
    if model_type == 'LDMSlotFormer':
        pred_slots = model.rollout(gt_inputs, pred_len)  # [B, pred_len, N, C]
        B, T = pred_slots.shape[:2]
        data_dict = {'slots': pred_slots.flatten(0, 1)}
        pred_imgs = model.log_images(
            data_dict,
            use_dpm=True,
            verbose=False,
        )['samples'].unflatten(0, (B, T))
    else:
        raise NotImplementedError(f'Unknown model type {model_type}')
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
def test_vp_imgs(model, data_dict, model_type, save_dir):
    """Encode images to slots and then decode back to images."""
    torch.cuda.empty_cache()
    data_idx = data_dict['data_idx'].cpu().numpy()  # [B]
    metric_fn = os.path.join(save_dir, f'metrics/{data_idx[0]}_metric.pkl')
    if os.path.exists(metric_fn):
        recon_dict = load_obj(metric_fn)
    else:
        gt_imgs = data_dict['img'][:, model.history_len:]
        pred_len = model.rollout_len
        if model_type == 'LDMSlotFormer':
            gt_inputs = data_dict['slots'][:, :model.history_len]
        else:
            raise NotImplementedError(f'Unknown model type {model_type}')
        pred_imgs = vp_imgs(model, gt_inputs, pred_len, model_type)
        recon_dict = eval_imgs(pred_imgs, gt_imgs)
        save_results(data_idx, pred_imgs, gt_imgs, recon_dict, save_dir)
    recon_dict = {k: torch.tensor(v) for k, v in recon_dict.items()}
    return recon_dict


def save_results(data_idx, pred_imgs, gt_imgs, metrics_dict, save_dir):
    """Save metrics and videos to individual frames."""
    # metrics
    metric_fn = os.path.join(save_dir, f'metrics/{data_idx[0]}_metric.pkl')
    os.makedirs(os.path.dirname(metric_fn), exist_ok=True)
    dump_obj(metrics_dict, metric_fn)
    # videos to frames
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, T, 3, H, W]
    pred_imgs = np.round(pred_imgs * 255.).astype(np.uint8)
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()  # [B, T, 3, H, W]
    gt_imgs = np.round(gt_imgs * 255.).astype(np.uint8)
    for i, (pred_vid, gt_vid) in enumerate(zip(pred_imgs, gt_imgs)):
        idx = data_idx[i]
        # [T, 3, H, W] -> [T, H, W, 3]
        pred_vid = pred_vid.transpose(0, 2, 3, 1)
        gt_vid = gt_vid.transpose(0, 2, 3, 1)
        pred_dir = os.path.join(save_dir, 'pred_vids', str(idx))
        gt_dir = os.path.join(save_dir, 'gt_vids', str(idx))
        os.makedirs(pred_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)
        for t, (pred, gt) in enumerate(zip(pred_vid, gt_vid)):
            Image.fromarray(pred).save(os.path.join(pred_dir, f'{t}.png'))
            Image.fromarray(gt).save(os.path.join(gt_dir, f'{t}.png'))


def save_vp_imgs(model, data_dict, model_type, save_dir):
    """Save the recon video instead of computing metrics."""
    gt_imgs = data_dict[
        'img'][:, model.history_len:]  # take only frames need to predict
    pred_len = model.rollout_len
    # fake a metric_dict for compatibility
    recon_dict = {'mse': torch.tensor(0.).type_as(gt_imgs)}
    pred_imgs = vp_imgs(model, data_dict['slots'][:, :model.history_len],
                        pred_len)  # taking gt slots for future rolling out.
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, T, 3, H, W]
    gt_imgs = to_rgb_from_tensor(gt_imgs).cpu().numpy()  # [B, T, 3, H, W]
    data_idx = data_dict['data_idx']  # [B]
    for i, (pred, gt) in enumerate(zip(pred_imgs, gt_imgs)):
        idx = data_idx[i].cpu().item()
        save_video(pred, os.path.join(save_dir, f'{idx}_pred.mp4'), fps=8)
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

    if args.save_video:
        save_dir = os.path.join(os.path.dirname(args.weight), 'vis')
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        method._test_step = lambda model, batch_data: \
            save_vp_imgs(model, batch_data, model_type=params.model,
                         save_dir=save_dir)
    else:
        save_dir = os.path.join(os.path.dirname(args.weight), 'eval')
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        method._test_step = lambda model, batch_data: \
            test_vp_imgs(model, batch_data, model_type=params.model,
                         save_dir=save_dir)
    method.test()
    print(f'{os.path.basename(args.params)} testing done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test video prediction')
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

    params.n_sample_frames = params.video_len // params.frame_offset
    if params.model == 'LDMSlotFormer':
        params.loss_dict['rollout_len'] = \
            params.n_sample_frames - params.rollout_dict['history_len']
        params.loss_dict['use_img_recon_loss'] = True
        params.dataset = 'physion_slots_test'
        params.slots_root = params.slots_root.replace('physion_training_slots',
                                                      'physion_test_slots')
    else:
        raise NotImplementedError(f'Unknown model {params.model}')
    params.load_img = True

    # adjust bs & GPU settings
    params.val_batch_size = args.bs  # DDP
    params.gpus = torch.cuda.device_count()
    params.ddp = (params.gpus > 1)

    torch.backends.cudnn.benchmark = True
    main(params)
