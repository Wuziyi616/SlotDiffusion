import os
import sys
import importlib
import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch

from nerv.training import BaseDataModule
from nerv.utils import load_obj, dump_obj, convert4save

from models import build_model, to_rgb_from_tensor
from datasets import build_dataset
from method import build_method
from vis import save_video


@torch.no_grad()
def gen_imgs(model, gt_imgs, model_type):
    """Encode imgs to slots and decode them back to imgs."""
    data_dict = {'img': gt_imgs}
    slots = model(data_dict)['slots']  # [B, T, N, C]
    # combine slots from different images to generate new images
    # switching the slots, we will shift the slots by 1 data item
    # i.e. data0 will have slot[0, :, 0], slot[1, :, 1], slot[2, :, 2], ...
    #      data1 will have slot[1, :, 0], slot[2, :, 1], slot[3, :, 2], ...
    #      data2 will have slot[2, :, 0], slot[3, :, 1], slot[4, :, 2], ...
    B, T, N, _ = slots.shape
    for i in range(N):
        slots[:, :, i] = torch.cat([slots[i:, :, i], slots[:i, :, i]])
    if model_type == 'SAVi':
        slots = slots.flatten(0, 1)
        pred_imgs, _, _, _ = model.decode(slots)
        pred_imgs = pred_imgs.unflatten(0, (B, T))
    elif model_type == 'STEVE':
        data_dict = {'slots': slots}
        pred_imgs = model(data_dict, recon_img=True, bs=24, verbose=False)
    else:
        ddpm_dict = {
            'img': gt_imgs.flatten(0, 1),
            'slots': slots.flatten(0, 1),
        }
        pred_imgs = model.dm_decoder.log_images(
            ddpm_dict,
            use_ddim=False,
            use_dpm=True,  # we use DPM-Solver by default
            same_noise=True,
            ret_intermed=False,
            verbose=False,
        )['samples']
        pred_imgs = pred_imgs.unflatten(0, (B, T))
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
        # save all generated videos to compute FVD later
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
    # not saving `gt_imgs` as we assume they have been saved in `test_recon.py`
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, T, 3, H, W]
    pred_imgs = np.round(pred_imgs * 255.).astype(np.uint8)
    for i, pred_vid in enumerate(pred_imgs):
        idx = data_idx[i]
        # [T, 3, H, W] -> [T, H, W, 3]
        pred_vid = pred_vid.transpose(0, 2, 3, 1)
        pred_dir = os.path.join(save_dir, 'comp_vids', str(idx))
        os.makedirs(pred_dir, exist_ok=True)
        for t, pred in enumerate(pred_vid):
            plt.imsave(os.path.join(pred_dir, f'{t}.png'), pred)


@torch.no_grad()
def save_gen_imgs(model, data_dict, model_type, save_dir):
    """Save the recon video instead of computing metrics."""
    gt_imgs = data_dict['img']
    # fake a metric_dict for compatibility
    recon_dict = {'mse': torch.tensor(0.).type_as(gt_imgs)}
    pred_imgs = gen_imgs(model, gt_imgs, model_type)
    pred_imgs = to_rgb_from_tensor(pred_imgs).cpu().numpy()  # [B, T, 3, H, W]
    data_idx = data_dict['data_idx']  # [B]
    for i, pred in enumerate(pred_imgs):
        idx = data_idx[i].cpu().item()
        pred = convert4save(pred)
        save_video(pred, os.path.join(save_dir, f'{idx}_comp.mp4'), fps=8)
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

    if args.save_video:  # save to mp4 for visualization
        save_dir = os.path.join(os.path.dirname(args.weight), 'vis')
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        method._test_step = lambda model, batch_data: \
            save_gen_imgs(model, batch_data, model_type=params.model,
                          save_dir=save_dir)
    else:  # save to individual frames for FVD computation
        save_dir = os.path.join(os.path.dirname(args.weight), 'eval')
        if args.local_rank == 0:
            os.makedirs(save_dir, exist_ok=True)
        method._test_step = lambda model, batch_data: \
            test_gen_imgs(model, batch_data, model_type=params.model,
                          save_dir=save_dir)
    method.test()
    print(f'{os.path.basename(args.params)} testing done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test video generation')
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
