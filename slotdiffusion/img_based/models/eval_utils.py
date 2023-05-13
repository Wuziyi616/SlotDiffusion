import numpy as np
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

import torch
import torch.nn.functional as F
import torchvision.ops as vops

FG_THRE = 0.5

###########################################
# Utils
###########################################


def postproc_mask(batch_masks):
    """Post-process masks instead of directly taking argmax.

    Args:
        batch_masks: [B, T, N, H, W]

    Returns:
        masks: [B, T, H, W]
    """
    batch_masks = batch_masks.clone()
    B, T, N, H, W = batch_masks.shape
    batch_masks = batch_masks.reshape(B * T, N, H * W)
    slots_max = batch_masks.max(-1)[0]  # [B*T, N]
    bg_idx = slots_max.argmin(-1)  # [B*T]
    spatial_max = batch_masks.max(1)[0]  # [B*T, H*W]
    bg_mask = (spatial_max < FG_THRE)  # [B*T, H*W]
    idx_mask = torch.zeros((B * T, N)).type_as(bg_mask)
    idx_mask[torch.arange(B * T), bg_idx] = True
    # set the background mask score to 1
    batch_masks[idx_mask.unsqueeze(-1) * bg_mask.unsqueeze(1)] = 1.
    masks = batch_masks.argmax(1)  # [B*T, H*W]
    return masks.reshape(B, T, H, W)


def masks_to_boxes_w_empty_mask(binary_masks):
    """binary_masks: [B, H, W]."""
    B = binary_masks.shape[0]
    obj_mask = (binary_masks.sum([-1, -2]) > 0)  # [B]
    bboxes = torch.ones((B, 4)).float().to(binary_masks.device) * -1.
    bboxes[obj_mask] = vops.masks_to_boxes(binary_masks[obj_mask])
    return bboxes


def masks_to_boxes(masks, num_boxes=7):
    """Convert seg_masks to bboxes.

    Args:
        masks: [B, T, H, W], output after taking argmax
        num_boxes: number of boxes to generate (num_slots)

    Returns:
        bboxes: [B, T, N, 4], 4: [x1, y1, x2, y2]
    """
    B, T, H, W = masks.shape
    binary_masks = F.one_hot(masks, num_classes=num_boxes)  # [B, T, H, W, N]
    binary_masks = binary_masks.permute(0, 1, 4, 2, 3)  # [B, T, N, H, W]
    binary_masks = binary_masks.contiguous().flatten(0, 2)
    bboxes = masks_to_boxes_w_empty_mask(binary_masks)  # [B*T*N, 4]
    bboxes = bboxes.reshape(B, T, num_boxes, 4)  # [B, T, N, 4]
    return bboxes


###########################################
# Image quality-related metrics
###########################################


def mse_metric(x, y):
    """x/y: [B, 3, H, W], np.ndarray in [0, 1]"""
    # people often sum over channel + spatial dimension in recon MSE
    return ((x - y)**2).sum(-1).sum(-1).sum(-1).mean()


def psnr_metric(x, y):
    """x/y: [B, 3, H, W], np.ndarray in [0, 1]"""
    psnrs = [
        peak_signal_noise_ratio(
            x[i],
            y[i],
            data_range=1.,
        ) for i in range(x.shape[0])
    ]
    return np.mean(psnrs)


def ssim_metric(x, y):
    """x/y: [B, 3, H, W], np.ndarray in [0, 1]"""
    x = x * 255.
    y = y * 255.
    ssims = [
        structural_similarity(
            x[i],
            y[i],
            channel_axis=0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
            data_range=255,
        ) for i in range(x.shape[0])
    ]
    return np.mean(ssims)


def perceptual_dist(x, y, loss_fn):
    """x/y: [B, 3, H, W], torch.Tensor in [-1, 1]"""
    return loss_fn(x, y).mean().item()


###########################################
# Segmentation-related metrics
###########################################


def preproc_masks_overlap(gt_mask, pred_mask, inst_overlap_mask=None):
    """Pre-process masks to handle overlapping instances.

    From [DINOSAUR](https://arxiv.org/pdf/2209.14860.pdf), set the overlapping
        pixels to background, for both pred and gt masks. Only on COCO dataset.
    """
    if inst_overlap_mask is None:
        return gt_mask, pred_mask
    # all masks are of shape [H*W] or [H, W], dtype int
    assert torch.unique(inst_overlap_mask).shape[0] <= 2  # at most [0, 1]
    # set gt overlapping pixels to background
    gt_mask = gt_mask.clone()
    gt_mask[inst_overlap_mask == 1] = 0
    # set pred overlapping pixels to another class
    pred_mask = pred_mask.clone()
    pred_mask[inst_overlap_mask == 1] = pred_mask.max() + 1
    return gt_mask, pred_mask


def adjusted_rand_index(true_ids, pred_ids, ignore_background=False):
    """Computes the adjusted Rand index (ARI), a clustering similarity score.

    Code borrowed from https://github.com/google-research/slot-attention-video/blob/e8ab54620d0f1934b332ddc09f1dba7bc07ff601/savi/lib/metrics.py#L111

    Args:
        true_ids: An integer-valued array of shape
            [batch_size, seq_len, H, W]. The true cluster assignment encoded
            as integer ids.
        pred_ids: An integer-valued array of shape
            [batch_size, seq_len, H, W]. The predicted cluster assignment
            encoded as integer ids.
        ignore_background: Boolean, if True, then ignore all pixels where
            true_ids == 0 (default: False).

    Returns:
        ARI scores as a float32 array of shape [batch_size].
    """
    if len(true_ids.shape) == 3:
        true_ids = true_ids.unsqueeze(1)
    if len(pred_ids.shape) == 3:
        pred_ids = pred_ids.unsqueeze(1)

    true_oh = F.one_hot(true_ids).float()
    pred_oh = F.one_hot(pred_ids).float()

    if ignore_background:
        true_oh = true_oh[..., 1:]  # Remove the background row.

    N = torch.einsum("bthwc,bthwk->bck", true_oh, pred_oh)
    A = torch.sum(N, dim=-1)  # row-sum  (batch_size, c)
    B = torch.sum(N, dim=-2)  # col-sum  (batch_size, k)
    num_points = torch.sum(A, dim=1)

    rindex = torch.sum(N * (N - 1), dim=[1, 2])
    aindex = torch.sum(A * (A - 1), dim=1)
    bindex = torch.sum(B * (B - 1), dim=1)
    expected_rindex = aindex * bindex / torch.clamp(
        num_points * (num_points - 1), min=1)
    max_rindex = (aindex + bindex) / 2
    denominator = max_rindex - expected_rindex
    ari = (rindex - expected_rindex) / denominator

    # There are two cases for which the denominator can be zero:
    # 1. If both label_pred and label_true assign all pixels to a single cluster.
    #    (max_rindex == expected_rindex == rindex == num_points * (num_points-1))
    # 2. If both label_pred and label_true assign max 1 point to each cluster.
    #    (max_rindex == expected_rindex == rindex == 0)
    # In both cases, we want the ARI score to be 1.0:
    return torch.where(denominator != 0, ari, torch.tensor(1.).type_as(ari))


def ARI_metric(x, y, inst_overlap_mask=None):
    """x/y: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(x.dtype)
    assert 'int' in str(y.dtype)
    if inst_overlap_mask is not None:
        x, y = x.clone(), y.clone()
        for i in range(x.shape[0]):
            x[i], y[i] = preproc_masks_overlap(x[i], y[i],
                                               inst_overlap_mask[i])
    return adjusted_rand_index(x, y, ignore_background=False).mean().item()


def fARI_metric(x, y, inst_overlap_mask=None):
    """x/y: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(x.dtype)
    assert 'int' in str(y.dtype)
    if inst_overlap_mask is not None:
        x, y = x.clone(), y.clone()
        for i in range(x.shape[0]):
            x[i], y[i] = preproc_masks_overlap(x[i], y[i],
                                               inst_overlap_mask[i])
    return adjusted_rand_index(x, y, ignore_background=True).mean().item()


def bbox_precision_recall(gt_pres_mask, gt_bbox, pred_bbox, ovthresh=0.5):
    """Compute the precision of predicted bounding boxes.

    Args:
        gt_pres_mask: A boolean tensor of shape [N]
        gt_bbox: A tensor of shape [N, 4]
        pred_bbox: A tensor of shape [M, 4]
    """
    gt_bbox, pred_bbox = gt_bbox.clone(), pred_bbox.clone()
    gt_bbox = gt_bbox[gt_pres_mask.bool()]
    pred_bbox = pred_bbox[pred_bbox[:, 0] >= 0.]
    N, M = gt_bbox.shape[0], pred_bbox.shape[0]
    assert gt_bbox.shape[1] == pred_bbox.shape[1] == 4
    # assert M >= N
    tp, fp = 0, 0
    bbox_used = [False] * pred_bbox.shape[0]
    bbox_ious = vops.box_iou(gt_bbox, pred_bbox)  # [N, M]

    # Find the best iou match for each ground truth bbox.
    for i in range(N):
        best_iou_idx = bbox_ious[i].argmax().item()
        best_iou = bbox_ious[i, best_iou_idx].item()
        if best_iou >= ovthresh and not bbox_used[best_iou_idx]:
            tp += 1
            bbox_used[best_iou_idx] = True
        else:
            fp += 1

    # compute precision and recall
    precision = tp / float(M)
    recall = tp / float(N)
    return precision, recall


def batch_bbox_precision_recall(gt_pres_mask, gt_bbox, pred_bbox):
    """Compute the precision of predicted bounding boxes over batch."""
    aps, ars = [], []
    for i in range(gt_pres_mask.shape[0]):
        ap, ar = bbox_precision_recall(gt_pres_mask[i], gt_bbox[i],
                                       pred_bbox[i])
        aps.append(ap)
        ars.append(ar)
    return np.mean(aps), np.mean(ars)


def hungarian_miou(gt_mask, pred_mask, ignore_background=True):
    """both mask: [H*W] after argmax, 0 is gt background index."""
    # in case GT only contains bg class
    if gt_mask.max().item() == 0 and ignore_background:
        return np.nan
    true_oh = F.one_hot(gt_mask).float()  # [HW, N]
    if ignore_background:
        true_oh = true_oh[..., 1:]  # only foreground
    pred_oh = F.one_hot(pred_mask).float()  # [HW, M]
    N, M = true_oh.shape[-1], pred_oh.shape[-1]
    # compute all pairwise IoU
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)  # [N, M]
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect  # same
    iou = intersect / (union + 1e-8)  # [N, M]
    iou = iou.detach().cpu().numpy()
    # find the best match for each gt
    row_ind, col_ind = linear_sum_assignment(iou, maximize=True)
    # there are two possibilities here
    #   1. M >= N, just take the best match mean
    #   2. M < N, some objects are not detected, their iou is 0
    if M >= N:
        assert (row_ind == np.arange(N)).all()
        return iou[row_ind, col_ind].mean()
    return iou[row_ind, col_ind].sum() / float(N)


def mean_best_overlap(gt_mask, pred_mask):
    """both mask: [H*W] after argmax, 0 is gt background index.

    From [DINOSAUR](https://arxiv.org/pdf/2209.14860.pdf), ignore background.
    """
    # in case GT only contains bg class
    if gt_mask.max().item() == 0:
        return np.nan
    true_oh = F.one_hot(gt_mask).float()  # [HW, N]
    # quote from email exchange with the authors:
    # > the ground truth mask corresponding to all background pixels is not
    # > used for matching
    true_oh = true_oh[..., 1:]  # only foreground
    pred_oh = F.one_hot(pred_mask).float()  # [HW, M]
    # compute all pairwise IoU
    intersect = (true_oh[:, :, None] * pred_oh[:, None, :]).sum(0)  # [N, M]
    union = true_oh.sum(0)[:, None] + pred_oh.sum(0)[None] - intersect  # same
    iou = intersect / (union + 1e-8)  # [N, M]
    iou = iou.detach().cpu().numpy()
    # find the best pred_mask for each gt_mask
    # each pred_mask can be used multiple times
    best_iou = iou.max(1)
    return best_iou.mean()


def miou_metric(gt_mask, pred_mask, inst_overlap_mask=None):
    """both mask: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(gt_mask.dtype)
    assert 'int' in str(pred_mask.dtype)
    if inst_overlap_mask is None:
        inst_overlap_mask = [None] * gt_mask.shape[0]
    else:
        inst_overlap_mask = inst_overlap_mask.flatten(1, 2)
    gt_mask, pred_mask = gt_mask.flatten(1, 2), pred_mask.flatten(1, 2)
    ious = []
    for i in range(gt_mask.shape[0]):
        gt_mask_, pred_mask_ = preproc_masks_overlap(gt_mask[i], pred_mask[i],
                                                     inst_overlap_mask[i])
        iou = hungarian_miou(gt_mask_, pred_mask_, ignore_background=False)
        ious.append(iou)
    if all(np.isnan(iou) for iou in ious):
        print('WARNING: all miou in a batch are nan')
        return np.nan
    return np.nanmean(ious)


def fmiou_metric(gt_mask, pred_mask, inst_overlap_mask=None):
    """both mask: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(gt_mask.dtype)
    assert 'int' in str(pred_mask.dtype)
    if inst_overlap_mask is None:
        inst_overlap_mask = [None] * gt_mask.shape[0]
    else:
        inst_overlap_mask = inst_overlap_mask.flatten(1, 2)
    gt_mask, pred_mask = gt_mask.flatten(1, 2), pred_mask.flatten(1, 2)
    ious = []
    for i in range(gt_mask.shape[0]):
        gt_mask_, pred_mask_ = preproc_masks_overlap(gt_mask[i], pred_mask[i],
                                                     inst_overlap_mask[i])
        iou = hungarian_miou(gt_mask_, pred_mask_, ignore_background=True)
        ious.append(iou)
    if all(np.isnan(iou) for iou in ious):
        print('WARNING: all miou in a batch are nan')
        return np.nan
    return np.nanmean(ious)


def mbo_metric(gt_mask, pred_mask, inst_overlap_mask=None):
    """both mask: [B, H, W], both are seg_masks after argmax."""
    assert 'int' in str(gt_mask.dtype)
    assert 'int' in str(pred_mask.dtype)
    if inst_overlap_mask is None:
        inst_overlap_mask = [None] * gt_mask.shape[0]
    else:
        inst_overlap_mask = inst_overlap_mask.flatten(1, 2)
    gt_mask, pred_mask = gt_mask.flatten(1, 2), pred_mask.flatten(1, 2)
    mbos = []
    for i in range(gt_mask.shape[0]):
        gt_mask_, pred_mask_ = preproc_masks_overlap(gt_mask[i], pred_mask[i],
                                                     inst_overlap_mask[i])
        mbo = mean_best_overlap(gt_mask_, pred_mask_)
        mbos.append(mbo)
    if all(np.isnan(mbo) for mbo in mbos):
        print('WARNING: all mbo in a batch are nan')
        return np.nan
    return np.nanmean(mbos)
