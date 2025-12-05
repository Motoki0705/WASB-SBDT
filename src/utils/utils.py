import os
import os.path as osp
import errno
import shutil
import random
import math
import numpy as np
from PIL import Image
import cv2
import torch
from torch import nn

def compute_l2_dist_mat(X, Y, axis=1):
    if X.shape[axis]!=Y.shape[axis]:
        raise RuntimeError('feat dims are different between matrices')
    X2 = np.sum(X**2, axis=1) # shape of (m)
    Y2 = np.sum(Y**2, axis=1) # shape of (n)
    XY = np.matmul(X, Y.T)
    X2 = X2.reshape(-1, 1)
    return np.sqrt(X2 - 2*XY + Y2)

def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

def list2txt(list_data):
    txt = ''
    for cnt, d in enumerate(list_data):
        txt += '{}'.format(d)
        if cnt<len(list_data)-1:
            txt += '-'
    # print(txt)
    return txt

def count_params(model, only_trainable=True):
    if only_trainable:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        num_params = sum(p.numel() for p in model.parameters() )
    return num_params

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def save_checkpoint(state, 
                    is_best, 
                    model_path,
                    best_model_name = 'best_model.pth.tar',
):
    mkdir_if_missing(osp.dirname(model_path))
    torch.save(state, model_path)
    if is_best:
        shutil.copy(model_path, osp.join(osp.dirname(model_path), best_model_name))

def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = ("%s" % seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def _top1(scores):
    batch, seq, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, seq, -1), 1)
    topk_inds = topk_inds % (height * width)
    topk_ys   = (topk_inds / width).int().float()
    topk_xs   = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_ys, topk_xs

def save_heatmaps(
    preds: dict,
    hms_gt: dict,
    img_paths: list,
    output_dir: str,
    epoch: int,
    batch_idx: int,
    max_samples: int = 4,
    colormap: int = cv2.COLORMAP_JET,
):
    """Save predicted heatmaps and ground truth heatmaps as images.

    Output structure:
        output_dir/heatmaps/epoch_{epoch}/
        ├── seq_pred_{seq_idx}/
        │   ├── t_00.png
        │   ├── t_01.png
        │   └── ...
        ├── seq_gt_{seq_idx}/
        │   ├── t_00.png
        │   └── ...
        └── seq_compare_{seq_idx}/
            ├── t_00.png  (GT | Pred side-by-side)
            └── ...

    Args:
        preds: Dict of predicted heatmaps {scale: Tensor[B, T, 1, H, W]}.
        hms_gt: Dict of ground truth heatmaps {scale: Tensor[B, T, 1, H, W]}.
        img_paths: List of input image paths (length B*T or nested).
        output_dir: Output directory path.
        epoch: Current epoch number.
        batch_idx: Batch index within the epoch.
        max_samples: Maximum number of samples to save per batch.
        colormap: OpenCV colormap for heatmap visualization.
    """
    epoch_dir = osp.join(output_dir, "heatmaps", f"epoch_{epoch:03d}")

    for scale, pred in preds.items():
        # pred: [B, T, C, H, W] or [B, T, H, W]
        pred_np = pred.detach().cpu().float()
        if pred_np.dim() == 5:
            pred_np = pred_np[:, :, 0]  # [B, T, H, W]

        gt = hms_gt.get(scale)
        if gt is not None:
            gt_np = gt.detach().cpu().float()
            if gt_np.dim() == 5:
                gt_np = gt_np[:, :, 0]  # [B, T, H, W]
        else:
            gt_np = None

        b, t, h, w = pred_np.shape
        num_samples = min(b, max_samples)

        for bi in range(num_samples):
            # Global sequence index across batches
            seq_idx = batch_idx * b + bi

            # Create directories for this sequence
            pred_dir = osp.join(epoch_dir, f"seq_pred_{seq_idx:04d}")
            gt_dir = osp.join(epoch_dir, f"seq_gt_{seq_idx:04d}")
            compare_dir = osp.join(epoch_dir, f"seq_compare_{seq_idx:04d}")

            mkdir_if_missing(pred_dir)
            if gt_np is not None:
                mkdir_if_missing(gt_dir)
                mkdir_if_missing(compare_dir)

            for ti in range(t):
                # Normalize prediction to [0, 255]
                pred_frame = pred_np[bi, ti].numpy()
                pred_frame = np.clip(pred_frame, 0, 1)
                pred_frame = (pred_frame * 255).astype(np.uint8)
                pred_colored = cv2.applyColorMap(pred_frame, colormap)

                # Save prediction
                cv2.imwrite(osp.join(pred_dir, f"t_{ti:02d}.png"), pred_colored)

                # Save GT if available
                if gt_np is not None:
                    gt_frame = gt_np[bi, ti].numpy()
                    gt_frame = np.clip(gt_frame, 0, 1)
                    gt_frame = (gt_frame * 255).astype(np.uint8)
                    gt_colored = cv2.applyColorMap(gt_frame, colormap)
                    cv2.imwrite(osp.join(gt_dir, f"t_{ti:02d}.png"), gt_colored)

                    # Save side-by-side comparison (GT | Pred)
                    comparison = np.hstack([gt_colored, pred_colored])
                    cv2.imwrite(osp.join(compare_dir, f"t_{ti:02d}.png"), comparison)


class AverageMeter(object):
    """Computes and stores the average and current value.
       
       Code imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

