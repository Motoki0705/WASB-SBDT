import os
import os.path as osp
import time
import logging
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler

from utils import save_checkpoint, AverageMeter, save_heatmaps

log = logging.getLogger(__name__)

def train_epoch(epoch, model, train_loader, loss_criterion, optimizer, device, use_fp16=False, scaler=None, grad_accum_steps: int = 1):
    batch_loss = AverageMeter()
    model.train()
    t_start = time.time()

    if grad_accum_steps < 1:
        raise ValueError(f'grad_accum_steps must be >= 1, got {grad_accum_steps}')

    optimizer.zero_grad()

    num_batches = len(train_loader)

    for batch_idx, (imgs, hms) in enumerate(tqdm(train_loader, desc='[(TRAIN) Epoch {}]'.format(epoch)) ):

        imgs = imgs.to(device, non_blocking=True)
        for scale, hm in hms.items():
            hms[scale] = hm.to(device, non_blocking=True)

        if use_fp16 and scaler is not None:
            with torch.amp.autocast(device_type=device, dtype=torch.float16):
                preds = model(imgs)
                loss_raw  = loss_criterion(preds, hms)
                loss = loss_raw / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            preds = model(imgs)
            loss_raw  = loss_criterion(preds, hms)
            loss = loss_raw / grad_accum_steps
            loss.backward()

        batch_loss.update(loss_raw.item(), preds[0].size(0))

        is_accum_step = ((batch_idx + 1) % grad_accum_steps == 0) or ((batch_idx + 1) == num_batches)

        if is_accum_step:
            if use_fp16 and scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

    t_elapsed = time.time() - t_start

    log.info('(TRAIN) Epoch {epoch} Loss:{batch_loss.avg:.6f} Time:{time:.1f}(sec)'.format(epoch=epoch, batch_loss=batch_loss, time=t_elapsed))
    return {'epoch':epoch, 'loss':batch_loss.avg}

@torch.no_grad()
def test_epoch(epoch, model, dataloader, loss_criterion, device, cfg, vis_dir=None, use_fp16=False, output_dir=None, save_heatmap_batches: int = 2):
    """Run test/validation epoch.

    Args:
        epoch: Current epoch number.
        model: Model to evaluate.
        dataloader: Test dataloader.
        loss_criterion: Loss function.
        device: Device to run on.
        cfg: Configuration dict.
        vis_dir: Visualization directory (legacy, unused).
        use_fp16: Use FP16 mixed precision.
        output_dir: Output directory for saving heatmaps.
        save_heatmap_batches: Number of batches to save heatmaps for at epoch end.
    """
    batch_loss    = AverageMeter()
    model.eval()
    
    t_start = time.time()
    num_batches = len(dataloader)
    
    # Store last N batches for heatmap saving
    last_batches = []
    
    for batch_idx, (imgs, hms, trans, xys_gt, visis_gt, img_paths) in enumerate(tqdm(dataloader, desc='[(TEST) Epoch {}]'.format(epoch))):
        imgs = imgs.to(device, non_blocking=True)
        for scale, hm in hms.items():
            hms[scale] = hm.to(device, non_blocking=True)

        if use_fp16:
            with autocast(dtype=torch.float16):
                preds = model(imgs)
                loss  = loss_criterion(preds, hms)
        else:
            preds = model(imgs)
            loss  = loss_criterion(preds, hms)

        batch_loss.update(loss.item(), preds[0].size(0))
        
        # Store last N batches for heatmap visualization
        if output_dir is not None and save_heatmap_batches > 0:
            if batch_idx >= num_batches - save_heatmap_batches:
                last_batches.append((batch_idx, preds, hms, img_paths))
    
    t_elapsed = time.time() - t_start

    # Save heatmaps from last batches
    if output_dir is not None and len(last_batches) > 0:
        log.info(f'Saving heatmaps for {len(last_batches)} batches to {output_dir}')
        for batch_idx, preds, hms, img_paths in last_batches:
            save_heatmaps(
                preds=preds,
                hms_gt=hms,
                img_paths=img_paths,
                output_dir=output_dir,
                epoch=epoch,
                batch_idx=batch_idx,
                max_samples=4,
            )

    log.info('(TEST) Epoch {epoch} Loss:{batch_loss.avg:.6f} Time:{time:.1f}(sec)'.format(epoch=epoch, batch_loss=batch_loss, time=t_elapsed))
    return {'epoch': epoch, 'loss':batch_loss.avg }


