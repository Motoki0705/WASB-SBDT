import os
import os.path as osp
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from omegaconf import DictConfig
import hydra
from hydra import initialize, compose

from datasets.tennis import Tennis
from utils import gen_heatmap, mkdir_if_missing, gen_video


def overlay_heatmap_on_image(img: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """Overlay single-channel heatmap (0-1) on BGR image."""
    if heatmap.max() > 0:
        hm = (heatmap * 255.0).clip(0, 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        alpha = 0.5
        overlay = (alpha * hm_color + (1.0 - alpha) * img).astype(np.uint8)
        return overlay
    else:
        return img


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    os.chdir(project_root)

    with initialize(version_base=None, config_path="../configs"):
        cfg: DictConfig = compose(config_name="train")

    tennis = Tennis(cfg)

    if len(tennis.test_clip_gts) == 0:
        raise RuntimeError("No test clips with GT found. Check dataset configuration.")

    # Pick the first available test clip
    (match, clip), clip_gt_dict = next(iter(tennis.test_clip_gts.items()))

    # Prepare output directories
    out_root = project_root / "outputs" / "tennis_gt_vis"
    vis_dir = out_root / f"{match}_{clip}"
    mkdir_if_missing(str(vis_dir))

    # Heatmap parameters (use same sigma as training heatmap generator)
    hm_cfg = cfg["dataloader"]["heatmap"]
    sigma = float(hm_cfg["sigmas"][0])

    # Iterate frames in order
    items = sorted(clip_gt_dict.items(), key=lambda kv: kv[0])
    for idx, (frame_path, center) in enumerate(items):
        if frame_path is None:
            continue
        img = cv2.imread(frame_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        if center.is_visible:
            cxy: Tuple[float, float] = center.xy
            heatmap = gen_heatmap((w, h), cxy, sigma)
        else:
            heatmap = np.zeros((h, w), dtype=np.float32)

        overlay = overlay_heatmap_on_image(img, heatmap)
        out_path = vis_dir / f"{idx:06d}.png"
        cv2.imwrite(str(out_path), overlay)

    video_path = out_root / f"{match}_{clip}.mp4"
    gen_video(str(video_path), str(vis_dir))
    print("Video saved to:", video_path)
if __name__ == "__main__":
    main()
