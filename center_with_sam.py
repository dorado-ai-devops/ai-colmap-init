#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


parser.add_argument("--input",      required=True, help="Input folder with images")
parser.add_argument("--output",     required=True, help="Output folder")
parser.add_argument("--checkpoint", required=True, help="SAM .pth checkpoint")
parser.add_argument("--size", type=int, default=768, help="Final square size")
parser.add_argument("--pad",  type=int, default=40,  help="Extra pixels around bbox")
args = parser.parse_args()

INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"

sam   = sam_model_registry[MODEL_TYPE](checkpoint=args.checkpoint).to(device)
mask_generator = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=4096,   # ~64×64 – ignore specks of dust
)

CROP_SIZE = args.size
PAD       = args.pad



def best_mask_for_image(masks: list[dict], h: int) -> np.ndarray:
    """Return the mask that best covers the lower 30‑90 % vertical band."""
    lower_band = slice(int(0.30 * h), int(0.90 * h))

    def band_score(m):
        return m["segmentation"][lower_band, :].sum()

    candidate = max(masks, key=band_score)
    if band_score(candidate) == 0:      # no mask intersects band
        candidate = max(masks, key=lambda m: m["area"])
    return candidate["segmentation"]


def square_letterbox(img: np.ndarray, size: int) -> np.ndarray:
    """Resize keeping aspect, pad with white to exact square `size×size`."""
    h, w = img.shape[:2]
    scale = size / max(h, w)
    if scale != 1:
        img = cv2.resize(img, dsize=None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    top    = (size - h) // 2
    bottom = size - h - top
    left   = (size - w) // 2
    right  = size - w - left

    return cv2.copyMakeBorder(
        img, top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],   # white
    )



for img_path in tqdm(sorted(INPUT_DIR.glob("*.[jp][pn]g"))):
    img = cv2.imread(str(img_path))
    if img is None:
        continue

    masks = mask_generator.generate(img)
    if not masks:
        print(f"[WARN] No mask found for {img_path.name}")
        continue

    H, W = img.shape[:2]
    mask = best_mask_for_image(masks, H)

   
    masked = img.copy()
    masked[~mask] = 255


    ys, xs = np.where(mask)
    x1 = max(0, xs.min() - PAD)
    x2 = min(W, xs.max() + PAD)
    y1 = max(0, ys.min() - PAD)
    y2 = min(H, ys.max() + PAD)

    crop = masked[y1:y2, x1:x2]
    if crop.size == 0:
        print(f"[WARN] Empty crop for {img_path.name}")
        continue

    final = square_letterbox(crop, CROP_SIZE)
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), final)
    print(f"[OK] {img_path.name} → saved")
