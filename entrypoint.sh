#!/usr/bin/env python3
# --------------------------------------------------------- no_background_sam.py
# Aísla el objeto principal y blanquea el fondo.
# Entrada y salida mantienen la resolución original.
# ------------------------------------------------------------------------------

import argparse
from pathlib import Path
import time
import logging

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------------------ argumentos
parser = argparse.ArgumentParser(description="Quita fondo con SAM")
parser.add_argument("--input",      required=True, help="Carpeta con imágenes")
parser.add_argument("--output",     required=True, help="Carpeta de salida")
parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth de SAM")
parser.add_argument("--max-side",   type=int, default=0,
                    help="Máx. lado que verá SAM (0 = sin downscale interno)")
parser.add_argument("--remove-wall", choices=["heuristic", "depth"], default="heuristic",
                    help="Modo de eliminación de pared")
args = parser.parse_args()

INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- modelo SAM
device     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"
sam        = sam_model_registry[MODEL_TYPE](checkpoint=args.checkpoint).to(device)
mask_gen   = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=4096,
)

# ---------------------------------------------------------------- MiDaS (si es necesario)
midas = None
midas_transform = None
if args.remove_wall == "depth":
    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
    midas_transform = Compose([
        Resize(192),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5]),
    ])

# -------------------------------------------------------------- utilidades
def resize_half(img: np.ndarray):
    return cv2.resize(img, dsize=None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

def resize_for_sam(img: np.ndarray, max_side: int):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img_small, scale
    return img, 1.0

def best_mask(masks: list[dict], h: int) -> np.ndarray:
    band = slice(int(0.3 * h), int(0.9 * h))
    def score(m): return m["segmentation"][band, :].sum()
    m_best = max(masks, key=score)
    return (m_best if score(m_best) else max(masks, key=lambda m: m["area"]))["segmentation"]

def estimate_depth(img: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        img_half = resize_half(img)
        img_rgb = cv2.cvtColor(img_half, cv2.COLOR_BGR2RGB)
        input_tensor = midas_transform(img_rgb).unsqueeze(0).to(device)
        depth = midas(input_tensor).squeeze().cpu().numpy()
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        return depth

def mask_from_ransac(depth: np.ndarray, max_error: float = 0.03) -> np.ndarray:
    h, w = depth.shape
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    X = np.stack([xs.ravel(), ys.ravel()], axis=1)
    Z = depth.ravel()
    valid = ~np.isnan(Z)
    X = X[valid]
    Z = Z[valid]

    model = make_pipeline(PolynomialFeatures(degree=1), RANSACRegressor(residual_threshold=max_error))
    model.fit(X, Z)
    Z_pred = model.predict(X)
    residuals = np.abs(Z - Z_pred)
    inliers = residuals < max_error

    mask = np.zeros(h * w, dtype=bool)
    mask[np.flatnonzero(valid)[inliers]] = True
    return mask.reshape(h, w)

# ---------------------------------------------------------------- pipeline
for img_path in sorted(INPUT_DIR.glob("*.[jp][pn]g")):
    start_time = time.time()
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"No se pudo leer {img_path.name}")
        continue

    img_small, scale = resize_for_sam(img, args.max_side)
    masks = mask_gen.generate(img_small)
    if not masks:
        logging.warning(f"Sin máscara para {img_path.name}")
        continue

    mask_small = best_mask(masks, img_small.shape[0])
    mask = (cv2.resize(mask_small.astype(np.uint8),
                       dsize=(img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
            if scale != 1.0 else mask_small)
    mask = ~mask

    if args.remove_wall == "depth":
        depth = estimate_depth(img)
        wall_mask = mask_from_ransac(depth)
        mask[wall_mask] = False

    if args.remove_wall == "heuristic":
        H, W = mask.shape
        mask_u8 = mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        cleaned = np.zeros_like(mask_u8)
        for i in range(1, n):
            x, y, w_box, h_box, area = stats[i]
            touches_top    = y == 0
            touches_bottom = y + h_box >= H - 1
            touches_left   = x == 0
            touches_right  = x + w_box >= W - 1
            is_horiz_band = (
                (touches_bottom or touches_top) and h_box < 0.20 * H and w_box > 0.50 * W
            )
            is_upper_wall = touches_top and h_box > 0.25 * H
            is_full_side_wall = touches_left and touches_right and w_box > 0.80 * W
            if not (is_horiz_band or is_upper_wall or is_full_side_wall):
                cleaned[labels == i] = 1
        if cleaned.sum() == 0:
            cleaned = mask_u8
        mask = cleaned.astype(bool)

    result = img.copy()
    result[~mask] = 255

    cv2.imwrite(str(OUTPUT_DIR / img_path.name), result)
    elapsed = time.time() - start_time
    logging.info(f"{img_path.name} procesada en {elapsed:.2f}s")
