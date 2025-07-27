#!/usr/bin/env python3
# --------------------------------------------------------- no_background_sam.py
# Aísla el objeto principal y blanquea el fondo sin plano ni pared.
# Mantiene la resolución original.
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
args = parser.parse_args()

INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- modelo SAM
device     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"
sam        = sam_model_registry[MODEL_TYPE](checkpoint=args.checkpoint).to(device)
# Pass generador rápido para segunda pasada
mask_gen   = SamAutomaticMaskGenerator(model=sam, points_per_side=32,
                                       pred_iou_thresh=0.9,
                                       stability_score_thresh=0.95,
                                       min_mask_region_area=4096)
mask_gen_rot = SamAutomaticMaskGenerator(model=sam, points_per_side=16,
                                         pred_iou_thresh=0.95,
                                         stability_score_thresh=0.95,
                                         min_mask_region_area=4096)

# -------------------------------------------------------------- utilidades
def resize_for_sam(img: np.ndarray, max_side: int):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        return img_small, scale
    return img, 1.0


def best_mask(masks: list[dict], h: int) -> np.ndarray:
    """Elige la máscara que mejor cubre la banda vertical media"""
    band = slice(int(0.3 * h), int(0.9 * h))
    m_best = max(masks, key=lambda m: m["segmentation"][band, :].sum())
    return m_best["segmentation"]

# ---------------------------------------------------------------- pipeline
for img_path in tqdm(sorted(INPUT_DIR.glob("*.[jp][pn]g")), desc="Procesando"):
    start = time.time()
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"No se pudo leer {img_path.name}")
        continue

    # --- PASADA 1: SAM + heurística de componentes ---
    img_s, scale = resize_for_sam(img, args.max_side)
    masks = mask_gen.generate(img_s)
    if not masks:
        logging.warning(f"Sin máscara para {img_path.name}")
        continue
    m_s = best_mask(masks, img_s.shape[0])
    mask = (cv2.resize(m_s.astype(np.uint8), (img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
            if scale != 1.0 else m_s)
    mask = ~mask

    # Heurística: eliminar bandas planas en bordes
    H, W = mask.shape
    stats_mask = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    _, labels, stats, _ = stats_mask
    clean = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, stats.shape[0]):
        x, y, w_box, h_box, _ = stats[i]
        top = (y == 0); bot = (y + h_box >= H)
        left = (x == 0); right = (x + w_box >= W)
        floor = bot and h_box < 0.2 * H and w_box > 0.5 * W
        wallU = top and h_box > 0.25 * H
        wallLR = left and right and w_box > 0.8 * W
        if not (floor or wallU or wallLR):
            clean[labels == i] = 1
    mask = clean.astype(bool) if clean.sum() else mask

    # --- PASADA 2: Flip trick para pared como suelo ---
    rot = cv2.rotate(img, cv2.ROTATE_180)
    rot_s, sc2 = resize_for_sam(rot, args.max_side)
    masks_r = mask_gen_rot.generate(rot_s)
    if masks_r:
        rb = best_mask(masks_r, rot_s.shape[0])
        mask_r = (cv2.resize(rb.astype(np.uint8), (rot.shape[1], rot.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
                  if sc2 != 1.0 else rb)
        mask_r = ~mask_r
        # heurística suelo en rotada
        Hf, Wf = mask_r.shape
        stats_r = cv2.connectedComponentsWithStats(mask_r.astype(np.uint8), 8)
        _, labf, stf, _ = stats_r
        cf = np.zeros_like(mask_r, dtype=np.uint8)
        for j in range(1, stf.shape[0]):
            x0, y0, w0, h0, _ = stf[j]
            floor_r = (y0 + h0 >= Hf) and (h0 < 0.2 * Hf) and (w0 > 0.5 * Wf)
            if not floor_r:
                cf[labf == j] = 1
        back = cv2.rotate(cf, cv2.ROTATE_180).astype(bool)
        mask &= back

    # --- Rellenar pequeños agujeros internos ---
    inv = (~mask).astype(np.uint8)
    stats_hole = cv2.connectedComponentsWithStats(inv, 8)
    _, lab_h, st_h, _ = stats_hole
    for k in range(1, st_h.shape[0]):
        area = st_h[k, cv2.CC_STAT_AREA]
        if area < 0.005 * H * W:
            inv[lab_h == k] = 0
    mask = ~inv.astype(bool)

    # --- Salida ---
    out = img.copy()
    out[~mask] = 255
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), out)
    logging.info(f"{img_path.name} → OK | {time.time()-start:.2f}s")