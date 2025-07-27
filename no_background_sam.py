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
mask_gen   = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=4096
)

# -------------------------------------------------------------- utilidades
def resize_for_sam(img: np.ndarray, max_side: int):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return img_small, scale
    return img, 1.0


def best_mask(masks: list[dict], h: int) -> np.ndarray:
    """Elige la máscara que mejor cubre la banda vertical media"""
    band = slice(int(0.3 * h), int(0.9 * h))
    return max(masks, key=lambda m: m["segmentation"][band, :].sum())["segmentation"]

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

    # Morfología inicial: closing + opening para suavizar
    # Kernel dinámico según resolución (1% de la altura, mínimo 3px)
    k = max(3, int(0.01 * H))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel).astype(bool)

    # Heurística: eliminar bandas planas en bordes
    H, W = mask.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size:
        # umbral dinámico: percentil 5% de áreas
        thresh_area = np.percentile(areas, 5)
    else:
        thresh_area = 0
    clean = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, stats.shape[0]):
        x, y, w_box, h_box, area = stats[i]
        top = (y == 0); bot = (y + h_box >= H)
        left = (x == 0); right = (x + w_box >= W)
        floor = bot and h_box < 0.2 * H and w_box > 0.5 * W
        wallU = top and h_box > 0.25 * H
        wallLR = left and right and w_box > 0.8 * W
        # descartar si área < umbral dinámico o cumple floor/wall
        if not (area < thresh_area or floor or wallU or wallLR):
            clean[labels == i] = 1
    mask = clean.astype(bool) if clean.sum() else mask

    # --- Rellenar pequeños agujeros internos (ruido) ---
    inv = (~mask).astype(np.uint8)
    _, lab_h, st_h, _ = cv2.connectedComponentsWithStats(inv, 8)
    for k in range(1, st_h.shape[0]):
        area = st_h[k, cv2.CC_STAT_AREA]
        if area < thresh_area:
            inv[lab_h == k] = 0
    mask = ~inv.astype(bool)

    # --- Salida ---
    out = img.copy()
    out[~mask] = 255
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), out)
    logging.info(f"{img_path.name} → OK | {time.time()-start:.2f}s")
