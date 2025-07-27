#!/usr/bin/env python3
# --------------------------------------------------------- no_background_sam.py
# Aísla el objeto principal y blanquea el fondo sin fondo plano ni pared.
# Mantiene la resolución original.
# ------------------------------------------------------------------------------

import argparse
from pathlib import Path
import time
import logging

import cv2
import numpy as np
import torch
from segment_anything import (
    SamAutomaticMaskGenerator,
    sam_model_registry,
    SamPredictor,
)
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

# ------------------------------------------------------------------ argumentos
parser = argparse.ArgumentParser(description="Quita fondo con SAM")
parser.add_argument("--input",      required=True, help="Carpeta con imágenes")
parser.add_argument("--output",     required=True, help="Carpeta de salida")
parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth de SAM")
parser.add_argument(
    "--max-side", type=int, default=0,
    help="Máx. lado que verá SAM (0 = sin downscale interno)"
)
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
predictor = SamPredictor(sam)

# -------------------------------------------------------------- utilidades
def resize_for_sam(img: np.ndarray, max_side: int):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(
            img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
        )
        return img_small, scale
    return img, 1.0


def best_mask(masks: list[dict], h: int) -> np.ndarray:
    """Elige la máscara que mejor cubre la banda vertical media."""
    band = slice(int(0.3 * h), int(0.9 * h))
    m_best = max(
        masks,
        key=lambda m: m["segmentation"][band, :].sum()
    )
    return m_best["segmentation"]


# ---------------------------------------------------------------- pipeline
for img_path in tqdm(sorted(INPUT_DIR.glob("*.[jp][pn]g")), desc="Procesando"):
    start_time = time.time()
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
    mask = (
        cv2.resize(
            m_s.astype(np.uint8),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        if scale != 1.0 else m_s
    )
    mask = ~mask

    # Heurística: eliminar bandas planas en bordes
    H, W = mask.shape
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    clean = np.zeros_like(mask_u8)
    for i in range(1, n):
        x, y, w_box, h_box, area = stats[i]
        top    = (y == 0)
        bot    = (y + h_box >= H)
        left   = (x == 0)
        right  = (x + w_box >= W)
        floor  = bot    and h_box < 0.2 * H and w_box > 0.5 * W
        wallU  = top    and h_box > 0.25 * H
        wallLR = left and right and w_box > 0.8 * W
        if not (floor or wallU or wallLR):
            clean[labels == i] = 1
    mask = clean.astype(bool) if clean.sum() else mask

    # --- PASADA 2: Flip trick para pared como suelo ---
    rot   = cv2.rotate(img, cv2.ROTATE_180)
    rot_s, sc2 = resize_for_sam(rot, args.max_side)
    rm    = mask_gen.generate(rot_s)
    if rm:
        rb  = best_mask(rm, rot_s.shape[0])
        rm2 = (
            cv2.resize(
                rb.astype(np.uint8),
                (rot.shape[1], rot.shape[0]),
                interpolation=cv2.INTER_NEAREST
            ).astype(bool)
            if sc2 != 1.0 else rb
        )
        rm2 = ~rm2
        Hf, Wf = rm2.shape
        lu8, labf, stf, _ = cv2.connectedComponentsWithStats(rm2.astype(np.uint8), 8)
        cf = np.zeros_like(lu8)
        for j in range(1, labf.max()+1):
            x0, y0, w0, h0, _ = stf[j]
            if not (y0 + h0 >= Hf and h0 < 0.2 * Hf and w0 > 0.5 * Wf):
                cf[labf == j] = 1
        back = cv2.rotate(cf.astype(np.uint8), cv2.ROTATE_180).astype(bool)
        mask &= back

    # --- PASADA 3: Refinamiento ROI con SamPredictor ---
    ys, xs = np.where(mask)
    if ys.size:
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        pad = 10
        y0p, y1p = max(0, y0-pad), min(img.shape[0], y1+pad)
        x0p, x1p = max(0, x0-pad), min(img.shape[1], x1+pad)
        crop_img  = img[y0p:y1p, x0p:x1p]
        predictor.set_image(crop_img)
        # prompt: toda la caja
        input_box = np.array([[0, 0, crop_img.shape[1], crop_img.shape[0]]])
        masks_pred, scores, _ = predictor.predict(
            box=input_box,
            multimask_output=False
        )
        refined = masks_pred[0]
        mask[y0p:y1p, x0p:x1p] = refined

    # --- Salida final ---
    out = img.copy()
    out[~mask] = 255
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), out)
    logging.info(
        f"{img_path.name} → OK | {time.time()-start_time:.2f}s"
    )
