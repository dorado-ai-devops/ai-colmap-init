#!/usr/bin/env python3
# --------------------------------------------------------- no_background_sam.py
# Aísla el objeto principal y blanquea el fondo.
# Entrada y salida mantienen la resolución original.
# ------------------------------------------------------------------------------

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm

# ------------------------------------------------------------------ argumentos
parser = argparse.ArgumentParser(description="Quita fondo con SAM")
parser.add_argument("--input",      required=True, help="Carpeta con imágenes")
parser.add_argument("--output",     required=True, help="Carpeta de salida")
parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth de SAM")
parser.add_argument("--max-side",   type=int, default=0,
                    help="Máx. lado que vera SAM (0 = sin downscale interno)")
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
    points_per_side=64,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=4096,
)

# -------------------------------------------------------------- utilidades
def resize_for_sam(img: np.ndarray, max_side: int):
    """Devuelve (img_reducida, escala) o (img,1.0) si no se reduce."""
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        img_small = cv2.resize(img, dsize=None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        return img_small, scale
    return img, 1.0

def best_mask(masks: list[dict], h: int) -> np.ndarray:
    """Elige la máscara que cubre mejor la franja vertical 30‑90 %."""
    band = slice(int(0.3 * h), int(0.9 * h))
    def score(m): return m["segmentation"][band, :].sum()
    m_best = max(masks, key=score)
    return (m_best if score(m_best) else max(masks, key=lambda m: m["area"]))["segmentation"]

# ---------------------------------------------------------------- pipeline
for img_path in tqdm(sorted(INPUT_DIR.glob("*.[jp][pn]g"))):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN] No se pudo leer {img_path.name}")
        continue

    img_small, scale = resize_for_sam(img, args.max_side)
    masks = mask_gen.generate(img_small)
    if not masks:
        print(f"[WARN] Sin máscara para {img_path.name}")
        continue

    mask_small = best_mask(masks, img_small.shape[0])
    mask = (cv2.resize(mask_small.astype(np.uint8),
                       dsize=(img.shape[1], img.shape[0]),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
            if scale != 1.0 else mask_small)
    mask = ~mask                       # 1) ahora 1 = objetos

    H, W = mask.shape
    mask_u8 = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    cleaned = np.zeros_like(mask_u8)

    for i in range(1, n):              # etiqueta 0 = fondo original
        x, y, w_box, h_box, area = stats[i]

        touches_top    = y == 0
        touches_bottom = y + h_box >= H - 1
        touches_left   = x == 0
        touches_right  = x + w_box >= W - 1

        # — reglas de descarte —
        is_floor_band = (
            touches_bottom and
            h_box < 0.20 * H and
            w_box > 0.50 * W
        )

        is_upper_wall = (
            touches_top and
            h_box > 0.25 * H            # pared alta
        )

        is_full_side_wall = (
            touches_left and touches_right and
            w_box > 0.80 * W
        )

        if not (is_floor_band or is_upper_wall or is_full_side_wall):
            cleaned[labels == i] = 1    # mantenemos componente

    if cleaned.sum() == 0:              # fallback de seguridad
        cleaned = mask_u8

    mask = cleaned.astype(bool)
    result = img.copy()
    result[~mask] = 255   

    cv2.imwrite(str(OUTPUT_DIR / img_path.name), result)
    print(f"[OK] {img_path.name} → guardado")
