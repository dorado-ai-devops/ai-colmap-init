#!/usr/bin/env python3
# no_back_again.py
# Remueve fondo con SAM y elimina automáticamente la banda del rodapié

import argparse
from pathlib import Path
import time
import logging

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# ------------------------------------------------------------------ argumentos
def parse_args():
    parser = argparse.ArgumentParser(description="Quita fondo y rodapié automáticamente con SAM")
    parser.add_argument("--input",    required=True, help="Carpeta con imágenes de entrada")
    parser.add_argument("--output",   required=True, help="Carpeta de salida")
    parser.add_argument("--checkpoint", required=True, help="Ruta al checkpoint .pth de SAM")
    parser.add_argument("--max-side", type=int, default=0,
                        help="Máximo lado interno para SAM (0 = sin downscale)")
    return parser.parse_args()

# ------------------------------------------------------------------ inicialización SAM
args = parse_args()
INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=args.checkpoint).to(device)
mask_gen = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=32,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=4096,
)

# ---------------------------------------------------------------- utilidades

def resize_for_sam(img, max_side):
    if max_side <= 0:
        return img, 1.0
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1.0:
        return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), scale
    return img, 1.0

# Identifica y elimina componentes del rodapié
def remove_skirt(mask: np.ndarray, min_height_ratio=0.02, max_height_ratio=0.2, width_coverage=0.8) -> np.ndarray:
    H, W = mask.shape
    labels, stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)[:2]
    cleaned = mask.copy()
    for i in range(1, labels.max()+1):
        x, y, w_box, h_box, area = stats[i]
        # Componentes que tocan el borde inferior y cubren casi todo el ancho y poca altura
        if y + h_box >= H - 1 and w_box >= width_coverage * W and h_box <= max_height_ratio * H:
            cleaned[labels == i] = False
    return cleaned

# ---------------------------------------------------------------- pipeline
t0 = time.time()
for img_path in tqdm(sorted(INPUT_DIR.glob("*.[jp][pn]g")), desc="Procesando"):
    t_start = time.time()
    img = cv2.imread(str(img_path))
    if img is None:
        logging.warning(f"No se pudo leer {img_path.name}")
        continue

    # 1) Generar máscaras automáticas
    img_s, scale = resize_for_sam(img, args.max_side)
    masks = mask_gen.generate(img_s)
    if not masks:
        logging.warning(f"Sin máscara para {img_path.name}")
        continue
    # 2) Seleccionar la máscara que mejor cubre la banda media
    h_small = img_s.shape[0]
    band = slice(int(0.3*h_small), int(0.9*h_small))
    best = max(masks, key=lambda m: m["segmentation"][band, :].sum())["segmentation"]

    # 3) Reescalar y computar máscara final de primer plano
    if scale != 1.0:
        mask = cv2.resize(best.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask = best.astype(bool)

    # 4) Invertir para obtener fondo y limpiar rodapié
    bg = ~mask
    bg = remove_skirt(bg)
    fg = ~bg

    # 5) Refinar bordes con morfología
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)

    # 6) Aplicar máscara y exportar
    out = img.copy()
    out[~fg] = 255
    cv2.imwrite(str(OUTPUT_DIR / img_path.name), out)
    dt = time.time() - t_start
    logging.info(f"{img_path.name} → OK | {dt:.2f}s")
logging.info(f"Procesado {len(list(INPUT_DIR.glob('*.[jp][pn]g')))} imágenes en {time.time()-t0:.2f}s")
