#!/usr/bin/env python3
"""
------
python mask_only_sam.py \
  --input      path/a/imagenes \
  --output     path/salida \
  --checkpoint path/sam_vit_b.pth \
  --max-side   2048   # 0 = sin reducción
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from tqdm import tqdm


# ---------------------------------------------------------------- argumentos --
parser = argparse.ArgumentParser(description="Aplica SAM y borra fondo (blanco)")
parser.add_argument("--input",      required=True, help="Carpeta de entrada")
parser.add_argument("--output",     required=True, help="Carpeta de salida")
parser.add_argument("--checkpoint", required=True, help="Checkpoint .pth de SAM")
parser.add_argument("--max-side",   type=int, default=2048,
                    help="Máx. lado que verá SAM (0 = sin downscale)")
args = parser.parse_args()

INPUT_DIR  = Path(args.input)
OUTPUT_DIR = Path(args.output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------- SAM model --
device     = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_TYPE = "vit_b"
sam        = sam_model_registry[MODEL_TYPE](checkpoint=args.checkpoint).to(device)
mask_gen   = SamAutomaticMaskGenerator(
    model=sam,
    points_per_side=64,
    pred_iou_thresh=0.9,
    stability_score_thresh=0.95,
    min_mask_region_area=4096,   # ignora motas pequeñas
)

# ------------------------------------------------------------ helper funcs ----
def resize_for_sam(img: np.ndarray, max_side: int):
    """Devuelve una versión reducida y el factor de escala."""
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
    """Elige la máscara que mejor cubre la franja vertical 30‑90 %."""
    band = slice(int(0.3 * h), int(0.9 * h))
    def score(m):      # nº de píxeles de la máscara dentro de la banda
        return m["segmentation"][band, :].sum()
    candidate = max(masks, key=score)
    if score(candidate) == 0:
        candidate = max(masks, key=lambda m: m["area"])
    return candidate["segmentation"]

# ---------------------------------------------------------------- pipeline ----
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

    # Re‑escalar la máscara si fue necesario
    if scale != 1.0:
        mask = cv2.resize(mask_small.astype(np.uint8),
                          dsize=(img.shape[1], img.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask = mask_small

    # Aplicar máscara (fondo = blanco)
    result = img.copy()
    result[~mask] = 255

    cv2.imwrite(str(OUTPUT_DIR / img_path.name), result)
    print(f"[OK] {img_path.name} → guardado")
