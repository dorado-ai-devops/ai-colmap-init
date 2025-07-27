# center_with_sam.py
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from tqdm import tqdm
import argparse

# CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True, help='Carpeta de entrada con las imagenes')
parser.add_argument('--output', required=True, help='Carpeta de salida')
parser.add_argument('--checkpoint', required=True, help='Ruta al .pth de SAM')
parser.add_argument('--size', type=int, default=768, help='Tamaño cuadrado de crop final')
args = parser.parse_args()

INPUT_DIR = Path(args.input)
OUTPUT_DIR = Path(args.output)
CHECKPOINT = args.checkpoint
MODEL_TYPE = "vit_b"
CROP_SIZE = args.size

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load SAM
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).to(device)
mask_generator = SamAutomaticMaskGenerator(sam)

def center_crop_from_mask(image, mask, crop_size):
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None

    cx, cy = int(np.mean(xs)), int(np.mean(ys))
    h, w = image.shape[:2]
    half = crop_size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half)
    y1, y2 = max(0, cy - half), min(h, cy + half)

    # Aplica máscara
    masked = image.copy()
    for c in range(3):
        masked[..., c] = np.where(mask, masked[..., c], 255)

    cropped = masked[y1:y2, x1:x2]
    final = cv2.copyMakeBorder(
        cropped,
        top=max(0, half - cy),
        bottom=max(0, (cy + half) - h),
        left=max(0, half - cx),
        right=max(0, (cx + half) - w),
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],
    )
    return final

# Procesa cada imagen
for img_path in tqdm(sorted(INPUT_DIR.glob("*.[jp][pn]g"))):
    img = cv2.imread(str(img_path))
    masks = mask_generator.generate(img)
    if not masks:
        continue
    best = max(masks, key=lambda m: m["area"])["segmentation"]
    cropped = center_crop_from_mask(img, best, crop_size=CROP_SIZE)
    if cropped is not None:
        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), cropped)
        print(f"Procesada: {img_path.name} -> {out_path.name}")
    else:
        print(f"Advertencia: No se pudo centrar {img_path.name}, máscara vacía.")