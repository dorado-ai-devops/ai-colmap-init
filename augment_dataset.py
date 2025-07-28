import albumentations as A, cv2, glob, os
import re

IMG_DIR = "images"
IMG_TYPE = "jpg"

aug = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussianBlur(p=0.3),
    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=5, p=0.7)
])

# Buscar último índice usado
existing = glob.glob(f"{IMG_DIR}/r_*.{IMG_TYPE}")
indices = [
    int(re.search(r"r_(\d+)\.", os.path.basename(f)).group(1))
    for f in existing if re.search(r"r_(\d+)\.", os.path.basename(f))
]
next_index = max(indices) + 1 if indices else 1

# Procesar imágenes
input_files = sorted(glob.glob(f"{IMG_DIR}/r_*.{IMG_TYPE}"))
print(f"[INFO] Generando 3 augmentaciones por imagen ({len(input_files)} originales)...")

for fn in input_files:
    im = cv2.imread(fn)
    if im is None:
        print(f"[WARN] No se pudo leer: {fn}")
        continue
    for i in range(3):
        out = aug(image=im)["image"]
        out_path = f"{IMG_DIR}/r_{next_index:03d}.{IMG_TYPE}"
        cv2.imwrite(out_path, out)
        print(f"[AUG] {out_path}")
        next_index += 1
