#!/usr/bin/env python3
"""
downscale.py <DATA_PATH> <factor>

Crea:
  DATA_PATH/images_lr/…        (imágenes escaladas)
  DATA_PATH/transforms_lr.json (focales y paths corregidos)
"""
import sys, json
from pathlib import Path
from PIL import Image

DATA   = Path(sys.argv[1]).expanduser()
FACTOR = int(sys.argv[2]) 

src = DATA / "images"
dst = DATA / "images_lr"
dst.mkdir(exist_ok=True)

print(f"Reescalando {src} → {dst} ×1/{FACTOR}")
for img in src.iterdir():
    im = Image.open(img)
    im.resize((im.width // FACTOR, im.height // FACTOR), Image.BICUBIC) \
      .save(dst / img.name)

tf_path = DATA / "transforms.json"
tf      = json.loads(tf_path.read_text())

for f in tf["frames"]:
    f["file_path"] = f["file_path"].replace("images", "images_lr")
    for k in ("fl_x", "fl_y", "cx", "cy"):
        f[k] /= FACTOR

out = DATA / "transforms_lr.json"
out.write_text(json.dumps(tf, indent=2))
print("Dataset LR listo:", out)
