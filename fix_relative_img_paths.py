import json
import os
import sys

transforms_path = sys.argv[1]
images_dir = sys.argv[2]

with open(transforms_path, 'r') as f:
    data = json.load(f)

missing_files = []
updated_count = 0

for frame in data.get("frames", []):
    orig_path = frame["file_path"]
    filename = os.path.basename(orig_path)
    corrected_path = os.path.join("images", filename)
    abs_image_path = os.path.join(images_dir, filename)

    frame["file_path"] = corrected_path
    updated_count += 1

    if not os.path.exists(abs_image_path):
        missing_files.append(abs_image_path)

with open(transforms_path, 'w') as f:
    json.dump(data, f, indent=2)

print(f"{updated_count} rutas actualizadas en transforms.json hacia imágenes en '{images_dir}'")

if missing_files:
    print("ERROR Rutas no corregidas:")
    for path in missing_files:
        print(f" - {path}")
    sys.exit(1)
else:
    print("Verificación completa: todos los archivos existen.")