#!/bin/bash
set -euo pipefail 

: "${DATA_PATH:?Variable DATA_PATH no definida}"
: "${DATASET_NAME:?Variable DATASET_NAME no definida}"
: "${GH_KEY:?Variable GH_KEY no definida}"
: "${IMG_COPY_MODE:?Variable IMG_COPY_MODE no definida}" 
: "${IMG_TYPE:?Variable IMG_TYPE no definida}" 

echo "==> Configurando SSH"
mkdir -p /root/.ssh
echo "$GH_KEY" > /root/.ssh/id_rsa
chmod 600 /root/.ssh/id_rsa
ssh-keyscan github.com >> /root/.ssh/known_hosts

echo "==> Clonando dataset..."
git clone git@github.com:dorado-ai-devops/ai-nerf-datasets.git /tmp/tmp_cloned

echo "==> Copiando imágenes al directorio de entrenamiento: $DATA_PATH"
mkdir -p "$DATA_PATH/images"

if [ "$IMG_COPY_MODE" == "TOTAL" ]; then
    cp -r /tmp/tmp_cloned/${DATASET_NAME}/images/*.${IMG_TYPE} "$DATA_PATH/images"
elif [[ "$IMG_COPY_MODE" =~ ^[0-9]+$ ]]; then
    for i in $(seq 0 $((IMG_COPY_MODE - 1))); do
        cp "/tmp/tmp_cloned/${DATASET_NAME}/images/r_${i}.${IMG_TYPE}" "$DATA_PATH/images"
    done
else
    echo "Error: IMG_COPY_MODE debe ser 'TOTAL' o un número entero."
    exit 1
fi

echo "==> Ejecutando pipeline COLMAP paso a paso..."

COLMAP_DIR="$DATA_PATH/colmap"
SPARSE_DIR="$COLMAP_DIR/sparse"
TEXT_DIR="$SPARSE_DIR/0_text"
DB_PATH="$COLMAP_DIR/database.db"
TRANSFORMS_PATH="$DATA_PATH/transforms.json"

rm -rf "$COLMAP_DIR" "$TRANSFORMS_PATH"
mkdir -p "$COLMAP_DIR"

# 1. Extracción de características
echo "==> Extrayendo características"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$DATA_PATH/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV

# 2. Matching secuencial
echo "==> Realizando matching secuencial"
colmap sequential_matcher \
  --database_path "$DB_PATH" \
  --SiftMatching.use_gpu 1

# 3. Matching exhaustivo complementario
echo "==> Realizando matching exhaustivo complementario"
colmap exhaustive_matcher \
  --database_path "$DB_PATH"

# 4. Mapeo
echo "==> Reconstruyendo modelo (mapper)"
mkdir -p "$SPARSE_DIR"
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$DATA_PATH/images" \
  --output_path "$SPARSE_DIR" \
  --Mapper.ba_global_max_refinements 5 \
  --Mapper.min_num_matches 5 \
  --Mapper.init_min_tri_angle 1

# 5. Conversión a TXT
echo "==> Convirtiendo modelo a formato TXT"
mkdir -p "$TEXT_DIR"
colmap model_converter \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$TEXT_DIR" \
  --output_type TXT

# 6. Generación de transforms.json
echo "==> Generando transforms.json"
python3 /colmap/scripts/python/colmap2nerf.py \
  --images "$DATA_PATH/images" \
  --text "$TEXT_DIR" \
  --colmap_db "$DB_PATH" \
  --out "$TRANSFORMS_PATH" \
  --colmap_camera_model OPENCV \
  --aabb_scale 2 \
  > "$DATA_PATH/colmap2nerf_stdout.log" \
  2> "$DATA_PATH/colmap2nerf_stderr.log"

if [ ! -f "$TRANSFORMS_PATH" ]; then
    echo "❌ Error: No se generó transforms.json"
    cat "$DATA_PATH/colmap2nerf_stderr.log"
    exit 1
fi

echo "✅ transforms.json generado correctamente"
echo "🔎 Primeras líneas:"
head -n 20 "$TRANSFORMS_PATH"

echo "📦 Dataset listo en $DATA_PATH"
echo "  - Imágenes: $(ls "$DATA_PATH/images" | wc -l)"
