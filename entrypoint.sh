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
    # Si IMG_COPY_MODE es TOTAL, copiar todas las imágenes
    cp -r /tmp/tmp_cloned/${DATASET_NAME}/images/*.${IMG_TYPE} "$DATA_PATH/images"
elif [[ "$IMG_COPY_MODE" =~ ^[0-9]+$ ]]; then
    # Si IMG_COPY_MODE es un número entero, copiar solo las primeras r imágenes
    for i in $(seq 0 $((IMG_COPY_MODE - 1))); do
        cp "/tmp/tmp_cloned/${DATASET_NAME}/images/r_${i}.${IMG_TYPE}" "$DATA_PATH/images"
    done
else
    echo "Error: IMG_COPY_MODE debe ser 'TOTAL' o un número entero."
    exit 1
fi

mkdir -p "${DATA_PATH:?}/colmap"
echo "==> Ejecutando reconstruccion COLMAP..."
if ! colmap automatic_reconstructor \
    --image_path "$DATA_PATH/images" \
    --workspace_path "$DATA_PATH/colmap" \
    --use_gpu 1; then
    echo "Error: Falló el proceso de generacion de matrices de cámara con COLMAP"
    exit 1
fi


mkdir -p "${DATA_PATH:?}/colmap/sparse/0_text"
echo "==> Convirtiendo modelo COLMAP a formato TXT..."
if ! colmap model_converter \
    --input_path "$DATA_PATH/colmap/sparse/0" \
    --output_path "$DATA_PATH/colmap/sparse/0_text" \
    --output_type TXT; then
    echo "Error: Falló la conversión del modelo COLMAP a formato TXT"
    exit 1
fi

echo "==> Generando transforms.json para Instant-NGP en formato OpenCV..."

TRANSFORMS_PATH="${DATA_PATH}/transforms.json"

python3 /colmap/scripts/python/colmap2nerf.py \
  --images "$DATA_PATH/images" \
  --text "$DATA_PATH/colmap/sparse/0_text" \
  --colmap_db "$DATA_PATH/colmap/database.db" \
  --out "$TRANSFORMS_PATH" \
  --colmap_camera_model OPENCV \
  --aabb_scale 2 \
  > "${DATA_PATH}/colmap2nerf_stdout.log" \
  2> "${DATA_PATH}/colmap2nerf_stderr.log"

EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Error: colmap2nerf.py falló con código $EXIT_CODE"
    echo "--- STDOUT ---"
    cat "${DATA_PATH}/colmap2nerf_stdout.log"
    echo "--- STDERR ---"
    cat "${DATA_PATH}/colmap2nerf_stderr.log"
    exit 1
fi

if [ ! -f "$TRANSFORMS_PATH" ]; then
    echo "Error: No se encontró transforms.json después de la conversión"
    echo "--- STDERR ---"
    cat "${DATA_PATH}/colmap2nerf_stderr.log"
    exit 1
fi

echo "transforms.json generado correctamente en ${TRANSFORMS_PATH}"
echo "Tamaño: $(du -h "$TRANSFORMS_PATH" | cut -f1)"
echo "Validación rápida (primeras líneas):"
head -n 20 "$TRANSFORMS_PATH"


if [ ! -f "$DATA_PATH/transforms.json" ]; then
    echo "Error: No se encontró transforms.json después de la conversión"
    exit 1
fi


if [ ! -d "$DATA_PATH/images" ] || [ ! -f "$DATA_PATH/transforms.json" ]; then
    echo "Error: Estructura final del dataset incompleta"
    exit 1
fi

echo "Dataset preparado exitosamente en $DATA_PATH"
echo "  - Imágenes: $(ls "$DATA_PATH/images" | wc -l) archivos"
echo "  - Reconstrucción COLMAP: completada"
echo "  - transforms.json: generado"
echo "==> Listo para entrenamiento con Instant-NGP"