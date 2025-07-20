#!/bin/bash
set -euo pipefail 


: "${DATA_PATH:?Variable DATA_PATH no definida}"
: "${DATASET_NAME:?Variable DATASET_NAME no definida}"
: "${GH_KEY:?Variable GH_KEY no definida}"
: "${IMG_TOTAL_SIZE:?Variable IMG_TOTAL_SIZE no definida}" 

echo "==> Limpieza previa de datos temporales y anteriores"
rm -rf /tmp/tmp_cloned
rm -rf "${DATA_PATH:?}/colmap"
rm -rf "${DATA_PATH:?}/images"
rm -f "${DATA_PATH:?}/transforms.json"


echo "==> Configurando SSH"
mkdir -p /root/.ssh
echo "$GH_KEY" > /root/.ssh/id_rsa
chmod 600 /root/.ssh/id_rsa
ssh-keyscan github.com >> /root/.ssh/known_hosts

echo "==> Clonando dataset..."
git clone git@github.com:dorado-ai-devops/ai-nerf-datasets.git /tmp/tmp_cloned

echo "==> Copiando imágenes al directorio de entrenamiento: $DATA_PATH"
mkdir -p "$DATA_PATH/images"
cp -r /tmp/tmp_cloned/${DATASET_NAME}/* "$DATA_PATH/images"

if [ "$IMG_TOTAL_SIZE" == "TOTAL" ]; then
    # Si IMG_TOTAL_SIZE es TOTAL, copiar todas las imágenes
    cp -r /tmp/tmp_cloned/${DATASET_NAME}/images/* "$DATA_PATH/images"
elif [[ "$IMG_TOTAL_SIZE" =~ ^[0-9]+$ ]]; then
    # Si IMG_TOTAL_SIZE es un número entero, copiar solo las primeras r imágenes
    for i in $(seq 0 $((IMG_TOTAL_SIZE - 1))); do
        cp "/tmp/tmp_cloned/${DATASET_NAME}/images/r_${i}" "$DATA_PATH/images"
    done
else
    echo "Error: IMG_TOTAL_SIZE debe ser 'TOTAL' o un número entero."
    exit 1
fi

mkdir -p "${DATA_PATH:?}/colmap"
echo "==> Ejecutando COLMAP..."
if ! colmap automatic_reconstructor \
    --image_path "$DATA_PATH/images" \
    --workspace_path "$DATA_PATH/colmap" \
    --use_gpu 1; then
    echo "Error: Falló el proceso de generacion de matrices de cámara con COLMAP"
    exit 1
fi

echo "==> Generando transforms.json para Instant-NGP en formato OpenCV..."
if ! python3 /colmap/scripts/python/colmap2nerf.py \
    --images "$DATA_PATH/images" \
    --text "$DATA_PATH/colmap" \
    --out "$DATA_PATH/transforms.json" \
    --colmap_camera_model OPENCV \
    --aabb_scale 2; then
    echo "Error: Falló la generación de transforms.json con colmap2nerf.py para OpenCV"
    exit 1
fi


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