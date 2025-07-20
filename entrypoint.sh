#!/bin/bash
set -e

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

mkdir -p ${DATA_PATH:?}/colmap"
echo "==> Ejecutando COLMAP..."
colmap automatic_reconstructor \
    --image_path "$DATA_PATH/images" \
    --workspace_path "$DATA_PATH/colmap" \
    --use_gpu 1

echo "==> Generando transforms.json para Instant-NGP..."
python3 /colmap/scripts/python/colmap2nerf.py \
    --colmap_path "$DATA_PATH/colmap" \
    --images "$DATA_PATH/images"

echo "==> Moviendo transforms.json a nivel superior..."
mv "$DATA_PATH/images/transforms.json" "$DATA_PATH/transforms.json"

echo "✅ Dataset preparado en $DATA_PATH"
