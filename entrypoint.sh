#!/usr/bin/env bash
set -euo pipefail


#helpers
log()  { local tag="$1"; shift; printf '[%s] %s\n' "$tag" "$*"; }
loge() { local tag="$1"; shift; printf '[%s][ERROR] %s\n' "$tag" "$*" >&2; }
die()  { loge "GENERAL" "$*"; exit 1; }


dbg_trap() { loge "GENERAL" "Abortado en la línea $LINENO"; }
trap dbg_trap ERR

ensure_dir() { mkdir -p "$1"; }

prepare_dirs() {
  ensure_dir /root/.ssh
  ensure_dir "$DATA_PATH/images"
  ensure_dir "$DATA_PATH/images_centered"
  ensure_dir "$DATA_PATH/colmap/sparse/0_text"
}

check_env() {
  local required=(DATA_PATH DATASET_NAME GH_KEY IMG_COPY_MODE IMG_TYPE)
  for var in "${required[@]}"; do
    [[ -z "${!var:-}" ]] && die "Variable $var no definida"
  done
}

############################################################
# START
############################################################
check_env
prepare_dirs

log "SSH" "Configurando SSH"
echo "$GH_KEY" > /root/.ssh/id_rsa
chmod 600 /root/.ssh/id_rsa
ssh-keyscan github.com >> /root/.ssh/known_hosts

log "DATASET" "Clonando dataset ${DATASET_NAME}"
git clone --depth 1 git@github.com:dorado-ai-devops/ai-nerf-datasets.git /tmp/tmp_cloned

log "DATASET" "Copiando imágenes a $DATA_PATH/images"
case "$IMG_COPY_MODE" in
  TOTAL)
    cp /tmp/tmp_cloned/${DATASET_NAME}/images/*.${IMG_TYPE} "$DATA_PATH/images" ;;
  ''|*[!0-9]*)
    die "IMG_COPY_MODE debe ser 'TOTAL' o un número entero." ;;
  *)
    for i in $(seq 0 $((IMG_COPY_MODE - 1))); do
      cp "/tmp/tmp_cloned/${DATASET_NAME}/images/r_${i}.${IMG_TYPE}" "$DATA_PATH/images"
    done ;;
esac

log "SAM" "Centrando objetos y removiendo fondos"
/venv_sr/bin/python3 /app/center_with_sam.py \
  --input "$DATA_PATH/images" \
  --output "$DATA_PATH/images_centered" \
  --checkpoint /app/checkpoints/sam_vit_b.pth \
  --size 768

rm -rf "$DATA_PATH/images"
mv "$DATA_PATH/images_centered" "$DATA_PATH/images"
log "SAM" "Imágenes reemplazadas por versiones centradas"

############################################################
# COLMAP
############################################################
COLMAP_DIR="$DATA_PATH/colmap"
SPARSE_DIR="$COLMAP_DIR/sparse"
TEXT_DIR="$SPARSE_DIR/0_text"
DB_PATH="$COLMAP_DIR/database.db"
TRANSFORMS_PATH="$DATA_PATH/transforms.json"

rm -rf "$COLMAP_DIR" "$TRANSFORMS_PATH"
ensure_dir "$TEXT_DIR"

log "COLMAP" "Extrayendo características"
colmap feature_extractor \
  --database_path "$DB_PATH" \
  --image_path "$DATA_PATH/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu 1

log "COLMAP" "Matching exhaustivo complementario"
colmap exhaustive_matcher --database_path "$DB_PATH"

log "COLMAP" "Reconstruyendo modelo (mapper)"
colmap mapper \
  --database_path "$DB_PATH" \
  --image_path "$DATA_PATH/images" \
  --output_path "$SPARSE_DIR" \
  --Mapper.ba_global_max_refinements 10 \
  --Mapper.min_num_matches 3 \
  --Mapper.init_min_tri_angle 0.5 \
  --Mapper.abs_pose_min_num_inliers 10 \
  --Mapper.filter_max_reproj_error 5

log "COLMAP" "Convirtiendo modelo a TXT"
colmap model_converter \
  --input_path "$SPARSE_DIR/0" \
  --output_path "$TEXT_DIR" \
  --output_type TXT

log "COLMAP" "Generando transforms.json"
python3 /colmap/scripts/python/colmap2nerf.py \
  --images "$DATA_PATH/images" \
  --text "$TEXT_DIR" \
  --colmap_db "$DB_PATH" \
  --out "$TRANSFORMS_PATH" \
  --colmap_camera_model OPENCV \
  > "$DATA_PATH/colmap2nerf_stdout.log" \
  2> "$DATA_PATH/colmap2nerf_stderr.log"

[[ -f "$TRANSFORMS_PATH" ]] || die "No se generó transforms.json"

log "COLMAP" "transforms.json generado correctamente"
head -n 20 "$TRANSFORMS_PATH"
cp "$TRANSFORMS_PATH" "$DATA_PATH/transforms.json_backup"

log "FIX" "Corrigiendo rutas relativas"
python3 /app/fix_relative_img_paths.py "$TRANSFORMS_PATH" "$DATA_PATH/images"

log "DOWNSCALE" "Reduciendo resolución x2"
python3 /app/downscale.py "$DATA_PATH" 2

log "SUMMARY" "Dataset listo en $DATA_PATH con $(ls "$DATA_PATH/images" | wc -l) imágenes"
