#!/usr/bin/env bash
set -euo pipefail

############################################################
# Mandatory environment variables                          #
############################################################
: "${DATA_PATH:?Environment variable DATA_PATH is not set}"
: "${DATASET_NAME:?Environment variable DATASET_NAME is not set}"
: "${GH_KEY:?Environment variable GH_KEY is not set}"
: "${IMG_COPY_MODE:?Environment variable IMG_COPY_MODE is not set}"
: "${IMG_TYPE:?Environment variable IMG_TYPE is not set}"
: "${SAM:?Environment variable SAM is not set}"
# Optional: FAST=1|true to skip downscale
FAST=${FAST:-0}

############################################################
# Logging helpers                                          #
############################################################
log()  { local tag="$1"; shift; printf '[%s] %s\n' "$tag" "$*"; }
loge() { local tag="$1"; shift; printf '[%s][ERROR] %s\n' "$tag" "$*" >&2; }
die()  { loge "GENERAL" "$*"; exit 1; }

trap _dbg_trap ERR
_dbg_trap() { loge "GENERAL" "Aborted at line $LINENO"; }

ensure_dir() { mkdir -p "$1"; }
prepare_dirs() {
  ensure_dir /root/.ssh
  ensure_dir "$DATA_PATH/images"
  ensure_dir "$DATA_PATH/images_no_bg"
  ensure_dir "$DATA_PATH/colmap/sparse/0_text"
}

prepare_dirs

log "SSH" "Configuring SSH access"
echo "$GH_KEY" > /root/.ssh/id_rsa && chmod 600 /root/.ssh/id_rsa
ssh-keyscan github.com >> /root/.ssh/known_hosts

log "DATASET" "Cloning dataset ${DATASET_NAME}"
git clone --depth 1 git@github.com:dorado-ai-devops/ai-nerf-datasets.git /tmp/tmp_cloned

log "DATASET" "Copying images to $DATA_PATH/images"
case "$IMG_COPY_MODE" in
  TOTAL) cp /tmp/tmp_cloned/${DATASET_NAME}/images/*.${IMG_TYPE} "$DATA_PATH/images" ;;
  ''|*[!0-9]*) die "IMG_COPY_MODE must be 'TOTAL' or an integer" ;;
  *) for i in $(seq 1 $((IMG_COPY_MODE))); do
       cp "/tmp/tmp_cloned/${DATASET_NAME}/images/r_${i}.${IMG_TYPE}" "$DATA_PATH/images"
     done ;;
esac

if [[ "${SAM}" =~ ^(1|true|TRUE)$ ]]; then
  log "SAM" "Removing backgrounds"
  /venv_sr/bin/python3 /app/no_background_sam.py \
    --input      "$DATA_PATH/images" \
    --output     "$DATA_PATH/images_no_bg" \
    --checkpoint /app/checkpoints/sam_vit_b.pth \
    --max-side   2048
  rm -rf "$DATA_PATH/images"
  mv "$DATA_PATH/images_no_bg" "$DATA_PATH/images"
  log "SAM" "Original images replaced"
else
  log "SAM" "Skipped (SAM=${SAM})"
fi

############################################################
# COLMAP                                                   #
############################################################
COLMAP_DIR="$DATA_PATH/colmap"
SPARSE_DIR="$COLMAP_DIR/sparse"
TEXT_DIR="$SPARSE_DIR/0_text"
DB_PATH="$COLMAP_DIR/database.db"
TRANSFORMS_PATH="$DATA_PATH/transforms.json"

# Limpiar resultados previos
rm -rf "$COLMAP_DIR" "$TRANSFORMS_PATH"
ensure_dir "$TEXT_DIR"

# Detectar GPU y número de hilos
if command -v nvidia-smi &> /dev/null; then
  GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  if [ "$GPU_COUNT" -ge 1 ]; then
    USE_GPU=1
    log "COLMAP" "CUDA GPU detectada: activando aceleración GPU"
  else
    USE_GPU=0
    log "COLMAP" "No hay GPU detectada: usando CPU"
  fi
else
  USE_GPU=0
  log "COLMAP" "nvidia-smi no encontrada: deshabilitando GPU"
fi
NUM_THREADS=${NUM_THREADS:-$(nproc)}

log "COLMAP" "Extrayendo features"
colmap feature_extractor \
  --database_path       "$DB_PATH" \
  --image_path          "$DATA_PATH/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --SiftExtraction.use_gpu "$USE_GPU" \
  --SiftExtraction.num_threads "$NUM_THREADS" \
  --SiftExtraction.max_image_size 3200

log "COLMAP" "Emparejando features (exhaustive matcher)"
colmap exhaustive_matcher \
  --database_path          "$DB_PATH" \
  --SiftMatching.use_gpu   "$USE_GPU" \
  --SiftMatching.num_threads "$NUM_THREADS" \
  --SiftMatching.max_ratio 0.8 \
  --SiftMatching.cross_check 1

log "COLMAP" "Reconstruyendo modelo (mapper)"
colmap mapper \
  --database_path             "$DB_PATH" \
  --image_path                "$DATA_PATH/images" \
  --output_path               "$SPARSE_DIR" \
  --Mapper.ba_use_gpu         "$USE_GPU" \
  --Mapper.num_threads        "$NUM_THREADS" \
  --Mapper.ba_global_max_refinements 10 \
  --Mapper.min_num_matches    3 \
  --Mapper.init_min_tri_angle 0.5 \
  --Mapper.abs_pose_min_num_inliers 10 \
  --Mapper.filter_max_reproj_error 5

log "COLMAP" "Convirtiendo modelo a TXT"
colmap model_converter \
  --input_path  "$SPARSE_DIR/0" \
  --output_path "$TEXT_DIR" \
  --output_type TXT

log "COLMAP" "Generando transforms.json"
python3 /colmap/scripts/python/colmap2nerf.py \
  --images           "$DATA_PATH/images" \
  --text             "$TEXT_DIR" \
  --colmap_db        "$DB_PATH" \
  --out              "$TRANSFORMS_PATH" \
  --colmap_camera_model OPENCV \
  > "$DATA_PATH/colmap2nerf_stdout.log" \
  2> "$DATA_PATH/colmap2nerf_stderr.log"

[[ -f "$TRANSFORMS_PATH" ]] || die "transforms.json no se generó"
log "COLMAP" "transforms.json generado satisfactoriamente"
head -n 20 "$TRANSFORMS_PATH"
cp "$TRANSFORMS_PATH" "$DATA_PATH/transforms.json_backup"

############################################################
# Fix relative paths                                       #
############################################################
log "FIX" "Ajustando rutas relativas en transforms.json"
python3 /app/fix_relative_img_paths.py "$TRANSFORMS_PATH" "$DATA_PATH/images"

############################################################
# DOWNSCALE (conditional) & SUMMARY                       #
############################################################
if [[ "${FAST}" =~ ^(1|true|TRUE)$ ]]; then
  log "DOWNSCALE" "Downscaling images by factor 2"
  #python3 /app/downscale.py "$DATA_PATH" 
  log "SUMMARY" "Dataset listo en $DATA_PATH con $(ls "$DATA_PATH/images" | wc -l) imágenes (downscaled)"
else
  log "DOWNSCALE" "Skipped (FAST=${FAST})"
  log "SUMMARY" "Dataset listo en $DATA_PATH con $(ls "$DATA_PATH/images" | wc -l) imágenes (original resolution)"
fi
