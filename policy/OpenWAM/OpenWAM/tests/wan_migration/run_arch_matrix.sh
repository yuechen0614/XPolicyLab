#!/bin/bash
# Architecture-level golden regression gate.
#
# Drives run_arch_compare.py across the full architecture x backbone x
# external-encoder matrix on TWO checkouts (a refactor/feature checkout and a
# baseline checkout), then asserts each config's forward + loss is bit-identical
# between them. This is the architecture-level analogue of run_backbone.py
# (which only covers the bare video backbone).
#
# Usage:
#   REF_DIR=/path/to/feature MAIN_DIR=/path/to/baseline \
#     bash tests/wan_migration/run_arch_matrix.sh [all|standard|encoder]
#
#   REF_DIR   feature/refactor checkout (default: this repo's root)
#   MAIN_DIR  baseline (golden) checkout to diff against (default:
#             REF_DIR/references/wuji-openwam-dev, a main-branch checkout)
#   OUT_DIR   where dumps + cmp_*.txt land (default: /mnt/data/wangyuran/arch_cmp)
#   subset    all (default) | standard (16 backbone configs) | encoder (5 swaps)
#
# Runs 4 configs in parallel (one per GPU). PASS criteria live in compare_arch.py.
set -u

REF_DIR="${REF_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
MAIN_DIR="${MAIN_DIR:-$REF_DIR/references/wuji-openwam-dev}"
OUT="${OUT_DIR:-/mnt/data/wangyuran/arch_cmp}"
SUBSET="${1:-all}"
NGPU="${NGPU:-4}"
mkdir -p "$OUT"

# Per-backbone (name, weights, resolution). VACE-1.3B / I2V-14B only support
# 480x832; TI2V-5B is flexible (here matched to the RoboTwin dataloader's 384x320).
TI2V="model.video_backbone.name=wan22_ti2v_5b model.video_backbone.model_path=/mnt/data/wangyuran/Wan2.2-TI2V-5B"
VACE="model.video_backbone.name=wan21_vace_1_3b model.video_backbone.model_path=/mnt/data/wangyuran/Wan2.1-VACE-1.3B"
I2V="model.video_backbone.name=wan21_i2v_14b_480p model.video_backbone.model_path=/mnt/data/limingleyang/weights/Wan2.1-I2V-14B-480P"
# Encoder swaps run on the TI2V-5B DiT structure with from_scratch=true (the DiT
# is rebuilt+reinitialized to the encoder's z_dim; VAE/text encoder stay loaded).
ENC_BASE="model=dual_system model.architecture.variant=joint_self_attn $TI2V model.video_backbone.from_scratch=true"

declare -a CONFIGS
# add NAME H W REF_OV MAIN_OV — both overrides are always explicit. They are
# identical for dual/tri/encoder and differ only for shared_backbone (its
# variant moved from architecture.variant to the action_backbone group).
add() { CONFIGS+=("$1|$2|$3|$4|$5"); }

if [ "$SUBSET" = smoke ]; then
  # Two representative configs (standard + a from_scratch encoder) to validate
  # the harness + orchestration end-to-end without the full ~30min matrix.
  ov="model=dual_system model.architecture.variant=joint_self_attn $TI2V"
  add "dual_self_ti2v" 384 320 "$ov" "$ov"
  ov="$ENC_BASE model.video_backbone.encoder.name=wan_vae model.video_backbone.encoder.model_path=/mnt/data/wangyuran/Wan2.2-TI2V-5B"
  add "enc_wan_vae" 384 320 "$ov" "$ov"
fi

if [ "$SUBSET" = all ] || [ "$SUBSET" = standard ]; then
  for vt in joint_self_attn:dual_self joint_cross_attn:dual_cross idm:dual_idm; do
    variant="${vt%%:*}"; tag="${vt##*:}"
    add "${tag}_ti2v" 384 320 "model=dual_system model.architecture.variant=$variant $TI2V" "model=dual_system model.architecture.variant=$variant $TI2V"
    add "${tag}_vace" 480 832 "model=dual_system model.architecture.variant=$variant $VACE" "model=dual_system model.architecture.variant=$variant $VACE"
    add "${tag}_i2v"  480 832 "model=dual_system model.architecture.variant=$variant $I2V" "model=dual_system model.architecture.variant=$variant $I2V"
  done
  add "tri_ti2v" 384 320 "model=tri_system $TI2V" "model=tri_system $TI2V"
  add "tri_vace" 480 832 "model=tri_system $VACE" "model=tri_system $VACE"
  add "tri_i2v"  480 832 "model=tri_system $I2V"  "model=tri_system $I2V"
fi

# shared_backbone: both sides now select the variant via
# `model.architecture.variant=X` (the action_backbone group was unified into a
# single shared_action_backbone.yaml), matching dual/tri_system.
if [ "$SUBSET" = all ] || [ "$SUBSET" = standard ] || [ "$SUBSET" = shared ]; then
  for vt in vanilla:shared_van moe:shared_moe; do
    variant="${vt%%:*}"; tag="${vt##*:}"
    ov="model=shared_backbone model.architecture.variant=$variant"
    add "${tag}_ti2v" 384 320 "$ov $TI2V" "$ov $TI2V"
    add "${tag}_vace" 480 832 "$ov $VACE" "$ov $VACE"
    add "${tag}_i2v"  480 832 "$ov $I2V"  "$ov $I2V"
  done
fi

if [ "$SUBSET" = all ] || [ "$SUBSET" = encoder ]; then
  # name:weights_dir. vjepa2_1 needs `pip install timm decord`; a submodule-era
  # baseline (e.g. e89cac6) also needs third_party/vjepa2 — see ARCH_COMPARE.md.
  for ep in \
    wan_vae:/mnt/data/wangyuran/Wan2.2-TI2V-5B \
    flux_vae:/mnt/data/wangyuran/FLUX.2-dev_VAE \
    dinov3:/mnt/data/wangyuran/DINOv3 \
    vjepa2_1:/mnt/data/limingleyang/weights/vjepa2_1 ; do
    en="${ep%%:*}"; path="${ep##*:}"
    ov="$ENC_BASE model.video_backbone.encoder.name=$en model.video_backbone.encoder.model_path=$path"
    add "enc_$en" 384 320 "$ov" "$ov"
  done
fi

run_pair() {
  local gpu="$1" name="$2" H="$3" W="$4" ref_ov="$5" main_ov="$6"
  echo "[$name] START gpu=$gpu $(date +%H:%M:%S)"
  local side dir out ov
  for side in ref main; do
    dir="$REF_DIR"; out="new"; ov="$ref_ov"
    [ "$side" = main ] && { dir="$MAIN_DIR"; out="old"; ov="$main_ov"; }
    CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=disabled PYTHONPATH="$dir" \
      python "$dir/tests/wan_migration/run_arch_compare.py" \
      --out "$OUT/${out}_$name.pt" --seed 42 --mode both --num-frames 9 --height "$H" --width "$W" \
      -- $ov > "$OUT/${side}_$name.log" 2>&1
  done
  if [ -f "$OUT/old_$name.pt" ] && [ -f "$OUT/new_$name.pt" ]; then
    PYTHONPATH="$REF_DIR" python "$REF_DIR/tests/wan_migration/compare_arch.py" \
      "$OUT/old_$name.pt" "$OUT/new_$name.pt" > "$OUT/cmp_$name.txt" 2>&1
    echo "[$name] $(grep '^RESULT' "$OUT/cmp_$name.txt" || echo NO_RESULT)"
  else
    echo "[$name] CRASH :: ref:$(grep -iE 'Error|Exception|No module' "$OUT/ref_$name.log" | tail -1) || main:$(grep -iE 'Error|Exception|No module' "$OUT/main_$name.log" | tail -1)"
  fi
}

gpu=0; n=0
for cfg in "${CONFIGS[@]}"; do
  IFS='|' read -r name H W ref_ov main_ov <<< "$cfg"
  run_pair "$gpu" "$name" "$H" "$W" "$ref_ov" "$main_ov" &
  gpu=$(( (gpu + 1) % NGPU )); n=$(( n + 1 ))
  [ $(( n % NGPU )) -eq 0 ] && wait
done
wait

echo "===== MATRIX DONE $(date +%H:%M:%S) ====="
echo "PASS: $(grep -l 'RESULT: PASS' "$OUT"/cmp_*.txt 2>/dev/null | wc -l) / $(ls "$OUT"/cmp_*.txt 2>/dev/null | wc -l) compared"
grep -L 'RESULT: PASS' "$OUT"/cmp_*.txt 2>/dev/null | sed 's/.*cmp_/FAIL: /;s/\.txt//'
