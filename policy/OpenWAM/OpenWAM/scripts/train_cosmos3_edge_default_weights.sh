#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Cosmos3-Edge training launcher — DEFAULT pretrained weights.
#
# dual_system / joint_cross_attn on the released nvidia/Cosmos3-Edge bundle
# (diffusers-style snapshot; the und text stream + Wan2.2 VAE are frozen, the
# gen pathway trains). No external text encoder — the bundle's tokenizer + the
# und tower handle text natively (context width auto-derives to 2048).
#
# Launch (single node, auto-detects GPUs):
#   bash scripts/train_cosmos3_edge_default_weights.sh
# Quick smoke (20 steps + a checkpoint, tiny batch):
#   bash scripts/train_cosmos3_edge_default_weights.sh training.debug=true training.batch_size=1
# Extra hydra overrides pass through, e.g.:
#   bash scripts/train_cosmos3_edge_default_weights.sh training.video_lr=5e-5
#
# MEMORY NOTE: 3.1B trainable-side transformer + z=48 latents; start at
# batch_size=8 on 97 GB H20s and raise once you've seen the peak. The frozen
# und tower runs under no_grad and adds no activation memory.
# This launcher pins joint_cross_attn; override with
# model.architecture.variant=joint_self_attn|idm (or model=shared_backbone
# model.architecture.variant=vanilla|moe), all of which this backbone supports.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# Reuse a venv carrying the GPU stack (torch+cu128 / diffusers>=0.38 / deepspeed).
VENV_BIN="${VENV_BIN:-/mnt/cpfs/zch/wuji-openwam-dev/.venv/bin}"
export PATH="${VENV_BIN}:${PATH}"

COSMOS3_MODEL_PATH="${COSMOS3_MODEL_PATH:-/mnt/cpfs/zch/assets/Cosmos3-Edge}"
DATASET_DIR="${DATASET_DIR:-/mnt/cpfs/wangyuran/RoboTwin2.0/dataset}"
OUTPUT_PATH="${OUTPUT_PATH:-/mnt/cpfs/zch/checkpoints/cosmos3_edge_joint_cross_attn}"
RUN_NAME="${RUN_NAME:-cosmos3_edge_joint_cross_attn}"

bash scripts/train.sh \
  model=dual_system \
  model.architecture.variant=joint_cross_attn \
  model/video_backbone=cosmos3_edge \
  model.video_backbone.model_path="${COSMOS3_MODEL_PATH}" \
  'model.freeze=[video_backbone.vae]' \
  dataloader.dataset_dir="${DATASET_DIR}" \
  dataloader.variant=both \
  dataloader.task_name=null \
  dataloader.val_ratio=0.0 \
  training.video_lr=1e-4 \
  training.batch_size=8 \
  training.gradient_accumulation_steps=1 \
  training.num_epochs=5 \
  training.save_steps=2000 \
  training.keep_last_k_ckpts=2 \
  training.dataset_num_workers=4 \
  training.output_path="${OUTPUT_PATH}" \
  project.wandb.run_name="${RUN_NAME}" \
  "$@"
