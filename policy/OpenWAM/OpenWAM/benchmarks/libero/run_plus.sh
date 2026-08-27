#!/usr/bin/env bash
# Run LIBERO-plus with its one-trial-per-task protocol and seven-category summary.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
INFERENCE_HORIZON="${INFERENCE_HORIZON:-10}"
RUN_ROOT="${RUN_ROOT:-${REPO_ROOT}/outputs/libero_plus/10epoch_horizon${INFERENCE_HORIZON}_${RUN_TAG}}"
CKPT_DIR="${CKPT_DIR:-/mnt/data/wangyuran/openwam_checkpoints/new-openwam-libero-plus-sft-10epoch-final}"
CKPT_NAME="${CKPT_NAME:-checkpoint_step_10850.safetensors}"
SERVER_PYTHON="${SERVER_PYTHON:-/usr/bin/python3.12}"
LIBERO_PLUS_PYTHON="${LIBERO_PLUS_PYTHON:-/mnt/data/wangyuran/miniconda3/envs/libero-plus/bin/python}"
LIBERO_PLUS_PATH="${LIBERO_PLUS_PATH:-/mnt/data/wangyuran/LIBERO-plus}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-2}"
CLIENT_MAX_ATTEMPTS="${CLIENT_MAX_ATTEMPTS:-3}"
CLIENT_RETRY_DELAY="${CLIENT_RETRY_DELAY:-2}"
BASE_PORT="${BASE_PORT:-8940}"

mkdir -p "${RUN_ROOT}"

echo "[$(date --iso-8601=seconds)] starting LIBERO-plus evaluation"
"${SERVER_PYTHON}" "${SCRIPT_DIR}/run_10epoch_all_suites.py" \
    --flavor plus \
    --ckpt-dir "${CKPT_DIR}" \
    --ckpt-name "${CKPT_NAME}" \
    --gpus "${GPUS}" \
    --replicas-per-gpu "${REPLICAS_PER_GPU}" \
    --client-max-attempts "${CLIENT_MAX_ATTEMPTS}" \
    --client-retry-delay "${CLIENT_RETRY_DELAY}" \
    --base-port "${BASE_PORT}" \
    --libero-python "${LIBERO_PLUS_PYTHON}" \
    --libero-path "${LIBERO_PLUS_PATH}" \
    --policy-config "${SCRIPT_DIR}/policy_config_plus.yml" \
    --num-trials 1 \
    --inference-horizon "${INFERENCE_HORIZON}" \
    --output-dir "${RUN_ROOT}" \
    "$@" \
    2>&1 | tee "${RUN_ROOT}/launcher.log"
echo "[$(date --iso-8601=seconds)] finished LIBERO-plus evaluation"
