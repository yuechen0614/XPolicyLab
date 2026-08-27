#!/usr/bin/env bash
# Smoke launcher for ordinary LIBERO and LIBERO-plus.
#
# Usage:
#   bash benchmarks/libero/run_smoke.sh ordinary task
#   bash benchmarks/libero/run_smoke.sh plus task
#
# Modes:
#   import  - verify package import and generated LIBERO_CONFIG_PATH
#   task    - import + retrieve one benchmark task and its BDDL file
#   env     - task + instantiate OffScreenRenderEnv; requires assets

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LIBERO_PATH="/mnt/data/wangyuran/LIBERO"
DEFAULT_LIBERO_PYTHON="/mnt/data/wangyuran/miniconda3/envs/libero/bin/python"
DEFAULT_LIBERO_PLUS_PATH="/mnt/data/wangyuran/LIBERO-plus"
DEFAULT_LIBERO_PLUS_PYTHON="/mnt/data/wangyuran/miniconda3/envs/libero-plus/bin/python"

FLAVOR="${1:-${LIBERO_FLAVOR:-ordinary}}"
MODE="${2:-${LIBERO_SMOKE_MODE:-import}}"

case "${FLAVOR}" in
    ordinary)
        REPO_ROOT="${LIBERO_PATH:-${DEFAULT_LIBERO_PATH}}"
        export LIBERO_PATH="${REPO_ROOT}"
        export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_ROOT:-${HOME}/.libero-openwam}"
        PYTHON_BIN="${LIBERO_PYTHON:-${DEFAULT_LIBERO_PYTHON}}"
        ;;
    plus)
        REPO_ROOT="${LIBERO_PLUS_PATH:-${DEFAULT_LIBERO_PLUS_PATH}}"
        export LIBERO_PLUS_PATH="${REPO_ROOT}"
        export LIBERO_CONFIG_PATH="${LIBERO_PLUS_CONFIG_ROOT:-${HOME}/.libero-openwam-plus}"
        PYTHON_BIN="${LIBERO_PLUS_PYTHON:-${DEFAULT_LIBERO_PLUS_PYTHON}}"
        ;;
    *)
        echo "[ERROR] Unknown LIBERO flavor '${FLAVOR}'. Use ordinary | plus." >&2
        exit 1
        ;;
esac

case "${MODE}" in
    import|task|env) ;;
    *)
        echo "[ERROR] Unknown smoke mode '${MODE}'. Use import | task | env." >&2
        exit 1
        ;;
esac

if [[ "${PYTHON_BIN}" == */* ]]; then
    [[ -x "${PYTHON_BIN}" ]] || {
        echo "[ERROR] Python not executable for ${FLAVOR}: ${PYTHON_BIN}" >&2
        exit 1
    }
else
    PYTHON_COMMAND="${PYTHON_BIN}"
    PYTHON_BIN="$(command -v "${PYTHON_COMMAND}")" || {
        echo "[ERROR] Python command not found for ${FLAVOR}: ${PYTHON_COMMAND}" >&2
        exit 1
    }
fi
if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "[ERROR] Repository not found for ${FLAVOR}: ${REPO_ROOT}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${LIBERO_SMOKE_GPU:-0}}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${LIBERO_SMOKE_GPU:-0}}"

echo "[libero-smoke] flavor=${FLAVOR} mode=${MODE} python=${PYTHON_BIN}"
echo "[libero-smoke] repo=${REPO_ROOT}"
echo "[libero-smoke] config=${LIBERO_CONFIG_PATH}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/smoke_libero.py" \
    --flavor "${FLAVOR}" \
    --mode "${MODE}" \
    --suite "${LIBERO_SMOKE_SUITE:-libero_spatial}" \
    --task-id "${LIBERO_SMOKE_TASK_ID:-0}" \
    --camera-size "${LIBERO_SMOKE_CAMERA_SIZE:-128}" \
    --steps "${LIBERO_SMOKE_STEPS:-1}"
