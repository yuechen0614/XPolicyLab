#!/usr/bin/env bash
# Run one LIBERO / LIBERO-plus task against an already-running OpenWAM server.
#
# Usage:
#   bash benchmarks/libero/single_eval.sh ordinary libero_spatial 0 8848 127.0.0.1
#
#   bash benchmarks/libero/single_eval.sh plus libero_spatial 0 8848 127.0.0.1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_LIBERO_PATH="/mnt/data/wangyuran/LIBERO"
DEFAULT_LIBERO_PYTHON="/mnt/data/wangyuran/miniconda3/envs/libero/bin/python"
DEFAULT_LIBERO_PLUS_PATH="/mnt/data/wangyuran/LIBERO-plus"
DEFAULT_LIBERO_PLUS_PYTHON="/mnt/data/wangyuran/miniconda3/envs/libero-plus/bin/python"

flavor="${1:-ordinary}"
suite="${2:-libero_spatial}"
task_id="${3:-0}"
port="${4:-${LIBERO_PORT:-8848}}"
host="${5:-${LIBERO_POLICY_HOST:-127.0.0.1}}"

case "${flavor}" in
    ordinary)
        repo_root="${LIBERO_PATH:-${DEFAULT_LIBERO_PATH}}"
        export LIBERO_PATH="${repo_root}"
        python_bin="${LIBERO_PYTHON:-${DEFAULT_LIBERO_PYTHON}}"
        ;;
    plus)
        repo_root="${LIBERO_PLUS_PATH:-${DEFAULT_LIBERO_PLUS_PATH}}"
        export LIBERO_PLUS_PATH="${repo_root}"
        python_bin="${LIBERO_PLUS_PYTHON:-${DEFAULT_LIBERO_PLUS_PYTHON}}"
        ;;
    *)
        echo "[ERROR] Unknown flavor '${flavor}'. Use ordinary | plus." >&2
        exit 1
        ;;
esac

if [[ "${python_bin}" == */* ]]; then
    [[ -x "${python_bin}" ]] || {
        echo "[ERROR] Python not executable for ${flavor}: ${python_bin}" >&2
        exit 1
    }
else
    python_command="${python_bin}"
    python_bin="$(command -v "${python_command}")" || {
        echo "[ERROR] Python command not found for ${flavor}: ${python_command}" >&2
        exit 1
    }
fi

if [[ -n "${POLICY_CONFIG_PATH:-}" ]]; then
    policy_config="${POLICY_CONFIG_PATH}"
elif [[ "${flavor}" == "plus" ]]; then
    policy_config="${SCRIPT_DIR}/policy_config_plus.yml"
else
    policy_config="${SCRIPT_DIR}/policy_config.yml"
fi
[[ -f "${policy_config}" ]] || { echo "[ERROR] policy config not found: ${policy_config}" >&2; exit 1; }
[[ -d "${repo_root}" ]] || { echo "[ERROR] LIBERO repo not found: ${repo_root}" >&2; exit 1; }

export PYTHONPATH="${repo_root}:${SCRIPT_DIR}:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

echo "flavor : ${flavor}"
echo "suite  : ${suite}"
echo "task_id: ${task_id}"
echo "server : ws://${host}:${port}"
echo "python : ${python_bin}"

PYTHONUNBUFFERED=1 "${python_bin}" "${SCRIPT_DIR}/single_eval.py" \
    --config "${policy_config}" \
    --flavor "${flavor}" \
    --suite "${suite}" \
    --task-id "${task_id}" \
    --host "${host}" \
    --port "${port}"
