#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Submit a multi-node Cosmos3-Edge training job to PAI-DLC.
#
# Modeled on scripts/dlc_robotwin_bench.sh (same CreateJob payload shape, region
# presets and CPFS mount), but submits a PyTorchJob that runs the training
# launcher on every pod. DLC sets WORLD_SIZE / RANK / MASTER_ADDR per pod, which
# scripts/train.sh already consumes as its NNODES / NODE_RANK / MASTER_ADDR
# fallbacks — no torchrun wiring needed here.
#
# Global batch is the knob: --global-batch is split into a per-GPU batch and
# gradient-accumulation steps over (pods x 8) GPUs, and the script refuses a
# combination that does not divide evenly (silently training at the wrong batch
# is worse than failing to submit).
#
# Prerequisites:
#   - `aliyun` CLI configured with VALID credentials for the target workspace
#     (`aliyun configure` / re-login; an expired OAuth token fails with
#     "init client failed failed to refresh token").
#   - Repo checkout + training venv reachable on the shared CPFS the pods mount.
#   - WANDB_API_KEY exported (or --no-wandb / WANDB_MODE=offline).
#
# Example (4 nodes x 8 H20, global batch 3072, 5 epochs):
#   WANDB_API_KEY=<key> bash scripts/dlc_train_cosmos3.sh \
#       --run-id cosmos3_edge_jsa_gb3072 --pod-count 4 \
#       --global-batch 3072 --epochs 5
#
# Validate the request body without submitting (key redacted):
#   bash scripts/dlc_train_cosmos3.sh --run-id test --print-payload
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Region presets (same infra as the bench launcher) ----
REGION="${REGION:-cn-hangzhou}"
WORKSPACE_ID="${WORKSPACE_ID:-}"
RESOURCE_ID="${RESOURCE_ID:-}"
QUOTA_INSTANCE_TYPE="${QUOTA_INSTANCE_TYPE:-ml.gu8tf.8.46xlarge}"
IMAGE="${IMAGE:-}"
VPC_ID="${VPC_ID:-}"
VSWITCH_ID="${VSWITCH_ID:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
EXTENDED_CIDR="${EXTENDED_CIDR-__UNSET__}"
CPFS_URI="${CPFS_URI:-}"

apply_region_preset() {
    case "${REGION}" in
        cn-hangzhou)
            : "${WORKSPACE_ID:=590221}"
            : "${RESOURCE_ID:=quotai7l7gqmouqw}"
            : "${IMAGE:=ego-pretrain-registry-vpc.cn-hangzhou.cr.aliyuncs.com/egoscale/openwam:openwam-bench-v1}"
            : "${VPC_ID:=vpc-bp1w4vnrtyrs76z1uyj4c}"
            : "${VSWITCH_ID:=vsw-bp1fv5lj7oh7hbii0j0ae}"
            : "${SECURITY_GROUP_ID:=sg-bp1hzayjxnkruu2rg26j}"
            : "${CPFS_URI:=bmcpfs://cpfs-00000ub3ici1dnniit2i0-vpc-egtdgw.cn-hangzhou.cpfs.aliyuncs.com/}"
            [[ "${EXTENDED_CIDR}" == "__UNSET__" ]] && EXTENDED_CIDR=""
            ;;
        cn-beijing)
            : "${WORKSPACE_ID:=405678}"
            : "${RESOURCE_ID:=quota1bgicjuzf8z}"
            : "${IMAGE:=wj-bj-acr-registry-vpc.cn-beijing.cr.aliyuncs.com/ai-repo/wuji-repo-beijing:openwam-bench-v1}"
            : "${VPC_ID:=vpc-2ze1ywgqtrxh03g7yrz15}"
            : "${VSWITCH_ID:=vsw-2zee4f58k649nzf6y8x7m}"
            : "${SECURITY_GROUP_ID:=sg-2ze9g0xva2jgqzfcxt6a}"
            : "${CPFS_URI:=bmcpfs://cpfs-03001ycys4kpfutqkv8lf-vpc-8sd4w3.cn-beijing.cpfs.aliyuncs.com/}"
            [[ "${EXTENDED_CIDR}" == "__UNSET__" ]] && EXTENDED_CIDR="10.33.1.0/24"
            ;;
        *)
            echo "[ERROR] No preset for region '${REGION}'. Supported: cn-hangzhou | cn-beijing." >&2
            exit 1
            ;;
    esac
}

JOB_NAME="${JOB_NAME:-cosmos3-train}"
JOB_PRIORITY="${JOB_PRIORITY:-9}"
POD_COUNT="${POD_COUNT:-4}"
GPUS_PER_POD="${GPUS_PER_POD:-8}"

# Paths as seen from inside the pods (CPFS is mounted at /mnt/cpfs/).
REPO="${REPO:-/mnt/cpfs/zch/openwam_cosmos3_dev}"
VENV_BIN="${VENV_BIN:-/mnt/cpfs/zch/wuji-openwam-dev/.venv/bin}"
MODEL_PATH="${COSMOS3_MODEL_PATH:-/mnt/cpfs/zch/assets/Cosmos3-Edge}"
DATASET_DIR="${DATASET_DIR:-/mnt/cpfs/wangyuran/RoboTwin2.0/dataset}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/mnt/cpfs/zch/checkpoints}"

VARIANT="${VARIANT:-joint_self_attn}"
GLOBAL_BATCH="${GLOBAL_BATCH:-3072}"
PER_GPU_BATCH="${PER_GPU_BATCH:-32}"   # measured peak: 26.7GB @16, 37.2GB @32, 62.1GB @64 (8-rank ZeRO-2)
EPOCHS="${EPOCHS:-5}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
SAVE_STEPS="${SAVE_STEPS:-1500}"       # micro-steps (global_step), not optimizer steps
KEEP_LAST_K="${KEEP_LAST_K:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"

WANDB_PROJECT="${WANDB_PROJECT:-open-wam}"
WANDB_ENTITY="${WANDB_ENTITY:-wuji-tech}"
USE_WANDB=1

RUN_ID=""
PRINT_PAYLOAD=0
EXTRA_ENVS=()
EXTRA_OVERRIDES=()

usage() {
    cat >&2 <<'EOF'
Usage: bash scripts/dlc_train_cosmos3.sh --run-id <id> [options]

Required:
  --run-id ID              Run name; used for DisplayName, output dir and wandb run.

Options:
  --pod-count N            Nodes (default 4; each contributes 8 GPUs)
  --global-batch N         Target global batch (default 3072)
  --per-gpu-batch N        Per-GPU micro-batch (default 32); accum is derived
  --epochs N               Training epochs (default 5)
  --lr LR                  Base learning rate (default 1e-4)
  --variant NAME           joint_self_attn | joint_cross_attn (default joint_self_attn)
  --save-steps N           Checkpoint interval in MICRO steps (default 1500)
  --repo PATH              Repo root on CPFS (default /mnt/cpfs/zch/openwam_cosmos3_dev)
  --region REGION          cn-hangzhou (default) | cn-beijing
  --no-wandb               Disable wandb (sets WANDB_MODE=disabled)
  --env KEY=VALUE          Extra CustomEnv (repeatable)
  --set K=V                Extra hydra override appended to the launcher (repeatable)
  --print-payload          Print the CreateJob body (WANDB_API_KEY redacted) and exit
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id) RUN_ID="$2"; shift 2;;
        --pod-count) POD_COUNT="$2"; shift 2;;
        --global-batch) GLOBAL_BATCH="$2"; shift 2;;
        --per-gpu-batch) PER_GPU_BATCH="$2"; shift 2;;
        --epochs) EPOCHS="$2"; shift 2;;
        --lr) LEARNING_RATE="$2"; shift 2;;
        --variant) VARIANT="$2"; shift 2;;
        --save-steps) SAVE_STEPS="$2"; shift 2;;
        --repo) REPO="$2"; shift 2;;
        --region) REGION="$2"; shift 2;;
        --job-name) JOB_NAME="$2"; shift 2;;
        --priority) JOB_PRIORITY="$2"; shift 2;;
        --image) IMAGE="$2"; shift 2;;
        --no-wandb) USE_WANDB=0; shift;;
        --env) EXTRA_ENVS+=("$2"); shift 2;;
        --set) EXTRA_OVERRIDES+=("$2"); shift 2;;
        --print-payload) PRINT_PAYLOAD=1; shift;;
        -h|--help) usage; exit 0;;
        *) echo "[ERROR] Unknown argument: $1" >&2; usage; exit 1;;
    esac
done

[[ -n "${RUN_ID}" ]] || { echo "[ERROR] --run-id is required." >&2; usage; exit 1; }
apply_region_preset
command -v aliyun >/dev/null 2>&1 || { echo "[ERROR] aliyun CLI not found in PATH" >&2; exit 1; }

# ---- Batch math: global = per_gpu x gpus x accum, must divide evenly ----
TOTAL_GPUS=$((POD_COUNT * GPUS_PER_POD))
PER_STEP=$((PER_GPU_BATCH * TOTAL_GPUS))
if (( GLOBAL_BATCH % PER_STEP != 0 )); then
    echo "[ERROR] global batch ${GLOBAL_BATCH} is not divisible by per_gpu_batch x gpus = ${PER_GPU_BATCH} x ${TOTAL_GPUS} = ${PER_STEP}." >&2
    echo "[ERROR] Pick a per-GPU batch that divides $((GLOBAL_BATCH / TOTAL_GPUS)) (= global / gpus)." >&2
    exit 1
fi
GRAD_ACCUM=$((GLOBAL_BATCH / PER_STEP))
OUTPUT_PATH="${OUTPUT_ROOT}/${RUN_ID}"

if (( USE_WANDB )); then
    if [[ -z "${WANDB_API_KEY:-}" ]]; then
        echo "[ERROR] WANDB_API_KEY is not set. Get it from https://wandb.ai/authorize," >&2
        echo "[ERROR] then re-run with: WANDB_API_KEY=<key> bash $0 ... (or pass --no-wandb)." >&2
        exit 1
    fi
    WANDB_ENVS=(
        "WANDB_API_KEY=${WANDB_API_KEY}"
        "WANDB_PROJECT=${WANDB_PROJECT}"
        "WANDB_ENTITY=${WANDB_ENTITY}"
        "WANDB_DIR=${OUTPUT_PATH}"
    )
else
    WANDB_ENVS=("WANDB_MODE=disabled")
fi

read -r -d '' TRAIN_SCRIPT <<CMD || true
set -euo pipefail
cd ${REPO}
export PATH=${VENV_BIN}:\$PATH
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export OUTPUT_PATH=${OUTPUT_PATH}
export RUN_NAME=${RUN_ID}
export COSMOS3_MODEL_PATH=${MODEL_PATH}
export DATASET_DIR=${DATASET_DIR}
mkdir -p ${OUTPUT_PATH}
bash scripts/train_cosmos3_edge_default_weights.sh \
  model.architecture.variant=${VARIANT} \
  training.batch_size=${PER_GPU_BATCH} \
  training.gradient_accumulation_steps=${GRAD_ACCUM} \
  training.num_epochs=${EPOCHS} \
  training.video_lr=${LEARNING_RATE} \
  training.save_steps=${SAVE_STEPS} \
  training.keep_last_k_ckpts=${KEEP_LAST_K} \
  training.dataset_num_workers=${NUM_WORKERS} \
  project.wandb.project=${WANDB_PROJECT} \
  project.wandb.entity=${WANDB_ENTITY} \
  ${EXTRA_OVERRIDES[@]:-} \
  2>&1 | tee -a ${OUTPUT_PATH}/train_rank\${RANK:-0}.log
CMD

# DLC runs UserCommand under /bin/sh (dash), which rejects `set -o pipefail`
# and other bashisms — wrap the body in an explicit bash -c. Single quotes in
# the body would break the wrapper, so refuse rather than ship a mangled command.
if [[ "${TRAIN_SCRIPT}" == *"'"* ]]; then
    echo "[ERROR] Generated training command contains a single quote; the bash -c wrapper cannot escape it." >&2
    exit 1
fi
USER_CMD="bash -c '${TRAIN_SCRIPT}'"

_ENVS_JSON=$(python3 - "$@" <<'PY' "${WANDB_ENVS[@]}" "${EXTRA_ENVS[@]:-}"
import json, sys
out = []
for kv in sys.argv[1:]:
    if not kv or "=" not in kv:
        continue
    k, _, v = kv.partition("=")
    out.append({"Key": k, "Value": v, "Visible": "public"})
print(json.dumps(out))
PY
)

export _JOB_NAME="${JOB_NAME}" _RUN_ID="${RUN_ID}" _WORKSPACE_ID="${WORKSPACE_ID}" \
       _RESOURCE_ID="${RESOURCE_ID}" _JOB_PRIORITY="${JOB_PRIORITY}" _USER_CMD="${USER_CMD}" \
       _VPC_ID="${VPC_ID}" _VSWITCH_ID="${VSWITCH_ID}" _SECURITY_GROUP_ID="${SECURITY_GROUP_ID}" \
       _EXTENDED_CIDR="${EXTENDED_CIDR}" _CPFS_URI="${CPFS_URI}" _IMAGE="${IMAGE}" \
       _POD_COUNT="${POD_COUNT}" _QUOTA_INSTANCE_TYPE="${QUOTA_INSTANCE_TYPE}" \
       _EXTRA_ENVS_JSON="${_ENVS_JSON}"

PAYLOAD=$(python3 <<'PY'
import json, os
body = {
    "DisplayName": f"{os.environ['_JOB_NAME']}-{os.environ['_RUN_ID']}",
    "JobType": "PyTorchJob",
    "WorkspaceId": os.environ["_WORKSPACE_ID"],
    "ResourceId": os.environ["_RESOURCE_ID"],
    "Priority": int(os.environ["_JOB_PRIORITY"]),
    "UserCommand": os.environ["_USER_CMD"],
    "UserVpc": {
        k: v for k, v in {
            "VpcId": os.environ["_VPC_ID"],
            "SwitchId": os.environ["_VSWITCH_ID"],
            "SecurityGroupId": os.environ["_SECURITY_GROUP_ID"],
            "ExtendedCIDRs": [os.environ["_EXTENDED_CIDR"]] if os.environ.get("_EXTENDED_CIDR") else None,
            "DefaultRoute": "eth0",
        }.items() if v is not None
    },
    "DataSources": [{"MountPath": "/mnt/cpfs/", "Uri": os.environ["_CPFS_URI"]}],
    "CustomEnvs": [
        {"Key": "NCCL_SOCKET_IFNAME", "Value": "eth0", "Visible": "public"},
        {"Key": "NCCL_IB_DISABLE", "Value": "0", "Visible": "public"},
        {"Key": "NCCL_P2P_DISABLE", "Value": "0", "Visible": "public"},
        {"Key": "NCCL_DEBUG", "Value": "WARN", "Visible": "public"},
        *json.loads(os.environ.get("_EXTRA_ENVS_JSON") or "[]"),
    ],
    "Settings": {"EnableRDMA": True, "QuotaInstanceTypes": [os.environ["_QUOTA_INSTANCE_TYPE"]]},
    "JobSpecs": [{
        "Type": "Worker",
        "Image": os.environ["_IMAGE"],
        "PodCount": int(os.environ["_POD_COUNT"]),
        "ResourceConfig": {"CPU": "128", "GPU": "8", "Memory": "512Gi", "SharedMemory": "512Gi"},
    }],
}
print(json.dumps(body, ensure_ascii=False))
PY
)

cat <<BANNER
╔══════════════════════════════════════════════════════════════╗
║  Cosmos3-Edge DLC training launcher
║  Region / WS   : ${REGION} / ${WORKSPACE_ID}
║  Pods × GPU    : ${POD_COUNT} × ${GPUS_PER_POD} = ${TOTAL_GPUS} (${QUOTA_INSTANCE_TYPE})
║  Batch         : ${PER_GPU_BATCH}/GPU × ${TOTAL_GPUS} GPU × ${GRAD_ACCUM} accum = ${GLOBAL_BATCH} global
║  Variant       : ${VARIANT}   Epochs: ${EPOCHS}   LR: ${LEARNING_RATE}
║  Repo / venv   : ${REPO}
║  Output        : ${OUTPUT_PATH}
║  wandb         : $( ((USE_WANDB)) && echo "${WANDB_ENTITY}/${WANDB_PROJECT} (run ${RUN_ID})" || echo disabled )
╚══════════════════════════════════════════════════════════════╝
BANNER

if (( PRINT_PAYLOAD )); then
    python3 -c '
import json,sys
b=json.loads(sys.stdin.read())
for e in b.get("CustomEnvs", []):
    if "API_KEY" in e["Key"] or "TOKEN" in e["Key"]:
        e["Value"] = "<REDACTED>"
print(json.dumps(b, indent=2, ensure_ascii=False))' <<< "${PAYLOAD}"
    exit 0
fi

echo "[INFO] Submitting DLC training job..."
JOB_RESP=$(aliyun pai-dlc CreateJob --region "${REGION}" --body "${PAYLOAD}")
JOB_ID=$(printf '%s' "${JOB_RESP}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["JobId"])')
echo "[INFO] Created job: ${JOB_ID}"
echo "[INFO] Console: https://pai.console.aliyun.com/?regionId=${REGION}&workspaceId=${WORKSPACE_ID}#/dlc/jobs/${JOB_ID}"
echo "[INFO] Status:  aliyun pai-dlc GetJob --region ${REGION} --JobId ${JOB_ID}"
echo "[INFO] Logs:    tail -f ${OUTPUT_PATH}/train_rank0.log   (shared CPFS)"
