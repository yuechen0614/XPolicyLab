#!/usr/bin/env bash
# One-step launcher: submit a RoboTwin-bench DLC job in cn-beijing, wait until
# it is Running, then start the benchmark web console locally against the
# shared CPFS log directory.
#
# Prerequisites:
#   - `aliyun` CLI configured with credentials and default region.
#   - `python3` available on this host.
#   - This host has CPFS mounted at /mnt/cpfs (same shared filesystem the DLC
#     pods see), so web_control can read the live log directory.
#
# --ckpt-dir and --run-id are REQUIRED (no defaults). Example:
#   bash scripts/dlc_robotwin_bench.sh \
#       --ckpt-dir /mnt/cpfs/wangyuran/openwam_checkpoints/robotwin_dual_system_joint_self_attention_64bs/ \
#       --run-id   DSJS_64bs_eval001 \
#       --mode     demo_clean \
#       --tasks    'adjust_bottle,open_laptop' \
#       --pod-count 4 \
#       --web-port 8765
#
# Dry-run the request body (does NOT submit, still validates --ckpt-dir/--run-id):
#   bash scripts/dlc_robotwin_bench.sh --ckpt-dir <path> --run-id <id> --print-payload
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Region presets (cloned from latest production robotwin-bench jobs) ----
# Selected by --region; individual fields can still be overridden via flag/env.
REGION="${REGION:-cn-beijing}"
WORKSPACE_ID="${WORKSPACE_ID:-}"
RESOURCE_ID="${RESOURCE_ID:-}"
QUOTA_INSTANCE_TYPE="${QUOTA_INSTANCE_TYPE:-ml.gu8tf.8.46xlarge}"
IMAGE="${IMAGE:-}"
VPC_ID="${VPC_ID:-}"
VSWITCH_ID="${VSWITCH_ID:-}"
SECURITY_GROUP_ID="${SECURITY_GROUP_ID:-}"
EXTENDED_CIDR="${EXTENDED_CIDR-__UNSET__}"  # empty string is a valid override (no CIDR)
CPFS_URI="${CPFS_URI:-}"

apply_region_preset() {
    case "${REGION}" in
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
        *)
            echo "[ERROR] No preset for region '${REGION}'. Supported: cn-beijing | cn-hangzhou." >&2
            echo "[ERROR] To use an unsupported region, export WORKSPACE_ID/RESOURCE_ID/IMAGE/VPC_ID/VSWITCH_ID/SECURITY_GROUP_ID/CPFS_URI before invocation." >&2
            exit 1
            ;;
    esac
}

JOB_NAME="${JOB_NAME:-robotwin-bench}"
JOB_PRIORITY="${JOB_PRIORITY:-9}"
POD_COUNT="${POD_COUNT:-8}"
CKPT_DIR="${CKPT_DIR:-}"               # REQUIRED — set via --ckpt-dir or CKPT_DIR env
ROBOTWIN_PATH="${ROBOTWIN_PATH:-/mnt/cpfs/zch/RoboTwin}"
ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON:-/mnt/cpfs/zch/envs/RoboTwin/bin/python}"
REPO_ROOT="${REPO_ROOT:-${REPO_ROOT_DEFAULT}}"

POLICY_NAME="${POLICY_NAME:-openwam}"
TASK_MODE="${TASK_MODE:-all}"          # demo_clean | demo_randomized | all
TASKS="${TASKS:-all}"                  # positional list passed to dlc_parallel_eval.sh
DENOISE_STEPS="${DENOISE_STEPS:-10}"
EXTRA_DEPLOY_ARGS=""                   # e.g. "--denoise-mode async --variance-shift-alpha 9.0"
RUN_ID="${ROBOTWIN_RUN_ID:-}"          # REQUIRED — set via --run-id or ROBOTWIN_RUN_ID env

WEB_HOST="${WEB_HOST:-0.0.0.0}"
WEB_PORT="${WEB_PORT:-8765}"
WEB_PYTHON="${WEB_PYTHON:-python3}"
WEB_OPEN_TIMEOUT_SEC="${WEB_OPEN_TIMEOUT_SEC:-1800}"
RUNNING_POLL_SEC="${RUNNING_POLL_SEC:-15}"
RUNNING_TIMEOUT_SEC="${RUNNING_TIMEOUT_SEC:-3600}"

PRINT_PAYLOAD=0
NO_WEB=0
NO_CKPT_CHECK=0
ATTACH_JOB_ID=""
EXTRA_ENVS=()  # additional CustomEnvs, each item "KEY=VALUE"

usage() {
    cat >&2 <<'EOF'
Usage: bash scripts/dlc_robotwin_bench.sh --ckpt-dir <path> --run-id <id> [options]

Submit a RoboTwin-bench DLC job in cn-beijing (cloned from the production
config), wait for it to reach Running, then start the benchmark web console
locally against the shared CPFS log directory.

Required:
  --ckpt-dir PATH        OpenWAM checkpoint dir (NO DEFAULT). Must exist on this host.
  --run-id ID            Shared run id used in log paths & ROBOTWIN_RUN_ID (NO DEFAULT).

Options:
  --policy-name NAME     -n flag for dlc_parallel_eval.sh (default: openwam)
  --mode MODE            demo_clean | demo_randomized | all (default: all)
  --tasks LIST           Task list or "all" or path to file (default: all)
  --denoise-steps N      Forwarded to scripts/deploy.py (default: 10)
  --extra-deploy ARGS    Extra args appended to dlc_parallel_eval.sh deploy block
  --pod-count N          Worker pods (default: 8)
  --image URI            Docker image (default: production openwam-bench-v1)
  --job-name NAME        DisplayName prefix (default: robotwin-bench)
  --priority N           Job priority 1..9 (default: 9)
  --region REGION        cn-beijing | cn-hangzhou (default: cn-beijing).
                         Switches workspace/quota/image/vpc/cpfs presets in one go.
  --web-host HOST        web_control bind host (default: 0.0.0.0)
  --web-port PORT        web_control port    (default: 8765)
  --web-python BIN       web_control python  (default: python3)
  --no-web               Skip starting web_control after the job is Running
  --no-ckpt-check        Skip the local --ckpt-dir existence check (submitting host lacks CPFS)
  --attach JOB_ID        Skip submission; reuse an existing job and open web_control
  --env KEY=VALUE        Append a CustomEnv to the job (repeatable, e.g. --env DRY_RUN_SLEEP_SEC=120)
  --print-payload        Print the CreateJob JSON and exit
  -h, --help             This help

Override defaults via env vars too (CKPT_DIR, RUN_ID, POD_COUNT, WEB_PORT ...).

Workflow:
  1. CreateJob via `aliyun pai-dlc create-job` (clone of dlcrvg3rew1o4u7j).
  2. Poll GetJob until Status=Running (failure states exit non-zero).
  3. Wait for the shared log directory created by rank-0 of the eval script.
  4. Exec `python3 benchmarks/web_control.py <log_dir> --host 0.0.0.0 --port 8765`.
EOF
}

while (( $# > 0 )); do
    case "$1" in
        --ckpt-dir)        CKPT_DIR="$2"; shift 2 ;;
        --policy-name)     POLICY_NAME="$2"; shift 2 ;;
        --mode)            TASK_MODE="$2"; shift 2 ;;
        --tasks)           TASKS="$2"; shift 2 ;;
        --denoise-steps)   DENOISE_STEPS="$2"; shift 2 ;;
        --extra-deploy)    EXTRA_DEPLOY_ARGS="$2"; shift 2 ;;
        --run-id)          RUN_ID="$2"; shift 2 ;;
        --pod-count)       POD_COUNT="$2"; shift 2 ;;
        --image)           IMAGE="$2"; shift 2 ;;
        --job-name)        JOB_NAME="$2"; shift 2 ;;
        --priority)        JOB_PRIORITY="$2"; shift 2 ;;
        --region)          REGION="$2"; shift 2 ;;
        --web-host)        WEB_HOST="$2"; shift 2 ;;
        --web-port)        WEB_PORT="$2"; shift 2 ;;
        --web-python)      WEB_PYTHON="$2"; shift 2 ;;
        --no-web)          NO_WEB=1; shift ;;
        --no-ckpt-check)   NO_CKPT_CHECK=1; shift ;;
        --attach)          ATTACH_JOB_ID="$2"; shift 2 ;;
        --env)             EXTRA_ENVS+=("$2"); shift 2 ;;
        --print-payload)   PRINT_PAYLOAD=1; shift ;;
        -h|--help)         usage; exit 0 ;;
        *)                 echo "[ERROR] Unknown arg: $1" >&2; usage; exit 1 ;;
    esac
done

apply_region_preset

for var in WORKSPACE_ID RESOURCE_ID IMAGE VPC_ID VSWITCH_ID SECURITY_GROUP_ID CPFS_URI; do
    [[ -n "${!var}" ]] || {
        echo "[ERROR] ${var} is empty after preset apply for region '${REGION}'. Set it explicitly." >&2
        exit 1; }
done

[[ -n "${CKPT_DIR}" ]] || {
    echo "[ERROR] --ckpt-dir is required (no default). Pass --ckpt-dir <path> or set CKPT_DIR." >&2
    usage; exit 1; }
[[ -n "${RUN_ID}" ]] || {
    echo "[ERROR] --run-id is required (no default). Pass --run-id <id> or set ROBOTWIN_RUN_ID." >&2
    usage; exit 1; }
[[ "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]] || {
    echo "[ERROR] --run-id must contain only alnum, '.', '_', '-' (got '${RUN_ID}'); it is interpolated into log paths." >&2; exit 1; }
[[ "${TASK_MODE}" =~ ^(demo_clean|demo_randomized|all)$ ]] || {
    echo "[ERROR] --mode must be demo_clean|demo_randomized|all (got '${TASK_MODE}')" >&2; exit 1; }
if [[ ! -d "${CKPT_DIR}" ]]; then
    # The check assumes the submitting host mounts the same CPFS as the pods.
    # When submitting from a laptop that does not, pass --no-ckpt-check after
    # verifying the path exists on the shared filesystem some other way.
    if (( NO_CKPT_CHECK )); then
        echo "[WARN] --ckpt-dir not visible here (${CKPT_DIR}); --no-ckpt-check given, trusting the pods can see it." >&2
    else
        echo "[ERROR] --ckpt-dir does not exist on this host: ${CKPT_DIR}" >&2
        echo "[ERROR] If this host does not mount CPFS, verify the path on the shared FS and pass --no-ckpt-check." >&2
        exit 1
    fi
fi
[[ -d /mnt/cpfs ]] || \
    echo "[HINT] /mnt/cpfs is not mounted on this host. If --ckpt-dir lives under /mnt/cpfs the DLC pods will still see it, but this host will not — web_control cannot tail the log dir from here." >&2
command -v aliyun >/dev/null 2>&1 || { echo "[ERROR] aliyun CLI not found in PATH" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "[ERROR] python3 not found in PATH" >&2; exit 1; }
if (( NO_WEB == 0 )); then
    command -v "${WEB_PYTHON}" >/dev/null 2>&1 || {
        echo "[ERROR] --web-python not found in PATH: ${WEB_PYTHON}" >&2; exit 1; }
fi

CKPT_DIR_NORMAL="${CKPT_DIR%/}"
LOG_DIR="${CKPT_DIR_NORMAL}/robotwin_eval_logs/${POLICY_NAME}_${TASK_MODE}_dlc_${RUN_ID}"

# UserCommand replicates the running prod job's invocation.
USER_CMD="cd ${REPO_ROOT}
. .venv/bin/activate
bash benchmarks/robotwin/dlc_parallel_eval.sh \\
    -m ${TASK_MODE} -n ${POLICY_NAME} -d ${CKPT_DIR} \\
    --denoise-steps ${DENOISE_STEPS} ${EXTRA_DEPLOY_ARGS} ${TASKS}"

export _JOB_NAME="${JOB_NAME}" _RUN_ID="${RUN_ID}" _WORKSPACE_ID="${WORKSPACE_ID}"
export _RESOURCE_ID="${RESOURCE_ID}" _JOB_PRIORITY="${JOB_PRIORITY}" _IMAGE="${IMAGE}"
export _POD_COUNT="${POD_COUNT}" _VPC_ID="${VPC_ID}" _VSWITCH_ID="${VSWITCH_ID}"
export _SECURITY_GROUP_ID="${SECURITY_GROUP_ID}" _EXTENDED_CIDR="${EXTENDED_CIDR}"
export _CPFS_URI="${CPFS_URI}" _ROBOTWIN_PATH="${ROBOTWIN_PATH}" _ROBOTWIN_PYTHON="${ROBOTWIN_PYTHON}"
export _QUOTA_INSTANCE_TYPE="${QUOTA_INSTANCE_TYPE}" _USER_CMD="${USER_CMD}"

# pack EXTRA_ENVS as JSON for python to consume (one python3 call regardless of count)
for kv in "${EXTRA_ENVS[@]}"; do
    [[ "${kv}" == *=* ]] || { echo "[ERROR] --env must be KEY=VALUE, got '${kv}'" >&2; exit 1; }
done
if (( ${#EXTRA_ENVS[@]} > 0 )); then
    _EXTRA_ENVS_JSON=$(python3 -c '
import json, sys
out = []
for kv in sys.argv[1:]:
    k, _, v = kv.partition("=")
    out.append({"Key": k, "Value": v, "Visible": "public"})
print(json.dumps(out))
' "${EXTRA_ENVS[@]}")
else
    _EXTRA_ENVS_JSON="[]"
fi
export _EXTRA_ENVS_JSON

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
    "DataSources": [{
        "MountPath": "/mnt/cpfs/",
        "Uri": os.environ["_CPFS_URI"],
    }],
    "CustomEnvs": [
        {"Key": "NCCL_SOCKET_IFNAME", "Value": "eth0",                       "Visible": "public"},
        {"Key": "NCCL_IB_DISABLE",    "Value": "0",                          "Visible": "public"},
        {"Key": "NCCL_P2P_DISABLE",   "Value": "0",                          "Visible": "public"},
        {"Key": "NCCL_DEBUG",         "Value": "INFO",                       "Visible": "public"},
        {"Key": "ROBOTWIN_PATH",      "Value": os.environ["_ROBOTWIN_PATH"],   "Visible": "public"},
        {"Key": "ROBOTWIN_PYTHON",    "Value": os.environ["_ROBOTWIN_PYTHON"], "Visible": "public"},
        {"Key": "ROBOTWIN_RUN_ID",    "Value": os.environ["_RUN_ID"],          "Visible": "public"},
        *json.loads(os.environ.get("_EXTRA_ENVS_JSON") or "[]"),
    ],
    "Settings": {
        "EnableRDMA": True,
        "QuotaInstanceTypes": [os.environ["_QUOTA_INSTANCE_TYPE"]],
    },
    "JobSpecs": [{
        "Type": "Worker",
        "Image": os.environ["_IMAGE"],
        "PodCount": int(os.environ["_POD_COUNT"]),
        "ResourceConfig": {
            "CPU": "128", "GPU": "8", "Memory": "512Gi", "SharedMemory": "512Gi"
        },
    }],
}
print(json.dumps(body, ensure_ascii=False))
PY
)

if (( PRINT_PAYLOAD )); then
    python3 -c 'import json,sys;print(json.dumps(json.loads(sys.stdin.read()), indent=2, ensure_ascii=False))' <<< "${PAYLOAD}"
    exit 0
fi

cat <<BANNER
╔══════════════════════════════════════════════════════════════╗
║  OpenWAM RoboTwin-bench DLC launcher                        ║
║  Region        : ${REGION}
║  Image         : ${IMAGE}
║  Pods × GPU    : ${POD_COUNT} × 8 (${QUOTA_INSTANCE_TYPE})
║  Ckpt          : ${CKPT_DIR}
║  Mode / Tasks  : ${TASK_MODE} / ${TASKS}
║  Denoise steps : ${DENOISE_STEPS}
║  Run id        : ${RUN_ID}
║  Log dir       : ${LOG_DIR}
║  Web console   : http://${WEB_HOST}:${WEB_PORT}/   (--no-web to skip)
╚══════════════════════════════════════════════════════════════╝
BANNER

if [[ -n "${ATTACH_JOB_ID}" ]]; then
    JOB_ID="${ATTACH_JOB_ID}"
    echo "[INFO] Attaching to existing job: ${JOB_ID}"
else
    echo "[INFO] Submitting DLC job..."
    JOB_RESP=$(aliyun pai-dlc CreateJob --region "${REGION}" --body "${PAYLOAD}")
    JOB_ID=$(printf '%s' "${JOB_RESP}" | python3 -c 'import json,sys;print(json.load(sys.stdin)["JobId"])')
    echo "[INFO] Created job: ${JOB_ID}"
    echo "[INFO] PAI console: https://pai.console.aliyun.com/?regionId=${REGION}&workspaceId=${WORKSPACE_ID}#/dlc/jobs/${JOB_ID}"
fi

echo "[INFO] Polling status (timeout=${RUNNING_TIMEOUT_SEC}s)..."
DEADLINE=$((SECONDS + RUNNING_TIMEOUT_SEC))
LAST_STATUS=""
while :; do
    JOB_JSON=$(aliyun pai-dlc GetJob --region "${REGION}" --JobId "${JOB_ID}")
    STATUS=$(printf '%s' "${JOB_JSON}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("Status",""))')
    if [[ "${STATUS}" != "${LAST_STATUS}" ]]; then
        echo "[INFO] [$(date +%H:%M:%S)] status: ${STATUS}"
        LAST_STATUS="${STATUS}"
    fi
    case "${STATUS}" in
        Running) break ;;
        Succeeded)
            echo "[WARN] Job finished before we observed Running (likely a fast --dryrun)." >&2
            echo "[WARN] Proceeding to web_control if the log dir exists." >&2
            break
            ;;
        Failed|Stopped)
            echo "[ERROR] Job ended with status=${STATUS}." >&2
            printf '%s\n' "${JOB_JSON}" | python3 -c \
                'import json,sys;d=json.load(sys.stdin);print("ReasonCode:",d.get("ReasonCode"));print("ReasonMessage:",d.get("ReasonMessage"))' >&2
            exit 2
            ;;
    esac
    if (( SECONDS >= DEADLINE )); then
        echo "[ERROR] Timed out waiting for Running (last status=${STATUS})." >&2
        exit 3
    fi
    sleep "${RUNNING_POLL_SEC}"
done

if (( NO_WEB )); then
    echo "[INFO] Job is Running. Skipping web_control (--no-web)."
    echo "[INFO] Log dir: ${LOG_DIR}"
    exit 0
fi

echo "[INFO] Waiting for shared log directory (rank-0 creates it after server warmup)..."
LOG_DEADLINE=$((SECONDS + WEB_OPEN_TIMEOUT_SEC))
while [[ ! -d "${LOG_DIR}" ]]; do
    # If the job dies after Running (OOM, scheduler kill, eval crash), bail fast
    # instead of waiting WEB_OPEN_TIMEOUT_SEC for a log dir that will never appear.
    JOB_JSON=$(aliyun pai-dlc GetJob --region "${REGION}" --JobId "${JOB_ID}")
    STATUS=$(printf '%s' "${JOB_JSON}" | python3 -c 'import json,sys;print(json.load(sys.stdin).get("Status",""))')
    case "${STATUS}" in
        Failed|Stopped)
            echo "[ERROR] Job ended with status=${STATUS} while waiting for log dir." >&2
            printf '%s\n' "${JOB_JSON}" | python3 -c \
                'import json,sys;d=json.load(sys.stdin);print("ReasonCode:",d.get("ReasonCode"));print("ReasonMessage:",d.get("ReasonMessage"))' >&2
            exit 2
            ;;
    esac
    if (( SECONDS >= LOG_DEADLINE )); then
        echo "[ERROR] Log dir did not appear within ${WEB_OPEN_TIMEOUT_SEC}s: ${LOG_DIR}" >&2
        echo "[HINT]  Inspect pod logs:" >&2
        echo "        aliyun pai-dlc GetPodLogs --region ${REGION} --JobId ${JOB_ID} --PodId <pod>" >&2
        exit 4
    fi
    sleep 10
done
echo "[INFO] Log dir ready: ${LOG_DIR}"

WEB_SCRIPT="${REPO_ROOT}/benchmarks/web_control.py"
[[ -f "${WEB_SCRIPT}" ]] || { echo "[ERROR] web_control script missing: ${WEB_SCRIPT}" >&2; exit 5; }

echo "[INFO] Starting web console at http://${WEB_HOST}:${WEB_PORT}/  (Ctrl-C to stop)"
cd "${REPO_ROOT}"
exec "${WEB_PYTHON}" "${WEB_SCRIPT}" "${LOG_DIR}" \
    --benchmark robotwin \
    --host "${WEB_HOST}" \
    --port "${WEB_PORT}"
