# OpenWAM — XPolicyLab policy adapter (RoboDojo `arx_x5`, EE control, batch)

Serves the OpenWAM RoboDojo SFT checkpoint through XPolicyLab's submit-policy
protocol. The model runs **in-process** inside the policy server (no second
WebSocket hop): observations are converted world → robot-base → EEF20, one
`engine.generate_batch` forward pass serves every running env (true batch
inference), and the returned EEF20 chunks are converted back to env-relative
world `ee` action dicts.

## Files

| File | Role |
|---|---|
| `deploy.yml` | XPolicyLab policy config; `eval_batch: true`, `action_type: ee` |
| `model.py` | `Model(ModelTemplate)`: obs/action frame conversion + batched inference |
| `deploy.py` | Standard XPolicyLab episode loops (copied from the reference adapter) |
| `eval.sh` / `setup_eval_policy_server.sh` / `setup_eval_env_client.sh` | Launchers |
| `OpenWAM/` | Vendored OpenWAM source tree (X_WAM-style; weights NOT included) |

The vendored `OpenWAM/` is an rsync of the upstream repo excluding `models/`
and `.git/`. To refresh it after upstream changes:

```bash
# NOTE: '/models/' must stay anchored (leading slash) — the source tree has
# nested code dirs also named models/ (openwam/model/video_backbone/wan/...).
rsync -a --delete --exclude='/models/' --exclude='.git/' \
  --exclude='open_wam.egg-info/' --exclude='.pytest_cache/' --exclude='__pycache__/' \
  /mnt/xspark-data/yuechen/OpenWAM/  RoboDojo/XPolicyLab/policy/OpenWAM/OpenWAM/
```

OpenWAM deploy hyperparameters come from the vendored
`OpenWAM/configs/deploy.yaml`; `model.py` force-overrides the
correctness-critical keys at load time (see Contract summary), so no separate
pinned yaml exists.

## Contract summary

- **Control**: absolute end-effector (`action_type: ee`), dual-arm `dual_x5`.
  Action dicts per step: `left_ee_pose` (7: xyz + quat wxyz, env-relative
  world), `left_ee_joint_state` (1: gripper [0,1]), `right_*` likewise.
- **State in**: `state/left_ee_pose`, `state/right_ee_pose` (env-relative
  world) + grippers → `env_relative_world_to_robot_base` (dual-X5 base
  constants) → EEF20 `[L xyz3, L rot6d6, L grip1, R xyz3, R rot6d6, R grip1]`.
- **Images**: `cam_head` / `cam_left_wrist` / `cam_right_wrist` RGB →
  the checkpoint's multiview L-shape canvas (384x320), composed by OpenWAM's
  own `ObsPreprocessor` (identical to training).
- **Prompt**: `"A video recorded from a robot's point of view executing the
  following instruction: " + instruction` (byte-identical to training).
- **Batching**: `eval_batch: true`. `get_action_batch(env_idx_list)` stacks
  all running envs into ONE batched forward; returns one action chunk per env
  (32 steps by default; `replan_steps` in `deploy.yml` truncates).
- **Correctness settings** (re-forced in `model.py` even if yaml drifts):
  `dit_cache.enabled=false`, `compile.enabled=false`, `decode_video=false`,
  `inference_mode=sync`. Batch equivalence was verified on real weights
  (`OpenWAM/scripts/verify_batch_equivalence.py`: B=1 parity exact, no
  cross-sample contamination).

## Environment

- Policy env: `/mnt/xspark-data/miniconda3/envs/openwam`
  (needs `msgpack`, `msgpack-numpy`, `pydantic`, `websockets` for the XPolicyLab WS server).
- Client env: `/mnt/xspark-data/miniconda3/envs/RoboDojo` (Isaac).
- OpenWAM source: vendored `OpenWAM/` in this dir (override with
  `OPENWAM_ROOT=...`); `model.py` prepends it to `sys.path` so it wins over
  any editable install in the policy env.
- Checkpoint (kept outside the repo): `${OPENWAM_CKPT_ROOT}/<ckpt_name>`
  (default root `/mnt/xspark-data/yuechen/OpenWAM/models`) or
  `OPENWAM_CKPT_DIR=...` full-path override.

## Run

### 0. Offline coordinate-chain check (no GPU)

```bash
cd RoboDojo/XPolicyLab/policy/OpenWAM/OpenWAM/scripts
PYTHONPATH=/mnt/xspark-data/yuechen/RoboDojo \
  /mnt/xspark-data/miniconda3/envs/openwam/bin/python verify_frames_contract.py
# real-weights adapter check without WS/Isaac (GPU): verify_adapter_local.py
```

### 1. Debug protocol loop (no Isaac; dummy = no checkpoint load)

```bash
cd RoboDojo/XPolicyLab/policy/OpenWAM
EVAL_ENV_TYPE=debug OPENWAM_ALLOW_DUMMY_POLICY=true \
  bash eval.sh RoboDojo stack_bowls New_OpenWAM_RoboDojo_SFT arx_x5 ee 0 0 0 \
  /mnt/xspark-data/miniconda3/envs/openwam /mnt/xspark-data/miniconda3/envs/RoboDojo
# real weights through the same loop: drop OPENWAM_ALLOW_DUMMY_POLICY
```

Success = `[MAIN] eval finished`.

Arg order (10): `bench task ckpt_name env_cfg action_type seed policy_gpu
env_gpu policy_conda_env client_conda_env`.

### 2. Single Isaac task (stack_bowls first)

```bash
cd /mnt/xspark-data/yuechen/RoboDojo
bash scripts/robodojo.sh eval \
  --policy-dir XPolicyLab/policy/OpenWAM \
  --task stack_bowls \
  --ckpt New_OpenWAM_RoboDojo_SFT \
  --policy-env /mnt/xspark-data/miniconda3/envs/openwam \
  --seed 0
```

### 3. Official benchmark (54 tasks x 3 seeds)

```bash
cd /mnt/xspark-data/yuechen/RoboDojo
for seed in 0 1 2; do
  bash scripts/robodojo.sh benchmark \
    --policy-dir XPolicyLab/policy/OpenWAM \
    --ckpt New_OpenWAM_RoboDojo_SFT \
    --policy-env /mnt/xspark-data/miniconda3/envs/openwam \
    --seed ${seed} --gpu-ids 0,1,2,3,4,5,6,7
done
bash scripts/robodojo.sh summarize
```

## Notes / knobs

- First server start on cold JuiceFS cache is slow (torch import + 24.8 GB
  safetensors read). Pre-warm with
  `dd if=<ckpt>/checkpoint_step_36255.safetensors of=/dev/null bs=64M`
  if the client's 1200 s server-wait is at risk.
- `replan_steps` (deploy.yml): actions executed per chunk before replanning;
  `null` = full 32-step chunk (matches OpenWAM `inference_horizon: null`).
- `OPENWAM_DEPLOY_CONFIG=...` overrides the deploy hyperparameters yaml per
  launch (default: vendored `OpenWAM/configs/deploy.yaml`).
- The debug client sends placeholder `np.ones(7)` poses; the adapter
  normalizes non-unit quaternions (warning above 1e-3 deviation) instead of
  rejecting, matching the reference adapter.
