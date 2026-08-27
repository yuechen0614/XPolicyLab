"""Real-weights XPolicyLab-adapter verification without the WS server or Isaac.

Drives XPolicyLab.policy.OpenWAM.model.Model directly with Isaac-shaped
synthetic observations and checks the full batched protocol surface:

  1. update_obs_batch + get_action_batch([0,1,2]) -> one batched forward,
     list[list[dict]] chunks with the exact ee keys/shapes/dtypes.
  2. Env-shrink replan: get_action_batch([0,2]) after env 1 finished.
  3. Single-env path: get_action().
  4. Action sanity: unit quats, grippers in [0,1], finite positions in a
     plausible workspace, chunk length == num_frames - 1 (32).
  5. reset() clears stored obs; get_action_batch afterwards raises.

The adapter defaults to the OpenWAM source vendored at
RoboDojo/XPolicyLab/policy/OpenWAM/OpenWAM/. Checkpoint dir comes from
OPENWAM_CKPT_DIR (weights live outside the repo).

Run (GPU, ~2 min model load):
  PYTHONPATH=/mnt/xspark-data/yuechen/RoboDojo CUDA_VISIBLE_DEVICES=0 \
  OPENWAM_CKPT_DIR=/mnt/xspark-data/yuechen/OpenWAM/models/New_OpenWAM_RoboDojo_SFT \
  python verify_adapter_local.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

_DEFAULT_CKPT_DIR = "/mnt/xspark-data/yuechen/OpenWAM/models/New_OpenWAM_RoboDojo_SFT"


def _synthetic_isaac_obs(env_idx: int) -> dict:
    rng = np.random.default_rng(100 + env_idx)

    def _cam(shift: int) -> dict:
        h, w = 480, 640
        yy, xx = np.mgrid[0:h, 0:w]
        base = ((xx + yy + 41 * (env_idx + 1) + shift) % 256).astype(np.uint8)
        rgb = np.stack([base, np.roll(base, 11, axis=1), np.roll(base, 23, axis=0)], axis=-1)
        return {"color": rgb, "shape": (h, w)}

    quat = np.array([1.0, 0.0, 0.0, 0.0])
    left_pose = np.concatenate([np.array([-0.25, -0.10, 0.95]) + rng.normal(0, 0.02, 3), quat])
    right_pose = np.concatenate([np.array([0.25, -0.10, 0.95]) + rng.normal(0, 0.02, 3), quat])
    return {
        "env_idx": env_idx,
        "instruction": "stack the bowls on the plate",
        "vision": {
            "cam_head": _cam(0),
            "cam_left_wrist": _cam(64),
            "cam_right_wrist": _cam(128),
        },
        "state": {
            "left_arm_joint_state": np.zeros(6, dtype=np.float32),
            "right_arm_joint_state": np.zeros(6, dtype=np.float32),
            "left_ee_pose": left_pose.astype(np.float32),
            "right_ee_pose": right_pose.astype(np.float32),
            "left_ee_joint_state": np.array([float(env_idx % 2)], dtype=np.float32),
            "right_ee_joint_state": np.array([float((env_idx + 1) % 2)], dtype=np.float32),
        },
    }


def _check_chunk(chunk: list, label: str, failures: list) -> None:
    if not isinstance(chunk, list) or not chunk:
        failures.append(f"{label}: chunk must be a non-empty list, got {type(chunk).__name__}")
        return
    for t, action in enumerate(chunk):
        for key, dim in (
            ("left_ee_pose", 7),
            ("left_ee_joint_state", 1),
            ("right_ee_pose", 7),
            ("right_ee_joint_state", 1),
        ):
            value = np.asarray(action.get(key))
            if value.shape != (dim,):
                failures.append(f"{label}[{t}].{key}: shape {value.shape} != ({dim},)")
                continue
            if not np.all(np.isfinite(value)):
                failures.append(f"{label}[{t}].{key}: non-finite values")
        for side in ("left", "right"):
            pose = np.asarray(action[f"{side}_ee_pose"], dtype=np.float64)
            qn = float(np.linalg.norm(pose[3:7]))
            if abs(qn - 1.0) > 1e-4:
                failures.append(f"{label}[{t}].{side}_ee_pose: quat norm {qn:.6g}")
            if float(np.abs(pose[:3]).max()) > 3.0:
                failures.append(f"{label}[{t}].{side}_ee_pose: position out of range {pose[:3]}")
            grip = float(np.asarray(action[f"{side}_ee_joint_state"])[0])
            if not 0.0 <= grip <= 1.0:
                failures.append(f"{label}[{t}].{side}_ee_joint_state: {grip} outside [0,1]")


def main() -> int:
    from XPolicyLab.policy.OpenWAM.model import Model

    model = Model(
        {
            "action_type": "ee",
            "env_cfg_type": "arx_x5",
            "ckpt_dir": os.environ.get("OPENWAM_CKPT_DIR", _DEFAULT_CKPT_DIR),
            "device": "cuda",
        }
    )

    import openwam

    print(f"[openwam] imported from {openwam.__file__}")

    failures: list = []
    obs_list = [_synthetic_isaac_obs(i) for i in range(3)]

    # 1. full batch
    model.reset()
    model.update_obs_batch(obs_list)
    chunks = model.get_action_batch([0, 1, 2])
    print(f"[1] batch B=3: chunks={len(chunks)}, chunk_len={len(chunks[0])}")
    if len(chunks) != 3:
        failures.append(f"batch B=3 returned {len(chunks)} chunks")
    if len({len(c) for c in chunks}) != 1:
        failures.append(f"chunk lengths differ: {[len(c) for c in chunks]}")
    if len(chunks[0]) != 32:
        failures.append(f"chunk length {len(chunks[0])} != 32 (num_frames 33 - 1)")
    for i, chunk in enumerate(chunks):
        _check_chunk(chunk, f"chunk[{i}]", failures)

    # rows must differ across envs (inputs differ)
    a0 = np.asarray(chunks[0][0]["left_ee_pose"])
    a1 = np.asarray(chunks[1][0]["left_ee_pose"])
    if float(np.abs(a0 - a1).max()) < 1e-6:
        failures.append("env 0 and env 1 produced identical actions for different obs")

    # 2. env shrink: env 1 finished; batch loop refreshes survivors then replans
    model.update_obs_batch([obs_list[0], obs_list[2]])
    shrunk = model.get_action_batch([0, 2])
    print(f"[2] shrink B=2: chunks={len(shrunk)}, chunk_len={len(shrunk[0])}")
    if len(shrunk) != 2:
        failures.append(f"shrunk batch returned {len(shrunk)} chunks")
    for i, chunk in enumerate(shrunk):
        _check_chunk(chunk, f"shrunk[{i}]", failures)

    # 3. single-env protocol path
    model.update_obs(obs_list[0])
    single = model.get_action()
    print(f"[3] single: chunk_len={len(single)}")
    _check_chunk(single, "single", failures)

    # 4. reset clears stored obs
    model.reset()
    try:
        model.get_action_batch([0])
        failures.append("get_action_batch after reset() should raise, but returned")
    except ValueError:
        print("[4] reset -> get_action_batch correctly raises")

    if failures:
        print("ADAPTER LOCAL VERIFICATION FAILED:")
        for f in failures:
            print("  -", f)
        return 1
    print("ADAPTER LOCAL VERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
