# Active-mixture EEF pose-frame contract

## Decision

The real-robot pose slots in the shared 80-D action/proprio space mean:

> A terminal-arm frame rigidly attached to the arm chain and independent of
> gripper/finger articulation, expressed in the dataset's robot-base frame.

The short machine-readable name is
`rigid_terminal_arm_frame_independent_of_gripper_motion` (see
`openwam.dataloader.utils.eef.EEF_POSE_FRAME_CONTRACT`).

This is deliberately not called a universal literal `flange_pose` or
`tcp_pose`. Across embodiments the published terminal endpoint can be a flange,
wrist-yaw link, hand base, last arm link, gripper mount, or a fixed tool frame.
The contract excludes a point whose transform changes with gripper
articulation, but it does not claim that every fixed point is upstream of every
possible static TCP. It aligns the endpoint *category*; it does not assert that
different robots share the same local origin, local-axis convention, or a
common base origin. Achieving that stronger equivalence would require
per-embodiment calibrated rigid transforms that several source releases do not
publish.

`worldengine` is not covered by this real-robot endpoint audit.

## RoboDojo `arx_x5`

RoboDojo is supported only for the dual-arm `arx_x5` embodiment. Its HDF5 and
live-observation source pose is `[xyz, quaternion wxyz]`: `xyz` is world
translation with `scene.env_origins[env_idx]` subtracted, while the quaternion
is still a world-frame orientation. Treating that mixed source pose as already
robot-base-relative is incorrect.

OpenWAM converts that mixed source pose with the dual-X5 base constants from
RoboDojo ``env_cfg/robot/dual_x5.yml``, stored in
``openwam/dataloader/robodojo_contract.py``. The shared transform in
``openwam/dataloader/utils/poses.py`` then expresses both
position and orientation in the corresponding left or right robot-base frame.
Isaac eval uses pinned copies under ``benchmarks/robodojo/``
(``robodojo-eef20-v1``) and must not import OpenWAM.
The canonical raw model layout is:

`[L xyz3, L rot6d6, L grip1, R xyz3, R rot6d6, R grip1]`.

For both X5 arms, the represented rigid terminal endpoint is `link6`. It is
independent of gripper/finger articulation and therefore satisfies the
terminal-arm contract above; no gripper-center or task-TCP offset is applied.

## Active datasets

| Dataset | Reader source used for pose | Physical endpoint | Decision |
|---|---|---|---|
| AgiBotWorld-Beta | `action.ee_base`, `observation.state.ee_base` | Axis-7 flange / arm end | Keep. This is the only release-proven endpoint available across its gripper and dex-hand buckets. |
| InternData-A1 | `*.ee_to_robot_pose` | Per-model EE attachment/controller link (hand base, last arm link, or gripper mount) | Keep. Do not use the downstream native `*.tcp_to_robot_pose`. |
| RoboCOIN | `eef_sim_pose_action`, `eef_sim_pose_state` | Published simulation FK terminal point; exact link varies by robot | Keep under the broad terminal-arm contract. Do not add a blanket TCP offset. |
| DROID | `other_information.action_wrist_pose`; `other_information.observation_gripper_pose6d` + `state[6]` | Commanded wrist and achieved rigid gripper-mount frame | Switched from the moving task-TCP streams. Stats contract v2 rejects the old TCP stats. |

## AgiBotWorld-Beta evidence

The official Beta schema calls `state/end/*` the robot flange and gives actions
the same semantics. Independent FK against the release G1 model over 20,000
real poses reproduced `ee_base` with position median error 5.20 µm, p99
11.24 µm, maximum 36.38 µm, and maximum rotation error 0.000409 rad.

No task-TCP column or release calibration is stored in the local dataset. Later
simulation assets contain end-effector-specific transforms (and the 19
dexterous-hand buckets have no universal single TCP), so applying one guessed
flange-to-TCP offset to all buckets would be incorrect.

Primary schema: [AgiBotWorld-Beta README](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta/blob/main/README.md#L357-L400).

## InternData-A1 evidence

All 244 clean buckets and all 2,648 parquet shards were scanned. The release
contains both EE and TCP columns in every bucket. Across 1,102,231,132
action/state arm-stream rows:

- EE and TCP quaternions are bit-identical.
- The maximum reconstructed world-position residual is 1.7401e-7 m.
- TCP is a fixed local translation from the selected EE link, with no
  exceptional bucket.

| Variant | Clean buckets | Current EE link | Release EE-to-TCP translation |
|---|---:|---|---:|
| Franka + Panda Hand | 13 | `panda_hand` (hand base, not literal flange) | local +z 0.095 m |
| Franka + Robotiq | 13 | `panda_link8` (end-of-arm/flange) | local +z 0.145 m |
| ARX Lift-2 | 127 | `link6` (last arm link / gripper base) | local +x 0.16157 m |
| Genie-1 | 7 | `arm_{l,r}_end_link` (last arm link / gripper mount) | local +z 0.22 m |
| Split Aloha/Piper | 84 | `link6` (last arm link / gripper base) | local +z 0.135 m |

The `*_ee_to_robot_pose` columns are retained because they are upstream of the
task TCP and are the simulator-controlled endpoint. `*_to_robot_pose` is also
retained instead of per-arm-base fields so both arms share one base frame.

Release converters: [Panda](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_franka_a1.py#L315-L421),
[Franka+Robotiq](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_frankarobotiq_a1.py#L311-L420),
[Lift-2](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_lift2_a1.py#L382-L487),
[Genie-1](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_genie1_a1.py#L384-L505), and
[Split Aloha](https://github.com/InternRobotics/InternDataEngine/blob/2a0a21f2c836df97c925729084e13d68950b4deb/policy/lmdb2lerobotv21/lmdb2lerobot_split_aloha_a1.py#L382-L487).

## RoboCOIN evidence and limits

RoboCOIN guarantees unified base axes/origins for `eef_sim_pose_*`, but its
public documentation does not identify one common target link and the FK field
generator is not published. Real parquet was therefore checked against public
robot models where possible.

Evidence levels: **A** = real parquet FK matches an official model and known TCP
candidates were rejected; **B** = only a subset of the robot type reaches A;
**C** = pose is rigid in arm joints / independent of moving fingers, but the
exact terminal link cannot be proved from public release artifacts.

| `robot_type` | Buckets | Best supported endpoint | Evidence |
|---|---:|---|---|
| `agilex_magic` | 72 | Piper `link6 == gripper_base` | A |
| `agilex_magic_decoupled` | 57 | Piper `link6 == gripper_base` | A |
| `aloha` | 9 | Piper `link6 == gripper_base` | A |
| `ai2_alphabot2` | 29 | Exact link not public | C |
| `alphabot2` | 10 | Exact link not public | C |
| `airbot_mmk2` | 142 | AIRBOT Play v3 `link6` | A |
| `discover_mmk2` | 70 | AIRBOT Play v3 `link6` | A |
| `g1dex3` | 7 | Unitree G1 `wrist_yaw_link` | A |
| `g1ego` | 35 | Unitree G1 `wrist_yaw_link` | A |
| `g1high` | 5 | Unitree G1 `wrist_yaw_link` | A |
| `galaxea_r1_lite` | 110 | Exact link/static-TCP choice not provable | C |
| `leju` | 49 | Exact link not public | C |
| `realman_rmc_aidal` | 35 | RM75B `link7` (jaw base is downstream) | A- |
| `ruantong_a2d` | 34 | 31 AgiBot-G1 buckets use `arm_end_link == gripper_base`; 3 Tianqin buckets unknown | B |
| `yinhe` | 13 | Exact link not public | C |

No type was proven to store a moving grasp-center TCP. The exact-link-unknown
types prevent a safe conversion of all RoboCOIN data to task TCP; such a change
would require per-embodiment (and for `ruantong_a2d`, per-family) calibration.

Primary definition and models: [RoboCOIN](https://github.com/FlagOpen/RoboCOIN/blob/b3261fe7cf92d18d9f8545c4b8ad9813dd1d2edd/README.md#eef_sim_pose-state--eef_sim_pose-action),
[Piper](https://github.com/agilexrobotics/piper_ros/blob/ac41fcbcdda598f01b51cf6175ed9a24d0dacadc/src/piper_description/urdf/piper_description.urdf),
[Unitree G1](https://github.com/unitreerobotics/xr_teleoperate/blob/845b25a32f7febedf220e830952a7134897adb9d/assets/g1/g1_body29_hand14.urdf),
[AIRBOT Play v3](https://github.com/TATP-233/DISCOVERSE/blob/d67f47c084aba0e0cf422a8725235f8b9238655a/models/urdf/airbot_play_v3_gripper.urdf),
[RealMan RM75B/AIDA](https://github.com/RealManRobot/URDF-to-XACRO/blob/ccacc05c1cf8fe5adf05c5f1de5d53b85f286558/rm_Lifting_robot_75B_jaw_description.zip), and
[AgiBot G1](https://huggingface.co/datasets/agibot-world/GenieSimAssets/blob/1eb3b68b740f87fb369b0146ee53f3bb3da6b0d0/G1_omnipicker/G1_omnipicker.urdf).

## DROID evidence and implementation

DROID's re-converted rows contain physically distinct wrist/mount and task-TCP
streams. Across the first 256 rows of all 398 shards (101,888 rows):

- action TCP and wrist rotations are identical;
- achieved TCP and gripper-mount rotations are identical;
- TCP minus wrist/mount lies on local +z;
- the distance is 0.14388646--0.15718069 m and varies with gripper state.

That variable offset proves the TCP is not one fixed flange transform. The
reader now uses `action_wrist_pose` and
`observation_gripper_pose6d + state[6]`. A full-shard lag sweep places the
command/observation position-error minimum at about four 10-Hz frames, verifying
that the wrist action remains a forward command.

Normalization was recomputed over the exact train population after the 426
prompt exclusions: 46,101,178 state rows plus 46,101,178 action rows. The v2
stats contract records the pose-frame semantics and exact source columns;
rot6d statistics remain identity-pinned.

Converter schema: [RoboInter DROID converter](https://github.com/InternRobotics/RoboInter/blob/main/RoboInterData/convert_to_lerobot/convert_droid_to_lerobot_anno_fast.py).
