# RoboCasa365 — unify_action 语义重构：base 并入 raw 向量（action/proprio 对称 25-D）

> 状态：**已实现，全部决策已锁定（2026-07-11）**。触发来源 = wayrise 2026-07-10 15:16 的设计层复审
> （取代其 11:30 的 approve，转 CHANGES_REQUESTED）。用户 2026-07-11 独立提出了同一目标形态。
> 本文档给出：目标形态、与 starVLA/BEHAVIOR 的对照结论、逐文件改动、最终决策。§3/§6 里"决策区/待拍板"
> 均已在 §7 拍定并实现；§8 记录复审后 code-review 又挑出、经评估决定"本 PR 不改"的两个既有质量项。

---

## 0. 触发与背景

- **wayrise 复审要求（PR #5，id=4672706587）**：
  1. 对照 **starVLA** 的 robocasa365 state/action 定义，说明取舍。
  2. **unify_action 语义重构**：base 并入 raw 向量，`unify_action=false` 就是完整原始维度，`unify_action=true`
     用**一张** `unify_action_map` 把整个 raw 向量映射进 80-D；base 不再是 map 之外的旁路
     （当前 = 硬编码 `_UNIFY_BASE` 直写 `[68:73)` + 独立 `'base'`/`'base_vel'` stats 块 + deploy 侧单独 gather/append）。
  3. 连带：删掉 `model_loader.py` 的 `base_slice/base_normalizer/base_vel_dst/base_vel_normalizer` 扩展，
     deploy 回到通用 `_UnifyAwareNormalizer`（gather 回 raw 宽度 + 单一 stats 反归一化），**本 PR 不再动共享 deploy 代码**。
  4. stats 从 `{eef 20 + base 5 + base_vel 3}` 三块收敛为**单一 raw 宽度**向量；`'eef'`=20-D 是仓库级 schema
     （`materialize_eef_stats`/robotwin/OXE 共用），**新宽度用新 action_mode 键，不要改 `'eef'` 语义**。
  5. **解耦** `mobile_base` 与 `unify_action`（当前 mobile 必须 unify，否则 raise）。
  6. state/proprio 侧 base 命令与可观测量的不对称如何组织，以 starVLA 为对照、BEHAVIOR 为参照
     （proprio 渲染进 action 命令空间、共用同一 map 同一 stats、无 achieved 值的维度 mask 掉）。
  7. 时机：PR 未合、无正式 ckpt，现在是改表示层/ stats schema 的窗口期。

- **仓库内先例 = `openwam/dataloader/behavior.py`（BEHAVIOR-1K，RAW-27）**：raw 向量本身 =
  `[L_pos3,L_rot6d6,L_grip1, R_pos3,R_rot6d6,R_grip1, base3, trunk4]`，一张 map
  `["0-9","34-43","68-70","71-74"]`，一份 combined stats，deploy 侧零特判。**这是我们要复制的模板。**

---

## 1. 对照 starVLA（wayrise 要点 1 的结论）

starVLA `examples/Robocasa_365/`（Franka PandaOmron 单臂移动，即"直接 robocasa365 支持"）：

| | starVLA | 本 repo（现状 & 重构后） |
|---|---|---|
| action | **12-D**：eef_pos3(**delta**) + eef_rot3(**axis-angle delta**) + grip1 + base_motion4 + control_mode1 | **80-D unify**（raw 25-D）：eef **full base-relative pose** + **rot6d** + base5 |
| state | **16-D**：base_pos3 + base_rot4(quat) + eef_pos_rel3 + eef_rot_rel4(quat) + gripper_qpos2 | 同一 16-D 原始 state（我们从中派生 arm10 + base 速度） |
| arm 表示 | per-step **delta**，axis-angle | **full pose**（非 delta），**rot6d**（eval 侧 bridge 成 OSC delta） |
| base | action 里 base_motion4；state 里 **absolute base pose**（非速度） | action=raw 命令 base5；proprio=body-frame **速度**（scene-invariant） |
| action vs state | 布局不同、**stats 分开**；state 走 **sin/cos**（不 min-max） | 目标：对称 25-D、**一张 map**；stats 见 §4/决策 |
| control_mode | action 第 11 维，min-max（无二值特判） | action base5 第 5 维，**保持被预测**（用户锁定，勿删） |

**结论（要写进 PR）**：我们与 starVLA 有**有意分歧**——full base-relative pose vs delta、rot6d vs axis-angle、
unified-80D 跨机器人头 vs GR00T 每-embodiment 头、proprio 放 body-frame 速度 vs state 放 absolute base pose。
理由：本模型是共享 action 头的 WAM，预测 full base-relative EEF 轨迹并在 eval 侧 bridge 成官方 OSC delta env
（保 leaderboard 可比性）。starVLA 的关键"可借鉴"点其实**不适用**（它 action/state 本就分开、不共用 stats、
state 不归一化）——真正的模板是仓库内 BEHAVIOR。

---

## 2. 目标形态（firm，无争议部分）

### 2.1 raw 25-D 向量（action 与 proprio 同布局）

```
[0:10)  left  eef  = eef_pos3 + rot6d6 + grip1     (full base-relative pose)
[10:20) right eef  = 全 0（单臂零填充，mask 掉）
[20:25) base 5     = [x_vel, y_vel, yaw_vel, torso/slot3, control_mode/slot4]
```

- `unify_action=false` → 直接输出这个 **25-D**（mobile）/ **20-D**（fixed-base，无 base 段）。
- `unify_action=true` → 用**一张** map 整体映射进 80-D：

```yaml
unify_action_map: ["0-9", "34-43", "68-72"]
#   raw[0:10)  -> unified[0:10)    左臂
#   raw[10:20) -> unified[34:44)   右臂（零填充+mask）
#   raw[20:25) -> unified[68:73)   base
```

`parse_unify_spec` 已支持多段（BEHAVIOR 就是 4 段），机制上 `["0-9","34-43","68-72"]`（dst_index 长 25）直接成立。

### 2.2 mask

- **action_mask**：valid = 左臂[0:10) + base[20:25)（**5 个 base 维全监督，含 control_mode**）；右臂[10:20) mask。
- **proprio_mask**：valid = 左臂[0:10) + base 速度[20:23)（见决策 1）；base slot3/slot4 见决策；右臂 mask。

### 2.3 stats & deploy（回到通用）

- 训练侧算**一份 25-D combined stats**（arm 20 子块 + base 5 子块拼接），持久化到
  `normalization_stats.npy` 的**新 action_mode 键**（如 `"eef_base"`；**不动 `'eef'`=20-D 语义**）。
- deploy `_UnifyAwareNormalizer` **回到通用**：action OUT = gather 80→25 + 整体 unnormalize；
  proprio IN = 整体 normalize + scatter 25→80。**删掉** `base_slice/base_normalizer/base_vel_dst/base_vel_normalizer`
  与对 `openwam.dataloader.robocasa365._UNIFY_BASE/_UNIFY_BASE_VEL` 的 import。
- **客户端协议不变**：server 反归一化后仍返回 raw `[arm20, base5]`（现在是"整体 gather"而非"arm+base 分别 gather"），
  eval bridge（arm20→OSC 12-D、base5 直通 env）无改动。

### 2.4 解耦 mobile_base / unify_action

- `mobile_base=false` → raw = 20-D 纯臂（fixed-base）；`mobile_base=true` → raw = 25-D。
- `unify_action=false` → 模型头直接吃 raw 宽度（20 或 25）；`true` → 80-D。二者正交，不再 raise 强耦合。

---

## 3. 与 action 对称的 proprio base 段（**决策区**）

实测（`OpenDrawer_eefbasevel_stats.npy`）：`base`（action 命令）范围 ≈ **[0,1]**，
`base_vel`（proprio 有限差分）范围 ≈ **[±0.002]** —— **~500× 尺度差**。二者是不同物理量
（归一化命令 vs 每帧位移），**不能天真共用一份 stats**。这带来两个必须先定的点：

### 决策 1 — proprio 的 base 速度（slots 20-22）怎么归一化？

| 选项 | 做法 | 满足 wayrise「一份 stats+零特判」 | 保留速度信号 | 代价 |
|---|---|---|---|---|
| **A′（推荐）** | 把 observed 速度**渲染进 action 命令空间**（÷ robocasa base 控制器 output scale × fps），与 action 命令共用同一份 stats。**完全对齐 BEHAVIOR**（`_BASE_VEL_OUTPUT_SCALE`） | ✅ | ✅ | 需从 robocasa 源码/数据定一个 scale 常量；eval client 也要同样 rescale |
| **C（最简）** | proprio 的整个 base 段 mask 掉（不放速度）。action 仍监督 base5 | ✅（速度维 mask，值无关） | ❌ 丢掉 proprio 速度 | 与你 spec「slots 0-2 放速度」冲突 |
| **B（折中）** | 布局+map 统一，但 base 速度维保留独立 stats（action 用命令 stats、proprio 用速度 stats） | ⚠️ 部分（deploy 仍需一点 base 特判） | ✅ | 没完全满足 wayrise，等于保留现状的一半特判 |

> 说明：若「共用一份 stats 但不 rescale」（A 无 rescale），proprio 速度被 action 命令 stats 压成 ~0 附近的一条缝，
> 信号几乎废掉——是「最差组合」（有 plumbing 无信号），不单列。

### 决策 2 — proprio 的 slot 3（对应 action 的 torso 命令）放什么？

- **mask 掉（推荐）**：`observation.state` 16-D **没有 torso qpos**；你 spec 里的 `z_pos`（base_position[2]）
  是**移动底盘原点的世界系 z**，≈ 常量、且与 torso 升降高度**是两个量**，也无法与 torso 命令共用 stats。
  → proprio base = `[vx, vy, vyaw, MASK, MASK]`。
- **放 base_z（你的原 spec）**：可行但它无法与 torso 命令共用一份 stats；若共用则被 torso 的 [0,0.34] 归一化成垃圾值，
  只能再 mask（那就等于没放）。除非给它独立 stats（又回到 B 的特判）。

> slot 4（control_mode）proprio 侧统一 **mask**（无 achieved 值，即你说的「留空」）。

---

## 4. 逐文件改动清单

1. **`openwam/dataloader/robocasa365.py`**（主战场）
   - 组装 raw 25-D（arm20 + base5）用于 **action 和 proprio 两侧**；删除 `_UNIFY_BASE`/`_UNIFY_BASE_VEL` 的
     post-unify 注入，改成"先拼 raw 再一张 map scatter"。
   - `unify_action_map` 解析为 25 段 dst_index；`action_dim` = 25（raw, unify off）/ 80（on）。
   - 25-D raw dim mask（arm 左 + base）；action/proprio 两套 mask 按 §2.2 + 决策。
   - `mobile_base` 与 `unify_action` 解耦（去掉互相 raise）。
   - stats：算/存**一份 25-D combined**（新键 `eef_base`）；`denormalize_action` 简化为 gather 80→25 + unnormalize 25。
   - 决策 1=A′ 时：`_base_velocity_body` 后接 rescale 到命令空间。
   - `_resolve_stats_path` / `_resolve_shared_stats` 后缀简化（mobile → `_eefbase_`；去掉 `vel` 变体）。
2. **`openwam/dataloader/robocasa365_stats_computation.py`**：输出 combined 25-D（新键）；CLI/文档同步。
3. **`openwam/deploy/model_loader.py`**：**删** `_UnifyAwareNormalizer` 的 base_slice/base_vel 四个参数与
   `_build_normalizer` 里读 `mobile_base`/`base_proprio_velocity` + import robocasa365 的两段；deploy 回到通用。
4. **`benchmarks/robocasa365/openwam2robocasa365_interface.py`**：proprio 发 25-D `[arm20, base5_proprio]`
   （A′ 时对速度做同样 rescale）；action bridge 不变。
5. **`benchmarks/robocasa365/single_eval.py` / `policy_config.yml`**：flag/键同步。
6. **`configs/dataloader/robocasa365.yaml`**：`unify_action_map: ["0-9","34-43","68-72"]`；新 action_mode 键；
   mobile_base/base_proprio_velocity 语义与解耦说明；docstring。
7. **`benchmarks/utils/action_conversion.py`**：A′ 时加 base 速度 rescale helper；`eef20d_to_robocasa12d` 不变。
8. **测试**：`tests/test_action_normalization.py` + `tests/dataloader/test_robocasa365.py` 的 mobile/base_vel 系列
   **按新对称形态重写**（整体 25-D round-trip、25-D stats、缺块 raise、宽度校验）；覆盖面不缩水（TDD：先写红）。
9. **module docstring / CHANGELOG.md**：更新表示层描述与已知 Gap。

---

## 5. 需要源码核实的常量（决策 1 选 A′ 时）

- robocasa 移动底盘控制器的 base velocity **output scale / max**（BEHAVIOR 的 `_BASE_VEL_OUTPUT_SCALE` 类比）。
- 数据集 fps / 录制 dt（把每帧位移 → 速度 → 命令空间）。
- 来源：`/home/l/projects/robocasa`、`/home/l/projects/robosuite` 的 PandaOmron base controller 配置 + v3 `meta/info.json` 的 fps。

---

## 6. 待用户拍板 → 然后我才动代码

1. 决策 1（proprio base 速度：A′ / C / B）。
2. 决策 2（proprio slot 3：mask / base_z）。
3. 次要（我可默认）：新 action_mode 键名（建议 `eef_base`）；是否保留 `base_proprio_velocity` 作为
   "proprio 速度 populate vs 全 mask" 的开关（等价于决策 1 的 A′ vs C 可用一个 flag 表达）。

决策定了我会：先补/改红测试 → 改 dataloader → 改 deploy（删特判）→ 改 eval interface → 跑 e2e smoke → 更新文档/CHANGELOG → push。

---

## 7. 已实现（2026-07-11，决策已锁）

用户拍板：**决策1 = A′**（proprio 速度换算进命令空间，per-axis `_BASE_VEL_PHYS_MAX=[0.75,0.88,1.33]`，与 action 共用一份 stats）；**决策2 = mask slot23**（torso 全数据集 29.1M 帧恒 0、base_z 是受力回弹噪声）；**决策3 = 保 5-D + 退化-stats 守卫**（torso 不砍维，面向 composite 未来）。`base_proprio_velocity` flag 移除（能力内含于 mobile）。

新增支撑数据（远程只读 composite 全 77 shard + 本地 atomic 全量）：**torso action 在整个 RoboCasa365（atomic 1.50M + composite 27.6M = 29.1M 帧）恒 0，零例外**；composite base_z 部分任务轨迹内动 ≤4.7cm（MixCakeFrosting 等受力回弹，与 torso 无关）。

改动文件：`openwam/dataloader/robocasa365.py`（raw25、一张 map、combined `eef_base` stats、`base_velocity_cmd` A′、解耦、删 `_UNIFY_BASE`/`_base_proprio_vel`）、`robocasa365_stats_computation.py`（combined 25-D、删 base_vel）、`openwam/deploy/model_loader.py`（`_UnifyAwareNormalizer`/`_build_normalizer` 删 base 特判、删 `_build_key_normalizer`）、`benchmarks/utils/action_conversion.py`（+`base_velocity_cmd`）、`benchmarks/robocasa365/{openwam2robocasa365_interface,single_eval}.py` + `policy_config.yml`（`mobile_base` 25-D proprio）、`configs/dataloader/robocasa365.yaml`（3-token map、`action_mode: eef_base`）、测试全套重写、CHANGELOG。

验收：dataloader 40 + bench 76 passed（robocasa365 env）；deploy 7/7 standalone（openwam-kn）；client↔dataloader `base_velocity_cmd` 逐字节一致；真数据 OpenDrawer mobile 80-D + denorm→25-D 往返一致；ruff clean；e2e train（mobile 25-D→80-D，OpenDrawer 100 step）+ deploy + eval smoke。

后续 commit（同一分支，均 CI 三绿 + e2e 复验）：
- `a46cfea` **mask_torso_action**（默认 true）：torso（raw idx 23）从 action loss mask 掉、eval 侧 bridge 强制置 0。torso 是活关节但全数据集恒 0，预测非 0 会误驱动。两侧 flag 一致，`false` 恢复预测。
- `f3ea64b`/`6f39cf5`/`4f56a3d` **gripper 表示层重做**（回应要点 1 的"gripper 语义"）：action 第 9 维改用**录制指令** `action.gripper_close`（{-1,+1}），而非旧的 achieved 宽度（旧值滞后指令 ~1 帧 + 7 帧渐变，corr −0.916 → 晚闭 5-6 帧掉 SR）；proprio 第 9 维 = achieved 宽度渲染到 [-1,+1]（端点 `_GRIPPER_WIDTH_OPEN=0.1`）；gripper stats **钉死 [-1,+1]**（模型 ±1 反归一化正好 ±1）；bridge 闭合阈值 **0.5**（`>0.5→闭`，不确定输出默认开、避免误抓）。
- `80514b7` **fps 断言**：`fps` 缺则 raise、mobile 时断言 `== DATASET_FPS(20)`（消除静默 fallback；A′ 换算依赖 fps）。

---

## 8. 复审后 code-review findings（评估后：本 PR 不改，理由如下）

独立 code-review 在重构后又扫了一遍。除已修的 fps 静默 fallback（`80514b7`）外，挑出两个**既有（pre-existing）、非 train↔eval bug** 的质量项。经数据核实，均**决定不在本 PR 修**：

- **rot6d 未 pin identity**（stats 归一化）。`robocasa365_stats_computation.py` 未调 `pin_rot6d_identity`，rot6d 6 维被逐维仿射缩放（实测 OpenDrawer：min-max 系数 1.0–1.2、z-score std 0.36–0.83），破坏"两个单位向量"的耦合、给旋转 loss 不均等权重 → 旋转学习质量损失。但：①两侧一致、round-trip 精确、bridge 端 GS 兜底，**不是 train/eval 分叉**；②**同款遗漏 robotwin 也有**（robotwin 用 `Normalizer` 类同样逐维缩放、stats 无 pin）——robocasa 正是"Mirrors robotwin"抄来的；③修法一行（`pin_rot6d_identity(eef20, ROT6D_DIMS_EEF20)`）但要重训。**决定**：作为跨-reader 的 repo 级 gap 单独 PR 处理（连 robotwin 一起），不在本 benchmark PR 扩大范围。

- **is_static 不含 base 速度**（训练重采样）。静止窗口过滤只看臂位姿，理论上"臂停 + 底盘导航"的窗口会被降权。但实测：该过滤在 robocasa365 上触发率 **~0%**（OpenDrawer/CloseDrawer 0.0%、TurnOnStove 0.1%——归一化空间 1e-5 阈值下相邻帧微动几乎永远超阈），等于抄自 robotwin 的一个在这套数据上**不触发的机制**。**决定**：保持现状（与 robotwin/agibotworld 对齐、廉价、无害），不打补丁也不删。
