# RoboCasa365 全量 + 移动底盘支持 — 实现 Plan

**Date:** 2026-07-04
**Repo:** knightnemo/OpenWAM, branch `feat/benchmark-robocasa365`
**Status:** 设计已定,待实现

---

## 1. 目标 & 范围

从"fixed-base 单臂子集"扩到 **完整 RoboCasa365(全 365 任务,含 248 个 mobile 任务)**,让策略能命令移动底盘(base velocity + torso),而不再硬填 `base_motion=0`。

- **训练**:全 365 pretrain 任务(不再筛 fixed-base)。
- **Eval**:官方 50 target 任务(18 atomic-seen + 16 composite-seen + 16 composite-unseen),和 leaderboard 同口径。
- **模型头**:复用统一 80-D 头,底盘塞进 reserved `[68:80)`。

---

## 2. 已锁定的设计决策(来自采访)

| # | 决策 | 结论 |
|---|---|---|
| A | 底盘动作表示 | **用 RoboCasa 原生**:action = 原生速度(x/y/yaw)+ torso 位置 + control_mode,**直发 env、不桥接**。不转绝对位姿、不做速度桥接。 |
| B | 手臂表示 | **保持绝对 EEF 位姿**(从 state 重建),eval 时桥接成 OSC-delta。共享头要求手臂在 `[0:9]` 必须绝对。 |
| C | eef 观测帧 | **保持 base-相对,不转世界绝对**。手臂 OSC 控制器是 base 系,转世界会耦合底盘+需 world→base 变换。 |
| D | base proprio 帧 | **世界绝对**(导航)。→ 混合系:手臂 base 系、底盘 world 系(移动操作标准解耦)。 |
| E | control_mode | 从录制 action 读入,当一维放 reserved;eval 阈值 0.5 二值化(gym_wrapper 已这么做)。数据里 {-1,+1} 都出现,是真信号。 |
| F | fixed-base 筛选 | **彻底移除**(fixed_base_tasks.json / _assert_fixed_base / moma 过滤全删)。 |
| G | base 朝向表示(proprio) | 原生四元数(4 维)。可选 rot6d,纯口味,先用四元数。 |

---

## 3. 背景:三套 action/state 定义(务必别混)

- **LeRobot-12D action**(训练布局 B):`[base x/y/yaw vel(3) | torso pos(1) | control_mode(1) | eef Δpos(3) | eef Δrot(3) | gripper(1)]`。相对/速度命令。
- **RoboCasa gym env 顺序**(`ACTION_SLICES`,env.step 吃的):`[eef pos(3) | eef rot(3) | grip(1) | base_motion(4) | control_mode(1)]`。同 12 数、不同 index 顺序。
- **我们的 80-D**:绝对 EEF 位姿(0-9 左臂)+ hand + reserved。当前 robocasa365 只用手臂、从 `observation.state` 重建(不读 action 字段)。

关键差异:RoboCasa 原生 = 相对/速度;我们 80-D 手臂 = 绝对位姿。**底盘用原生、手臂用绝对**。

RoboCasa **16-D state**:`base_position(0:3 world) | base_rotation(3:7 quat world) | eef_pos_rel(7:10 base系) | eef_rot_rel(10:14 quat base系) | gripper_qpos(14:16)`。**注意:torso 高度、control_mode 不在 state 里** → proprio 侧给不了(masked)。

---

## 4. 目标 80-D 布局(最终)

### Action(80-D)

| 80-D 槽 | 内容 | 来源 | eval 处理 |
|---|---|---|---|
| `[0:3)` | 左 eef xyz(绝对) | 从 state 重建 | 桥接→OSC Δpos |
| `[3:9)` | 左 eef rot6d(绝对) | 从 state 重建 | 桥接→OSC Δrot |
| `[9]` | 左 gripper | 从 state 重建 | 直发 |
| `[10:34)` | 左手 | — | 单臂用不到,masked |
| `[34:44)` | 右 eef+grip | — | 单臂用不到,masked |
| `[44:68)` | 右手 | — | masked |
| **`[68]`** | **base x 速度** | **读 LeRobot action[0]** | **原生直发** |
| **`[69]`** | **base y 速度** | action[1] | 直发 |
| **`[70]`** | **base yaw 速度** | action[2] | 直发 |
| **`[71]`** | **torso 位置** | action[3] | 直发 |
| **`[72]`** | **control_mode** | action[4] | 阈值 0.5→±1 直发 |
| `[73:80)` | reserved(7 空) | — | masked |

### Proprio(80-D)

| 80-D 槽 | 内容 | 来源 |
|---|---|---|
| `[0:3)` | 左 eef xyz(base 系,绝对) | state eef_pos_rel |
| `[3:9)` | 左 eef rot6d(base 系) | state eef_rot_rel(quat→rot6d) |
| `[9]` | 左 gripper 指间距 | state gripper_qpos[0]-[1] |
| `[10:68)` | 左手/右臂/右手 | masked(单臂) |
| **`[73]`** | **base x 位置(world)** | **state base_position[0]** |
| **`[74]`** | **base y 位置(world)** | state base_position[1] |
| **`[75:79)`** | **base 四元数(world)** | state base_rotation[3:7] |
| 其余 reserved | masked | — |

**注意 action 与 proprio 的 base 用不同 reserved 槽**(action `[68:73)` = 速度/torso/mode;proprio `[73:79)` = 位置/朝向)。原因:底盘 action 是速度、proprio 是位姿,量纲和维度都不同(yaw:action 1 维速度 vs proprio 需 4 维四元数),**无法同槽对齐**,故分开放。手臂 `[0:9]` proprio/action 仍同槽(都绝对位姿)。base_position 的 z(≈0.7 恒定)丢弃;torso/control_mode 不在 state → proprio 无。

---

## 5. 中间"raw"向量(scatter 前)

为复用 `unify_action` 机制,定义每侧的 raw 源向量 + map:

- **raw action = 25-D**:`[左臂10(绝对) | 右臂10(0) | base5(x/y/yaw vel, torso, mode)]`
  `unify_action_map(action) = ["0-9", "34-43", "68-72"]`(左臂→0-9,右臂→34-43,base→68-72)
- **raw proprio = 26-D**:`[左臂10 | 右臂10(0) | base_pos2 + base_quat4]`
  `unify_action_map(proprio) = ["0-9", "34-43", "73-78"]`(base pos/quat→73-78)
- **mask**:左臂 valid、右臂 masked、base 段 valid。用 `LEFT_ARM_DIM_MASK`(手臂)+ base 段全 True 拼出 `_unify_dim_mask`(action / proprio 各一份,因 base 槽不同)。

> 因 action/proprio 的 map 不同,`RoboCasa365Dataset` 里要维护**两个** dst_index / dim_mask(现有代码只有一个,共用于 action+proprio;要拆开)。

---

## 6. 数据流

### 训练(dataloader)
```
episode parquet
 ├─ observation.state[16] ──> 手臂绝对 EEF(state_to_arm10, 不变) ──┐
 │                        └─> base 位姿(base_pos + base_quat) ─────┤─> proprio raw26 ─scatter─> 80-D
 └─ action[12] ───────────> base 命令(x/y/yaw vel, torso, mode) ──┐
                                                                    ├─> action raw25 ─scatter─> 80-D
    手臂绝对 EEF future(state_to_arm10 of state[1:]) ───────────────┘
 → 归一化(手臂 stats + base stats)→ map_to_unify → 80-D + mask
```

### 部署 / eval
```
model 输出 80-D
 └─ deploy(_UnifyAwareNormalizer): 80 ─gather─> raw25(手臂20绝对 + base5) ─unnormalize─> 送 client
      client(interface + bridge):
        手臂20绝对 ─eef20d_to_robocasa12d(+proprio+osc_scale)─> OSC Δpos/Δrot/grip
        base5 ────────────────────────────────────────────────> x/y/yaw/torso 直发 + mode 阈值0.5
      → 拼成 12-D env action(ACTION_SLICES 顺序)→ env.step
```

---

## 7. 逐文件改动

### 7.1 `openwam/dataloader/robocasa365.py`(核心)
- **新增读 action 字段**:加 `_read_action(ep, start, end)`(读 parquet `action` 列 [12]),类比 `_read_state`,带 lru_cache。
- **base action 提取**:从 LeRobot action[0:5] 取 `[x_vel, y_vel, yaw_vel, torso, control_mode]`。窗口对齐:action[t] 是 state t→t+1 的命令,故窗口 base action = `action[start : start+num_frames-1][0:5]`(和手臂 future 对齐)。
- **base proprio 提取**:从 state 取 `base_position[0:2]`(x,y)+ `base_rotation[3:7]`(quat)= 6-D,取窗口首帧。
- **80-D scatter**:改 `_build_sample`——手臂走现有绝对路径(0-9);新增 base 段拼进 raw25/raw26,再 `map_to_unify`。
- **两套 dst_index/mask**:`__init__` 里拆成 `_unify_dst_index_action`(map `["0-9","34-43","68-72"]`)和 `_unify_dst_index_proprio`(map `["0-9","34-43","73-78"]`),各配 `_unify_dim_mask_action` / `_unify_dim_mask_proprio`。
- **denormalize_action**:`unmap_from_unify` 用 action 的 dst_index → 得 raw25(手臂20 + base5),反归一(手臂用 arm stats、base 用 base stats)。
- **`action_dim` / `state_dim` property**:unify 开时 = `UNIFY_DIM`(80)。
- **归一化**:手臂 stats(现有 20-D)+ **新增 base stats**(5-D action:vel×3 + torso + mode;或 6-D proprio pose)。base velocity/torso 用 min-max;control_mode ∈{-1,+1} min-max 恒等。

### 7.2 `openwam/dataloader/robocasa365_stats_computation.py`
- 新增 base 维度的统计:遍历 episode 的 `action[0:5]`(base 段)算 min/max/mean/std。
- 落盘 stats 扩成 `{"eef": {20-D 手臂}, "base": {base 段}}` 或统一成 25-D 向量。deploy 端要能读回。

### 7.3 `configs/dataloader/robocasa365.yaml`
- `unify_action: true`(默认开,因为全量走 mobile + 统一头)。
- `unify_action_map`:拆成 action/proprio 两个,或加 base 段说明。
- **删** fixed-base 相关注释、`fixed_base_tasks.json` 引用。
- 加 base/torso/control_mode 的 scale 说明。

### 7.4 移除 fixed-base 筛选(决策 F)
- **删** `benchmarks/robocasa365/fixed_base_tasks.json`。
- `robocasa365.py`:删 `_fixed_base_task_names()`、`_FIXED_BASE_JSON`、`_resolve_task_roots` 里的 moma 过滤(root 发现所有 `*/lerobot` 桶,不再 drop)。
- `benchmarks/robocasa365/single_eval.py`:删 `_assert_fixed_base()` + 调用。
- `benchmarks/robocasa365/README.md`:改 scope 段(全 365 / 50 target),删 fixed-base 表 + PanTransfer orphan 段(全量后不再是孤儿)。
- `configs/dataloader/robocasa365.yaml`:删 112/111 计数注释。
- `tests/dataloader/test_robocasa365.py`:删 `test_root_mode_filters_to_fixed_base`、`_fixed_base_task_names` 相关断言。

### 7.5 `benchmarks/utils/action_conversion.py`(bridge)
- `eef20d_to_robocasa12d`:目前硬填 `base_motion=0 / control_mode=-1`。改成**接收 base 5-D 参数**(或新增 `assemble_robocasa12d(arm20, base5, proprio, scales)`):手臂桥接不变,base 直发(velocity/torso 原样,control_mode 阈值 0.5→±1),按 `ACTION_SLICES` 顺序拼 12-D。
- 保留旧 fixed-base 行为?否(决策 F);但可留一个 `base=None → 填0` 的兜底给纯手臂 ckpt。

### 7.6 `benchmarks/robocasa365/openwam2robocasa365_interface.py`
- server 现在回 **raw25**(手臂20 + base5),不是 20;`act()` 拆分:手臂20→bridge,base5→直发,拼 12-D。
- **proprio 发送**(⚠️ 实际实现与此计划不同):最终 proprio **默认保持 20-D**(base **位姿**是场景相关的,不入 proprio)。可选的 `base_proprio_velocity=true` 加入 base **速度**(body 系 `[vx,vy,vyaw]`,场景无关)→ 发送 **23-D** `[arm20, base_vel3]`,server scatter 到 `[68:71)`。见 README 的 "Base-velocity proprio" 一节。
- `STATE_DIM` 20→新值;`checks` 里 `state_dim_is_20` 相应更新。
- `DEFAULT_STATE_KEYS`:base 现在要用(不再只 debug)。

### 7.7 `openwam/deploy/model_loader.py` — `_UnifyAwareNormalizer`
- 现在 gather 80→20(手臂)。改成 80→raw25(手臂 20 + base 5),用 action 的 dst_index。
- 反归一:手臂 20-D 用 eef stats,base 5-D 用 base stats。
- 送 client 的 action = raw25(client 再 bridge)。
- 读 config 的 `unify_action_map`(现在是 action map)构建 dst_index。

### 7.8 `configs/model/dual_system.yaml`
- `architecture.action_dim: 80` / `state_dim: 80` — **已经是默认**,无需改。训练命令不用再 override(全量 unify)。

### 7.9 Eval scope(50 target)
- `benchmarks/robocasa365/` 加官方 50 target 任务清单(见 §9),`multi_eval.sh` / task-file 指向它。
- 每任务 eval:sim 有 `_check_success`,无需额外。
- `step_limits.yml`:mobile 任务 horizon 更长(composite 到 2400),补齐这 50 个。
- 注意 4 个 unseen(composite_unseen)训练时 hold out。

### 7.10 测试 `tests/dataloader/test_robocasa365.py`
- 改 mock bucket:parquet 加 `action[12]` 列(现在只有 observation.state)。
- 新增:base action 读取 + scatter 到 [68:72] 测试;base proprio scatter 到 [73:78];mask base 段 valid;denormalize 80→25 形状;control_mode 二值化。
- 删 fixed-base filter 测试。
- 更新 unify 测试(现在 action 80-D 含 base)。

---

## 8. 数据流验证(e2e)
沿用本会话的 e2e 路径,换成 mobile 任务:train(80-D 含 base)→ deploy(80→25 gather)→ 真 RoboCasa sim(mobile 任务)→ env.step 收 12-D(含非零 base)。debug bundle 检查 base_motion 非零、control_mode 切换。

---

## 9. 官方 50 target 任务(eval 集,已从 dataset_registry 抓)
**Atomic(18)**:CloseBlenderLid, CloseFridge, CloseToasterOvenDoor, CoffeeSetupMug, NavigateKitchen, OpenCabinet, OpenDrawer, OpenStandMixerHead, PickPlaceCounterToCabinet, PickPlaceCounterToStove, PickPlaceDrawerToCounter, PickPlaceSinkToCounter, PickPlaceToasterToCounter, SlideDishwasherRack, TurnOffStove, TurnOnElectricKettle, TurnOnMicrowave, TurnOnSinkFaucet
**Composite(33,含 seen+unseen)**:ArrangeBreadBasket, ArrangeTea, BreadSelection, CategorizeCondiments, CuttingToolSelection, DeliverStraw, DessertAssembly, GarnishPancake, GatherTableware, GetToastedBread, HeatKebabSandwich, KettleBoiling, LoadDishwasher, MakeIceLemonade, PackIdenticalLunches, PanTransfer, PortionHotDogs, PreSoakPan, PrepareCoffee, RecycleBottlesByType, RinseSinkBasin, ScrubCuttingBoard, SearingMeat, SeparateFreezerRack, SetUpCuttingStation, StackBowlsCabinet, SteamInMicrowave, StirVegetables, StoreLeftoversInBowl, WaffleReheat, WashFruitColander, WashLettuce, WeighIngredients
> registry 数到 51(18+33);论文写 50(18+16 seen+16 unseen=50 composite=32);差 1 个 composite,实现时以 registry target 为准、按 split_group 分 seen/unseen。

---

## 10. Open questions / 风险

1. **base action 窗口对齐**:action[t] 命令 state t→t+1。手臂 future = state[1:]、base future = action[0:T-1],需确认时序一致(off-by-one 风险)。
2. **control_mode 归一化**:{-1,+1} 直接当连续维,模型输出 sigmoid-ish;eval 阈值 0.5。是否要单独不归一化?(min-max 恒等,应无碍)
3. **base velocity 尺度**:action ctrlrange x/y ±1、yaw ±1.5;归一化后是否和手臂尺度失衡影响 loss 权重?或给 base 段单独 loss 权重。
4. **torso 只有 action 无 proprio**:模型预测 torso 但感知不到当前 torso 高度 → 可能学不准。可接受(RoboCasa 观测限制),或考虑从别处补 torso 观测。
5. **proprio 维度膨胀**:server 收 26-D proprio,`obs_preprocess` 的 state_dim 校验要放开/改。
6. **归一化 stats 落盘格式**:eef + base 两段,deploy 要能读回并分别反归一。定一个清晰 schema。
7. **mobile eval 更慢**:composite mobile horizon 到 2400 步,eval 1100 rollout 会更久。

---

## 11. 实现顺序(建议)

1. **dataloader**(§7.1-7.3):base 读取 + 80-D scatter + stats + config;单测(§7.10)绿。
2. **移除 fixed-base**(§7.4):删 json/filter/断言/测试。
3. **deploy**(§7.7):`_UnifyAwareNormalizer` 80→25。
4. **bridge + interface**(§7.5-7.6):base 直发 + proprio 扩展。
5. **eval scope**(§7.9):50 target 清单 + step_limits。
6. **e2e**(§8):train→deploy→mobile sim 跑通。
7. 更新 README + PR。

---

## 附:与 fixed-base 版的关系
- 全量后 **PanTransfer 孤儿问题自动消失**(mobile "Serving Food" 训练任务回归)。
- fixed-base 的绝大部分代码(手臂绝对 EEF、L-shape 多视角、eef stats)**复用**;主要是**新增 base 通道**,不是重写。
- 单臂 EEF 那套 20-D(0-9 + 34-43 masked)不动;base 是纯增量(reserved 段)。
