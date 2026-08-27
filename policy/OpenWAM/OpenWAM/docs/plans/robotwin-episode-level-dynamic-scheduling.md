# RoboTwin 评测调度优化 — episode 级动态分配 + 环境亲和 — 实现 Plan

**Date:** 2026-07-24
**Repo:** OpenWAM, branch `feat/robotwin_eval_plus`
**Status:** 设计草案,待评审

---

## 1. 问题

现在 `parallel_eval.sh` / `dlc_parallel_eval.sh` 的**工作单元是一整个 `task|mode`**:队列里一个条目 =
一个任务的全部 episode(默认 `test_num=100`),被某个 worker 领走后由**一个** `single_eval.sh` 进程从头
跑到尾(100 个 episode),期间该 worker 独占一张卡。

后果(用户原话):8×8=64 张卡起 64 个 server + 64 个 worker,50 任务 × 2 mode = 100 个 job。头一批 64 个并行,
快的 worker 陆续领走剩下的 36 个;但到**尾部**,当只剩 18 个 job 时,只有 18 个 worker 在忙、**46 张卡空闲**;
最坏情况只剩最后 1 个 job 时,**63 张卡空转等一个任务跑完 100 个 episode**。严重浪费。

**根因**:切分粒度是「整任务」,不是「每个 episode」。一个 job 不可再分,所以尾部无法把一个任务的剩余 episode
摊到空闲卡上。

## 2. 目标

- 把可调度的最小单元从「整任务」降到「**单个 episode**」,让空闲卡在尾部能**加入正在跑的任务**、并行消化其剩余 episode。
- 同时**尊重环境启动开销**:episode 优先分配给**已经启动过该任务环境**的卡(环境亲和),避免频繁重启环境。
- 只有当**空闲卡数多于「还没开始的任务数」**时,才对某个正在跑的任务**追加启动新环境**去并行它;且当某任务
  **剩余 episode 少于阈值 θ** 时不再为它追加环境(新环境启动期间该任务很可能已被原环境跑完,追加白费开销)。
- 结果:任何时刻只要还有未完成的 episode,就没有空闲卡;尾部「最后一个任务」被多卡并行,整轮墙钟时间显著下降。

## 3. 关键前置事实(基于 RoboTwin `script/eval_policy.py` 真实源码)

`main()` 里循环外只做一次:`class_decorator(task_name)` 构造 `TASK_ENV`、`get_model()` 连策略 server、
`st_seed = 100000 * (1 + seed)`(`--seed` 默认 0 → `st_seed = 100000`)、`test_num = 100`。
`eval_policy()` 的核心是**拒绝采样的 seed 流**(每轮 `now_seed += 1`,只有专家通过才算一个有效 episode):

```python
now_seed = st_seed;  succ_seed = 0;  now_id = 0
while succ_seed < test_num:                                  # 默认 100
    # ① 专家 check(每个 seed 都跑一次完整脚本专家,很贵)
    try:
        TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
        episode_info = TASK_ENV.play_once()
        TASK_ENV.close_env()
    except UnStableError: TASK_ENV.close_env(); now_seed += 1; continue   # 该 seed 作废
    except Exception:     TASK_ENV.close_env(); now_seed += 1; continue

    if TASK_ENV.plan_success and TASK_ENV.check_success():   # 专家能解 → 这张 seed 有效
        succ_seed += 1; suc_test_seed_list.append(now_seed)
    else:
        now_seed += 1; continue                              # 专家解不了 → 丢弃

    # ② 用【同一个 now_seed】重新布场景,给被测策略 rollout
    TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
    results = generate_episode_descriptions(task_name, [episode_info["info"]], test_num)
    instruction = np.random.choice(results[0][instruction_type])   # ← 进程级 RNG,非 seed 决定
    TASK_ENV.set_instruction(instruction)
    # (可选)ffmpeg 录 episode{TASK_ENV.test_num}.mp4
    succ = False; reset_func(model)                          # → interface.reset_model → server reset
    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        eval_func(TASK_ENV, model, TASK_ENV.get_obs())       # ← 我们的 interface.eval()
        if TASK_ENV.eval_success: succ = True; break
    if succ: TASK_ENV.suc += 1                               # 打印 Success rate: suc/test_num
    now_id += 1
    TASK_ENV.close_env(clear_cache=((succ_seed + 1) % clear_cache_freq == 0))
    TASK_ENV.test_num += 1;  now_seed += 1
return now_seed, TASK_ENV.suc
```

**对本方案至关重要的五点**:

1. **episode 由 seed 决定**。要把一个任务的 episode 拆到多个环境实例并行,必须让各实例领**互不相交的 seed**,
   否则同一 seed = 同一场景 = 重复评测。→ 由**中心分派器持有一个 per-(task,mode) 单调 seed 计数器**统一发号,
   天然去重。
2. **seed 消耗速率不定**(拒绝采样),"凑够 `test_num` 个有效 episode 要试多少 seed"事先未知,且**每次专家 check
   都是一次完整专家 rollout(贵)**。→ 只能**惰性发号**(worker 一次要一张 seed),不能静态预切 seed 区间;并行也把
   专家 check 的成本一并摊开。
3. **摊薄的重开销是"进程构造",不是"场景"**:`setup_demo`/`close_env` 本来就每 episode 做;循环外只做一次的是
   模块 import + `TASK_ENV` 构造 + CUDA/curobo 预热 + warp 补丁 + server 连接。→ **一个 worker 进程 = 一个
   (task,mode) 的评测**:同任务连跑多 episode 复用同一进程(= 现在整任务跑法的既有行为,零新增风险);换任务 = 进程
   退出、supervisor 起新进程(换 task 模块 / 新 `TASK_ENV` 才是未验证的坑,进程级隔离规避)。
4. **`reset_func(model)` 每个有效 episode 一次** → 给该 worker **独占的**策略 server(port base+i)发 `reset` 清 chunk
   缓冲,worker 之间互不干扰。
5. **两个单进程假设会失效,须改**:
   - 成绩聚合:`main()` 写 `_result.txt = suc_nums/test_num`、每 episode 打印 `Success rate: X/Y`(shell 现在
     grep 它)。episode 拆到多进程后这套失效 → 聚合**上移到 dispatcher**(§6)。
   - 视频撞名:`episode{TASK_ENV.test_num}.mp4` 的 `test_num` 是**进程内局部计数**,同任务多进程写同一目录会覆盖
     → 视频输出按 (node,worker)/seed 命名空间隔离(§6)。

> 语义与上游的关系(须在文档/日志声明):分派器按 seed 顺序发出的仍是连续前缀 `st_seed, st_seed+1, …`,而某 seed
> 有效与否是确定的,所以**前 `test_num` 个有效 episode 的场景与上游单进程逐位一致**;差异只在收尾——见 §5.2 的
> commit 握手可做到**精确 `test_num`、零 overshoot**(边界处"选哪几张有效 seed"仍有微小时序非确定,属并行固有)。
> 需完全逐 episode 复现时用 `strict / --no-dup` 模式:每 job 恒定 1 环境、不追加,退化为单流。

## 4. 总体架构

三个角色(单机 / 多机 DLC 统一)。**策略 server 完全不变**(仍是每卡一个、常驻)。改造的是 RoboTwin 客户端侧的调度。

```
                    ┌───────────────────────────────────────────────┐
                    │  Dispatcher(中心分派器,rank0 上 1 个 TCP 服务) │
                    │  持有 per-(task,mode): target/done/单调seed计数 │
                    │  + 环境亲和与追加环境的调度策略 + 结果聚合        │
                    └───────────────▲───────────────────────────────┘
             REQUEST_TASK/EPISODE   │   (TCP, 跨节点)
             REPORT / assign        │
        ┌───────────────┬──────────┴────────┬───────────────┐
   ┌────┴────┐     ┌────┴────┐          ┌────┴────┐
   │ slot0   │     │ slot1   │   ...    │ slotK   │   每卡一个 slot(= GPU i + server port base+i)
   │supervisor│    │supervisor│         │supervisor│  常驻 shell 循环:向 dispatcher 要任务→起 worker 进程→退出后再要
   └────┬────┘     └────┬────┘          └────┬────┘
   episode_worker   episode_worker      episode_worker    一个进程 = 一个(task,mode)的评测;跑到 drain 就退出
   (booted env T)   (booted env U)      (booted env T')   同任务多 episode 复用同一进程(环境亲和)
```

- **Dispatcher**:新 `benchmarks/robotwin/dispatcher.py`。单机跑在 `127.0.0.1`;DLC 跑在 rank0,地址经共享 FS
  文件 `.dispatcher_addr`(或 `MASTER_ADDR`)广播给各节点。持久化 `results.jsonl` + `summary.tsv`(权威成绩来源)。
- **Supervisor**:每个 GPU slot 一个常驻 shell 循环(在 `parallel_eval.sh` / `dlc_parallel_eval.sh` 里)。向 dispatcher
  申请「该起哪个任务」;拿到 `(task,mode)` 就起 `episode_worker.py`(带上该 slot 的 GPU / server port);worker 退出后
  再申请;拿到 `exit` 就结束该 slot。
- **episode_worker**:新 `benchmarks/robotwin/episode_worker.py`。复用 `eval_policy_wrapper.py` 的模块加载 + 猴补
  (`_prewarm_cuda_for_curobo` / `_patch_warp_torch_namespace` / `class_decorator` / planner fallback)。构造**一个**任务
  环境后,循环:`REQUEST_SEED` 拿 seed → 专家 check → 失败 `REPORT_PROBE` 换下一张;通过则 `REQUEST_COMMIT` →
  批准才用同一 seed 重布场景 + 策略 rollout + `REPORT_RESULT`(见 §5.2)→ 直到 `drain`/未批准,退出。

## 5. 调度算法(Dispatcher 核心)

Dispatcher 维护每个 job = `(task, mode)` 的状态:
```
target        # 目标有效 episode 数(= test_num,可被 ROBOTWIN_TEST_NUM 覆盖)
done          # 已完成并回报的有效 episode 数(成功+失败)
committed     # 已通过专家 check、经 dispatcher 批准、正在策略 rollout 的有效 episode 数
probing       # 已发出 seed、专家 check 结果未知的在途数
next_seed     # 单调 seed 发号器(初值 st_seed = 100000*(1+base_seed))
live_envs     # 当前正在跑该 job 的 episode_worker 数
started       # 是否已被起过(区分“未开始”vs“进行中”)
```
不变式:`done + committed <= target`(commit 握手保证,见 §5.2);`remaining = target - done - committed`。
全局:`free_slots`(空闲 GPU slot 队列)、`unstarted_jobs`(started=False 的 job 数)。

### 5.1 slot 要任务(`assign_task`)——决定一个空闲 slot 该起哪个 job

按优先级:
1. **有未开始的 job** → 分配「剩余最多」的未开始 job(先铺开覆盖面,`started=True`)。
2. **没有未开始的 job,但有进行中的 job**(尾部场景)→ 考虑**追加环境**去并行某个进行中的 job:
   - 候选:`remaining = target - done - committed` **≥ θ**(阈值,§5.3)且 `live_envs < cap`(每 job 环境上限,§5.3)。
   - 在候选里选 **ETA 最长**的:即 `remaining / live_envs` 最大者(最欠并行、最拖尾的)。
   - 分配它(`live_envs += 1`)。
3. **没有可追加的 job**(全部要么完成、要么剩余 < θ 不值得追加)→ 回 `exit`,该 slot 结束。

> 直觉:阶段 1 把 64 张卡铺满不同任务;任务陆续完成后进入阶段 2,空闲卡不断「加入」还在跑且剩余够多的任务;
> 到最后一个任务时,只要它剩余 ≥ θ 就会被多卡瓜分,直到剩余 < θ(此时再追加也来不及,交给在跑的环境收尾)。

### 5.2 worker 要 episode / commit 握手 / 回报(精确 test_num、零 overshoot)

利用真实代码里"专家 check(便宜地判定 seed 是否有效)在贵的策略 rollout **之前**"这个天然提交点,加一步 commit 握手,
既不重复评测、又恰好跑满 `test_num`:

- `REQUEST_SEED(task,mode)`:若 `done + committed >= target` → 回 `{drain:true}`(worker 退出,`live_envs -= 1`);
  否则发号 `seed = next_seed++`,`probing += 1`,回 `{seed}`。(按 `done+committed` 门控,不按 `probing`——因为
  probe 可能被拒,得略微超发以填满流水线。)
- worker 用 `seed` 跑**专家 check**:
  - **专家失败** → `REPORT_PROBE{seed, valid:false}` → `probing -= 1`(seed 作废),worker 立刻再 `REQUEST_SEED`。
  - **专家通过** → `REQUEST_COMMIT{seed}`:dispatcher `probing -= 1`;若 `done + committed < target` → `committed += 1`
    回 `{commit:true}`,否则回 `{commit:false}`(worker 丢弃这张已布好的有效场景,退出并让 slot 重分配)。
- 拿到 `commit:true` 的 worker 才**用同一 seed 重布场景 + 策略 rollout**,完事 `REPORT_RESULT{seed, success, steps,
  step_limit_hit}` → dispatcher `committed -= 1; done += 1`,累加成功数。

> 不变式 `done + committed <= target` 恒成立 → **最终恰好 `target` 个 episode,零 overshoot**。唯一"浪费"是边界处
> 少量已通过专家 check 却被 `commit:false` 丢弃的场景(≤ 当时该 job 在途 probe 数),可忽略。
> commit 握手比"先跑完再计数"多一次 RTT,但相对一次策略 rollout(秒级~十秒级)可忽略。

### 5.3 阈值与上限(可配置)

- **θ = `MIN_REMAINING_FOR_DUP`**:某 job 剩余有效 episode 少于 θ 时不再追加新环境。
  含义:环境启动耗时 ≈ B 秒,单 episode ≈ E 秒,单个在跑环境在新环境启动完成前能再消化约 `B/E` 个 episode。
  故建议 `θ ≈ ceil(B/E) * live_envs` 的动态估计;实现上先给**固定默认 θ=8**(或 CLI/env `--min-remaining-for-dup`),
  后续可用实测 B、E 自适应。
- **cap = `MAX_ENVS_PER_JOB`**:每 job 并行环境上限 = `max(1, ceil(remaining / θ))`,再夹到「总 slot 数」。
  防止对一个只剩 θ 个 episode 的任务堆几十个环境(它们刚启动就 drain,纯浪费)。
- 兼顾**公平**:阶段 2 选 ETA 最长的,避免某任务被饿死。

## 6. 结果聚合与产物

成绩来源从「各 `single_eval.sh` 进程各自写 `_result.txt` + grep」**上移到 dispatcher**(因为一个任务的 episode 现在散在
多个进程/时间段):

- Dispatcher 收到每条 `REPORT_RESULT` 追加到 `results.jsonl`:
  `{ts, task, mode, seed, success, steps, step_limit_hit, node, worker}`。
- Dispatcher 实时维护 per-(task,mode) 的 `done / suc`,收尾时写 `summary.tsv`
  (列对齐现有:`task mode ... status ...` + 新增 `success_rate episodes step_limit_hits`),
  并保留 `run.env`(参数快照)。
- `export_results_csv.py`:改成优先读 dispatcher 的 `results.jsonl` / `summary.tsv`(无需再从各 task log grep
  "Success rate");旧 log 解析路径保留做兼容回退。
- `benchmarks/web_control.py` / `dlc_web_console.py`:数据源切到 dispatcher 的实时状态(可加一个 `GET /api/state`
  由 dispatcher 直接吐 JSON,或继续读 `results.jsonl`),即可实时看每任务进度、并行环境数、空闲卡数。

## 7. 跨节点(DLC)

- Dispatcher 只在 **rank0** 起一个 TCP 服务;监听 `0.0.0.0:<dispatch_port>`。
- 地址广播:rank0 把 `host:port` 写到共享 FS `${LOG_DIR}/.dispatcher_addr`;其余节点轮询该文件拿地址(复用现有
  `READY_FILE` 的等待惯例)。也可直接用 `MLP_WORKER_0_HOST` / `MASTER_ADDR`。
- 各节点的每个 slot supervisor 都连到同一个 dispatcher;调度、发号、去重全局一致。
- 共享 FS 仅用于:日志、`results.jsonl`(可由 rank0 dispatcher 单写,避免多写者)、地址广播、node-done 哨兵。
  **废弃** `queue/pending` + `mv` 认领协议(被 dispatcher 取代);`--dry-run` 改为「dispatcher + 假 worker(sleep)」
  验证调度而不起 server/sim。

## 8. 涉及文件

| 文件 | 改动 |
|---|---|
| `benchmarks/robotwin/dispatcher.py` | **新增**。TCP 分派器 + §5 调度策略 + §6 聚合 + 持久化。可独立 `python -m` 起。 |
| `benchmarks/robotwin/episode_worker.py` | **新增**。复用 wrapper 猴补;构造单任务环境;循环 REQUEST_SEED→专家check→REQUEST_COMMIT→rollout→REPORT_RESULT。 |
| `benchmarks/robotwin/eval_policy_wrapper.py` | 抽出可复用函数(模块加载 / prewarm / warp 补丁 / planner fallback / class_decorator 包装),供 episode_worker 调用;保留原 `main()` 供 single_eval 用。 |
| `benchmarks/robotwin/parallel_eval.sh` | 改:起 dispatcher(localhost)→起 N 个 server(不变)→起 N 个 slot supervisor 循环(替代原 flock 队列 worker)。 |
| `benchmarks/robotwin/dlc_parallel_eval.sh` | 改:rank0 起 dispatcher + 广播地址;各节点起 server + slot supervisor;删 `queue/mv` 逻辑;`--dry-run` 改造。 |
| `benchmarks/robotwin/single_eval.sh` / `multi_eval.sh` | **保留不动**(单任务调试仍走 RoboTwin 原生整任务循环)。 |
| `benchmarks/robotwin/export_results_csv.py` | 数据源优先 dispatcher 产物;保留旧解析回退。 |
| `benchmarks/web_control.py` / `dlc_web_console.py` | 数据源接 dispatcher(实时进度 / 并行度 / 空闲卡)。 |
| `benchmarks/robotwin/README.md` | 更新:新调度模型、θ/cap 旋钮、determinism 说明、strict 模式、dry-run 新语义。 |

## 9. 协议(worker ↔ dispatcher,JSON over TCP,一发一收)

```
→ HELLO          {node, worker, gpu, port}                     ← {ok}
→ REQUEST_TASK   {node, worker}                                ← {action:"run", task, mode, base_seed} | {action:"exit"}
→ REQUEST_SEED   {task, mode}                                  ← {seed:N} | {drain:true}
→ REPORT_PROBE   {task, mode, seed, valid:false}              ← {ok}                    # 专家 check 失败,seed 作废
→ REQUEST_COMMIT {task, mode, seed}                           ← {commit:true} | {commit:false}   # 专家通过,问是否 rollout
→ REPORT_RESULT  {task, mode, seed, success, steps, step_limit_hit}   ← {ok}           # rollout 完成
```
连接可长连(一个 episode_worker 一条连接);dispatcher 单线程事件循环 + 锁保护计数即可(QPS 极低,瓶颈在 sim)。
worker 掉线:dispatcher 按连接断开回收其 `probing`/`committed` 占位(该 seed 归还,后续用新 seed 补足 target),不死等。

## 10. 分阶段实施

- **M1 Dispatcher + 协议(纯逻辑,可单测)**:实现 §5 调度 + §9 协议 + §6 聚合;写单测覆盖:
  seed 去重、commit 握手下 `done` 恰好 = target(零 overshoot)、drain/未批准边界、阶段1→2 切换、θ/cap 生效、
  ETA 公平选择、worker 掉线回收。用「假 worker」(sleep + 伪造 valid/success)端到端跑通,不接 sim。等价于新 `--dry-run`。
- **M2 episode_worker(接真 RoboTwin)**:从 `eval_policy.py` 抽出单 episode 逻辑,复用 wrapper 猴补;先跑
  `ROBOTWIN_TEST_NUM` 小值单卡验证成功率与旧版一致(strict/no-dup 模式下应统计等价)。
- **M3 parallel_eval.sh 单机多卡**:supervisor 循环替换 flock 队列;验证尾部无空闲卡、单任务被多卡瓜分。
- **M4 dlc_parallel_eval.sh 多机**:dispatcher 地址广播 + 跨节点连接;删 mv 队列;多节点 smoke。
- **M5 产物 / web / csv / README**:聚合产物切换、实时看板、文档与 determinism 说明。

## 11. 备选与取舍

- **备选 B(轻量,不引中心分派器)**:把 job 切成「seed 分片」——`task×mode×shard`(每片固定 seed 区间、约 10 个
  episode),沿用现有 `queue/mv` 认领,并给认领加「亲和」:优先认领**自己刚跑过的任务**的下一片。
  - 优点:改动小、无需 TCP、复用现有 FS 基建。
  - 缺点:① 每片仍各自启动一次环境(除非命中亲和),尾部仍有「分片启动开销」;② 亲和是尽力而为、非精确;
    ③ seed 分片是**预切**,拒绝采样下每片有效 episode 数不均,尾部仍可能长尾;④ 不满足用户明确要的
    「按剩余数阈值决定是否追加环境」这类动态策略。
  - 结论:**采用 A(中心分派器)**,与用户描述(分派线程持计数 + 环境亲和 + 阈值追加环境)一致;B 作为降级参考。
- **进程内换任务 vs 进程级换任务**:选**进程级**(worker=一个 task-env 生命周期,换任务=退出重启),规避 SAPIEN
  场景在进程内反复重建的稳定性问题;代价是换任务时一次进程重启(sim 侧,策略 server 不重启)。
- **精确计数 vs 简化**:默认用 §5.2 的 commit 握手 → 恰好 `test_num` 个 episode、零 overshoot(代价:一次 RTT +
  边界处丢弃极少量已通过专家 check 的场景)。若想省掉握手,可退到"先 rollout 再计数"的简化版,代价是 ≤ (并发环境−1)
  的超采(成绩按实际 N 计,样本更多反更准);二者皆可,推荐 commit 版以对齐上游 `test_num` 口径。

## 12. 风险

- 抽取 RoboTwin 单 episode 逻辑需**忠实复刻**上游 `eval_policy` 的 seeding/reset/rollout/成功判定,否则成绩偏移。
  缓解:M2 用 strict/no-dup 模式与旧版逐任务比对成功率。
- Dispatcher 成为 rank0 单点;崩溃则整轮卡住。缓解:连接断开/心跳超时回收 `probing`/`committed`(worker 掉线其在途 seed 归还,
  允许重发以补足 target)+ 清晰日志。
- θ/cap 默认值不当会「追加太晚(尾部仍有空转)」或「追加太早(白起环境)」。缓解:暴露旋钮 + 实时看板观察,
  后续用实测 B/E 自适应 θ。
