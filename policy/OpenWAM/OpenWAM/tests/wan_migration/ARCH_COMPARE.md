# Architecture-level golden regression gate

A differential harness that asserts a `video_backbone` change is **numerically
identical** end-to-end across every supported architecture, real backbone, and
external encoder — not just the bare backbone (that's `run_backbone.py`).

Use it before merging any `video_backbone` refactor: run the matrix between your
feature checkout and a baseline checkout; every config's forward + loss must be
bit-identical.

## Files

| File | Role |
|---|---|
| `run_arch_compare.py` | The harness. Builds one architecture on real weights, feeds a fixed synthetic batch, dumps deterministic forward outputs + loss for one checkout. |
| `compare_arch.py` | Diffs two dumps: forward `atol=0`, loss `Δ=0`, action-init fingerprint equal, no `_pipe.*` state_dict prefix. |
| `run_arch_matrix.sh` | Orchestrates the full matrix across two checkouts, 4 GPUs in parallel. |

## How the harness works

It re-implements **no** model logic. It only (1) reuses the production
construction path, (2) pins all randomness, (3) dumps tensors.

1. **Reuse the trainer's construction** — `build_arch()` mirrors
   `OpenWAMTrainer.__init__` line-for-line:
   `seed_everything` → `resolve_architecture_config` → `build_architecture`
   (loads the real Wan weights) → `set_dtype_device` → `init_training_schedulers`
   → `set_training_runtime`. `cfg` comes from `hydra.compose("train", overrides)`,
   same source as `scripts/train.py`. Only deepspeed/freeze/normalizer are
   dropped — none affect forward numerics.

2. **Fixed synthetic batch** — not the real dataloader (video decode / shuffle /
   workers are nondeterministic). A fixed `torch.Generator` builds
   video/prompt/action/proprio so the inputs are byte-identical across processes
   and checkouts.

3. **Deterministic forward** — `deterministic_forward()` replays `compute_loss`'s
   noise/timestep construction but with a **fixed generator + fixed timestep id**
   (no global `randn_like` / `randint`). The prediction then depends only on
   weights + graph, never on global-RNG consumption order — this is what makes
   `atol=0` comparison possible.

4. **Two dump paths** — forward path dumps `(video_pred, action_pred)`; loss path
   re-seeds globally then calls the real `compute_loss` and dumps the three loss
   scalars. Both under `no_grad` (no backward — required to fit the 14B I2V
   backbone on one GPU).

5. **Action fingerprint** — float64 sum of all `action_backbone` params, a probe
   for "did the random init line up across checkouts".

**Why it diffs main↔refactor:** the two checkouts are different code, but
`build_architecture / prepare_inputs / compute_loss / forward` share signatures,
so the *same* harness script runs against each via `PYTHONPATH`. Fixed input +
fixed seed + deterministic forward ⇒ a clean refactor produces byte-identical
output. `cfg.project.seed=42` alone aligns the randomly-initialized action
backbone across checkouts (verified by the fingerprint) — no weight sharing
needed.

## Running the gate

```bash
# 1. Make a baseline worktree at merge-base(feature, main) — the commit the
#    refactor branched from, so the diff isolates the video_backbone change and
#    not unrelated main commits:
cd <repo>
BASE=$(git merge-base HEAD origin/main)
git worktree add --detach /mnt/data/wangyuran/arch-main "$BASE"
# pull the harness into the baseline worktree (adds test scripts, not repo code):
git -C /mnt/data/wangyuran/arch-main checkout origin/<feature-branch> -- tests/wan_migration/

# 2. Run the matrix (4 GPUs). Standard backbones need no extra setup:
REF_DIR=$(pwd) MAIN_DIR=/mnt/data/wangyuran/arch-main \
  bash tests/wan_migration/run_arch_matrix.sh standard

# 3. Read the verdict:
grep -h '^RESULT' /mnt/data/wangyuran/arch_cmp/cmp_*.txt
```

Single config (debugging):

```bash
PYTHONPATH=. python tests/wan_migration/run_arch_compare.py \
  --out /tmp/new.pt --seed 42 --mode both --num-frames 9 --height 384 --width 320 -- \
  model=dual_system model.architecture.variant=joint_self_attn \
  model.video_backbone.name=wan22_ti2v_5b \
  model.video_backbone.model_path=/mnt/data/wangyuran/Wan2.2-TI2V-5B
# ...run again under the baseline checkout to /tmp/old.pt...
python tests/wan_migration/compare_arch.py /tmp/old.pt /tmp/new.pt
```

## Coverage matrix (23 configs)

**6 architectures × 3 real backbones**:

| | TI2V-5B | VACE-1.3B | I2V-14B |
|---|:--:|:--:|:--:|
| dual_system joint_self_attn | ✓ | ✓ | ✓ |
| dual_system joint_cross_attn | ✓ | ✓ | ✓ |
| dual_system idm | ✓ | ✓ | ✓ |
| shared_backbone vanilla | ✓ | ✓ | ✓ |
| shared_backbone moe | ✓ | ✓ | ✓ |
| tri_system joint_self_attn | ✓ | ✓ | ✓ |

**External-encoder swaps** (dual_system joint_self_attn, `from_scratch=true`):
`wan_vae`, `flux_vae`, `dinov3`, `vjepa2_1`, `vjepa2`.

## Environment notes (not refactor bugs — both checkouts need them)

- **wandb**: harness sets `WANDB_MODE=disabled` (no login needed).
- **I2V-14B memory**: loss path is `no_grad`; the matrix exports
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **VACE / I2V resolution**: must be 480×832 (the matrix sets this).
- **V-JEPA encoders**: need `pip install timm decord` on both checkouts. The REF
  side vendors the ViT in-tree (no submodule); a submodule-era baseline (e.g.
  `e89cac6`) still needs `git submodule update --init third_party/vjepa2`. Without
  these, `vjepa2_1` fails to import (`ModuleNotFoundError: timm`, plus `app`/`src`
  on the submodule-era side).
