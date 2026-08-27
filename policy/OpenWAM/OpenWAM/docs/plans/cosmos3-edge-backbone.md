# Cosmos3-Edge backbone integration — plan + findings

Status: **delivered 2026-08-08** — all three milestones green.

- **M1** (joint_cross_attn): 20-step debug train on RoboTwin adjust_bottle
  (losses healthy), self-contained checkpoint (vae component spec +
  `text_tokenizer/` copy + stats), deploy-path load-back verified.
- **M2** (joint_self_attn / MoT): 20-step debug train via the und-prefix-KV +
  GQA KV-expand path (`BlockLoopState.prefix_kv_*`, rectangular joint mask in
  `mot_driver.run_joint_loop`).
- **M3** (deploy): policy server on the M1 checkpoint answered
  `inference_single_test.py --test` (ping / 20-D action / reset). Note:
  `deploy.py` has no `--compile-mode` flag — compile is yaml-only; use the
  positional override `optimization.compile.mode=none` when needed.
- **Golden parity** (GPU, real weights): batched engine reproduces the native
  pipeline's cond pass — velocity cosine 0.999969, relative MSE 6e-5; per-layer
  bf16 drift curve 0.3%→7.7% (kernel reduction-order noise, bounded by a
  depth-scaled ladder in `tests/test_cosmos3_golden_parity.py`).
- Tests: `tests/test_cosmos3_*` — CPU (registry / api hygiene / scheduler /
  text_pack / block-loop parity / joint_self_attn / deploy artifacts) + GPU
  (`real_load`, `golden_parity`; gated on `COSMOS3_EDGE_ASSET_PATH` /
  `COSMOS3_GOLDEN_PATH`).
- Server artifacts: dev tree `/mnt/cpfs/zch/openwam_cosmos3_dev`, golden dump
  `/mnt/cpfs/zch/assets/cosmos3_golden/golden_step0.pt`, smoke checkpoints
  `/mnt/cpfs/zch/checkpoints/cosmos3_m{1,2}_smoke/`.

The sections below record the load-bearing facts the implementation is built on.

## What Cosmos3-Edge actually is

Dual-stream Mixture-of-Transformers (`Cosmos3OmniTransformer`, vendored at
`openwam/model/video_backbone/cosmos3/_vendor/` from diffusers @`6ad35739`):
28 layers × two full parameter sets per layer — **und** stream (text, causal,
never reads gen) and **gen** stream (video/action, bidirectional, attends
`[all und K; all gen K]`). No AdaLN, no cross-attention, no external text
encoder. GQA 16 Q-heads / 8 KV-heads / head_dim 128, relu² gateless MLP,
Nemotron RMSNorm, `k_norm_und_for_gen` third norm on und-K consumed by gen
(Edge: `qk_norm_for_text=false` → `norm_q`/`norm_k` are `Identity`).
Timestep conditioning = additive scatter of `time_embedder(time_proj(t ×
timestep_scale=0.001))` onto **noisy** tokens only; clean (conditioning)
frames get no timestep signal. i2v conditioning = clean latents pinned at
their positions (no channel concat). `lm_head` (268M params) is dead in the
diffusion forward — drop at load. VAE = Wan2.2-TI2V `AutoencoderKLWan`
(z=48, 4× temporal / 16× spatial, `latents_mean/std` in its config).

## Phase-0 verified facts

### Vendor gate (training venv: diffusers 0.38.0, transformers 4.51.3, torch 2.7.1)
- Vendored file (sole edit: relative→absolute imports) instantiates the mini
  Edge-flavoured config, forward + backward pass on CPU; real Edge weights load
  via `from_pretrained(bundle, subfolder="transformer")`: 3.370B params
  (layers 2.819B, embed 268M, lm_head 268M, action heads 8.5M). All Edge
  structure checks pass (`k_norm_und_for_gen` loaded non-zero, `norm_q`
  Identity, gateless MLP, action heads real).
- Tokenizer on venv-native transformers 4.51.3 works: `PreTrainedTokenizerFast
  .from_pretrained(bundle, subfolder="text_tokenizer")` + auto chat template.
  `eos=11`, `<|vision_start|>=20` (match config.json).

### Golden reference (`/mnt/cpfs/zch/assets/cosmos3_golden/golden_step0.pt`, 112 MB)
Native `Cosmos3OmniPipeline` i2v run on H20: prompt "A robot arm picks up a red
bottle from the table and lifts it straight up.", bundle asset image, 29 frames
480×832 @24fps, 2 steps, guidance 5.0, seed 42. Payload: full transformer
kwargs + velocity outputs for cond (`call0_*`) and uncond (`call1_*`) passes,
layer {0,1,13,27} und/gen in/out for the cond pass, shared rotary 4-tuple,
scheduler grid, final latents, input image, all configs.
- Sequence: 3188 = 68 text (`und_len`) + 3120 vision (8×15×26 patch grid,
  T_lat=8, H=480/16/2=15, W=832/16/2=26, T-major/H/W raster).
- Conditioning: noisy frames `[1..7]` (latent frame 0 clean, excluded);
  `vision_timesteps` length 2730 = noisy tokens only, value 995.0 at step 0.
- mRoPE positions are **float32** (fps modulation on): text = arange(68) on
  all 3 axes; vision T starts at `und_len + 15000 = 15068`, advances 1.0 per
  latent frame at 24 fps; H/W restart at 0 (`reset_spatial_ids`), W fastest.

### Scheduler discovery (affects deploy, not training)
`use_native_flow_schedule=True` was active and the pipeline passed the linear
ramp `linspace(1−1/1000, 0, N+1)[:-1]`, but UniPC `set_timesteps` with
`use_karras_sigmas=True` **ignored the custom sigmas** and recomputed the
karras grid — observed σ = [0.995025, 0.128160] = flow-transform σ/(1+σ) of
the checkpoint's `sigma_max=200 / sigma_min=0.147`. This is the bug open PR
huggingface/diffusers#14272 fixes (its Edge scheduler disables karras before
applying the ramp + flow_shift). Consequence: OpenWAM's shifted-flow Euler
adapter matches NVIDIA's *intended* semantics (linear flow grid + shift); do
not chase the currently-buggy main grid. NVIDIA train-time shift is
resolution-keyed: {256: 3, 480: 5, 720: 10} (matches OpenWAM's default
`shift_video=5.0` at 480p). Training target: rectified-flow velocity ε−x₀
(UniPC `flow_prediction` inverts as x₀ = x_t − σ·v), identical to the
interpolant hardcoded in `architectures/base.py` compute_loss.

### Spike env recipe (golden reruns)
`PYTHONPATH=/mnt/cpfs/zch/cosmos3_spike/pylib` over the training venv, target
dir holds: diffusers @6ad35739 sdist, huggingface_hub 1.27.0, transformers
5.14.1, safetensors 0.8.0 (all `pip install --no-deps --target`). transformers
4.5x hard-pins hub<1.0 at import, hence the 5.x shadow — spike only; the
production path never needs it. Dump script: `/tmp/cosmos3_golden_dump.py` on
the server (also in session scratchpad).

## Implementation contracts (from the codebase maps; file:line verified 2026-08-08)

- `preprocess_input_for_train(*, frames=None, text=None, **kw)` must emit
  `input_latents` (B,C,T,H,W) / `context` / `seq_lens`; `context` must be real
  (ActionDiT cross-attn + proprio concat consume it) → use the und final-layer
  hidden (2048-wide), `text_dim=2048`.
- `prepare(**pipeline_inputs)` needs `**_ignored`; architectures inject
  `force_per_token_t_mod=True` / `zero_clean_prefix_t_mod=True` and forward
  every preprocess key. Must set `grid_frames/height/width` and store both
  gradient-checkpointing flags; `time_mod`/`rope_freqs` may be 0-dim
  placeholders (predict2.5 precedent), real state in `extras`.
- und tower is computed **once** in `prepare()` (batched, right-padded, causal
  + padding mask), caching per-layer rotated `(k_und_for_gen, v_und)` in
  `extras` for `run_block`/`pre_attn_at_layer`.
- Scheduler adapter: reuse `CosmosFlowSchedulerAdapter` math verbatim
  (σ-shift transform, `timesteps = sigmas × num_train_timesteps` — hard
  constraint: `generate()`/`denoise_schedule.py` invert σ = t/num_train_ts
  outside the adapter), default shift 5.0.
- Child naming is the freeze API: VAE → `self.vae`, transformer → `self.dit`;
  no text_encoder child (loader's Reason1-clearing branch won't trigger).
- Deploy: add the new registry name to the cosmos prefix test in
  `openwam/deploy/model_loader.py:166-173`; `save_deploy_assets` no-ops when
  `model_path` unreadable; inference placeholder latents are z=48, spatial
  //16 (NOT predict2.5's z=16 //8).
- MoT (Phase 2): backbone's `pre_attn_at_layer` returns gen-stream q and
  prefix-concatenated k/v (`[k_und_layer_i; k_gen]`) with KV
  `repeat_interleave`d 8→16 heads (mathematically identical; avoids the
  flat-cat + single-`num_heads` rearrange break in `mot_driver.py:232-234` /
  `:124-142`). ABC gains a default-0 prefix-KV-token count; the driver mask
  becomes rectangular `(S_q, n_prefix+S_q)`; und rows never enter the joint
  sequence (updated inside the backbone half-steps → text never sees action).
- api-hygiene: zero public members beyond the `VideoBackbone` ABC.
- GPU test gating: `COSMOS3_EDGE_ASSET_PATH` (default
  `/mnt/cpfs/zch/assets/Cosmos3-Edge`) + `OPENWAM_COSMOS3_EDGE`, `pytest.skip`.
