"""Architecture-level differential harness: drive any of the 6 WAM architectures
end-to-end on REAL video-backbone weights and dump forward outputs + loss for a
main-vs-refactor bit-identical comparison.

It reuses the exact production construction sequence that
``OpenWAMTrainer.__init__`` runs (seed_everything -> resolve_architecture_config
-> build_architecture -> init_training_schedulers -> set_training_runtime) but
replaces the dataloader with a FIXED synthetic batch, so the inputs are
byte-identical across two process-isolated runs of different checkouts.

    # refactor checkout
    PYTHONPATH=. python tests/wan_migration/run_arch_compare.py \
        --out /tmp/new_dual.pt --seed 42 -- \
        model=dual_system \
        model.video_backbone.name=wan22_ti2v_5b \
        model.video_backbone.model_path=/mnt/data/wangyuran/Wan2.2-TI2V-5B
    # main checkout: same command, --out /tmp/old_dual.pt
    python tests/wan_migration/compare_arch.py /tmp/old_dual.pt /tmp/new_dual.pt

Hydra overrides go after ``--``. The forward path feeds fixed noise + fixed
timesteps (no global-RNG draws) so the dumped prediction depends only on weights
+ graph; the loss path re-seeds globally and runs the real ``compute_loss`` so
its scalar mirrors a true training step.
"""

import argparse
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def synthetic_batch(B, num_frames, H, W, action_dim, action_len, state_dim, use_proprio, seed=0):
    """Deterministic list-of-dict batch matching the dataloader sample contract
    consumed by ``BaseWAMArchitecture.prepare_inputs`` (video/prompt/action/proprio)."""
    g = torch.Generator().manual_seed(seed)
    batch = []
    for _ in range(B):
        frames = [
            Image.fromarray((torch.rand(H, W, 3, generator=g) * 255).to(torch.uint8).numpy(), "RGB")
            for _ in range(num_frames)
        ]
        sample = {
            "video": frames,
            "prompt": "a robot arm performing a manipulation task",
            "action": torch.randn(action_len, action_dim, generator=g),
        }
        if use_proprio:
            sample["proprio"] = torch.randn(state_dim, generator=g)
        batch.append(sample)
    return batch


def build_arch(cfg, seed):
    """Mirror OpenWAMTrainer.__init__'s construction (minus deepspeed/freeze/normalizer)."""
    from openwam.model import build_architecture, resolve_architecture_config
    from openwam.train.utils.seeding import seed_everything

    seed_everything(seed, rank=0)
    resolved = resolve_architecture_config(cfg.model)
    arch = build_architecture(resolved.registry_name, resolved.params)
    arch.set_dtype_device(arch.dtype, arch.device)
    arch.init_training_schedulers(1000)
    arch.set_training_runtime(
        use_gradient_checkpointing=False,
        use_gradient_checkpointing_offload=False,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    )
    arch.eval()
    return arch, resolved.registry_name


def action_fingerprint(arch):
    """float64 param sum of action_backbone — flags random-init mismatch across branches."""
    ab = getattr(arch, "action_backbone", None)
    if ab is None:
        return None
    total = 0.0
    numel = 0
    for p in ab.parameters():
        total += p.detach().to(torch.float64).sum().item()
        numel += p.numel()
    return {"sum": total, "numel": numel}


def deterministic_forward(arch, inputs, *, seed, v_tid_frac=0.5, a_tid_frac=0.5):
    """Replicate compute_loss's noise/timestep build but fully deterministic
    (fixed generator + fixed timestep ids), so the prediction depends only on
    weights + computation graph, not on global-RNG consumption order."""
    vb = arch.video_backbone
    ab = arch.action_backbone
    dtype, device = arch.dtype, arch.device
    g = torch.Generator().manual_seed(seed)

    inp = dict(inputs)
    input_latents = inp["input_latents"]
    B = input_latents.shape[0]

    vnoise = torch.randn(input_latents.shape, generator=g).to(dtype=dtype, device=device)
    num_vts = len(vb.scheduler.timesteps)
    v_tid = torch.full((B,), int(num_vts * v_tid_frac), dtype=torch.long)
    if hasattr(vb, "add_training_noise"):
        latents = vb.add_training_noise(input_latents, vnoise, v_tid)
    else:
        vsigma = vb.scheduler.sigmas[v_tid].to(dtype=dtype, device=device).view(B, 1, 1, 1, 1)
        latents = (1 - vsigma) * input_latents + vsigma * vnoise
    if inp.get("first_frame_latents") is not None:
        latents[:, :, 0:1] = inp["first_frame_latents"]
    v_ts = vb.scheduler.timesteps[v_tid].to(dtype=dtype, device=device)

    noisy_actions = a_ts = None
    actions = inp.pop("actions", None)
    if actions is not None and ab is not None:
        asched = ab.scheduler
        actions = actions.to(dtype=dtype, device=device)
        anoise = torch.randn(actions.shape, generator=g).to(dtype=dtype, device=device)
        num_ats = len(asched.timesteps)
        a_tid = torch.full((B,), int(num_ats * a_tid_frac), dtype=torch.long)
        asigma = asched.sigmas[a_tid].to(dtype=dtype, device=device)
        noisy_actions = asched.add_noise(actions, anoise, asigma.view(B, 1, 1))
        a_ts = asched.timesteps[a_tid].to(dtype=dtype, device=device)

    fi = dict(inp)
    proprio = fi.pop("proprio", None)
    proprio_mask = fi.pop("proprio_mask", None)
    for k in (
        "use_gradient_checkpointing",
        "use_gradient_checkpointing_offload",
        "action_is_pad",
        "video_is_pad",
        "max_timestep_boundary",
        "min_timestep_boundary",
    ):
        fi.pop(k, None)
    fi["latents"] = latents
    if proprio_mask is not None:
        fi["_proprio_sample_mask"] = proprio_mask

    with torch.no_grad():
        vpred, apred = arch(
            noisy_actions,
            a_ts,
            proprio=proprio,
            use_gradient_checkpointing=False,
            **fi,
            timestep=v_ts,
        )
    return vpred, apred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num-frames", type=int, default=9)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--width", type=int, default=320)
    ap.add_argument("--action-dim", type=int, default=20)
    ap.add_argument("--action-len", type=int, default=16)
    ap.add_argument("--state-dim", type=int, default=20)
    ap.add_argument("--mode", choices=["forward", "loss", "both"], default="both")
    ap.add_argument("overrides", nargs="*", help="hydra overrides after `--`")
    args = ap.parse_args()

    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(PROJECT_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="train", overrides=list(args.overrides))

    arch, registry_name = build_arch(cfg, args.seed)
    use_proprio = bool(getattr(arch, "uses_proprioception", False))

    batch = synthetic_batch(
        args.batch,
        args.num_frames,
        args.height,
        args.width,
        args.action_dim,
        args.action_len,
        args.state_dim,
        use_proprio,
        seed=0,
    )

    payload = {
        "registry_name": registry_name,
        "seed": args.seed,
        "geom": {
            "dim": int(arch.video_backbone.dim),
            "num_layers": int(arch.video_backbone.num_layers),
            "num_heads": int(arch.video_backbone.num_heads),
        },
        "action_fp": action_fingerprint(arch),
        "has_pipe_prefix": any(k.startswith("_pipe.") for k in arch.state_dict().keys()),
    }

    if args.mode in ("forward", "both"):
        with torch.no_grad():
            inputs = arch.prepare_inputs(batch)
        vpred, apred = deterministic_forward(arch, inputs, seed=args.seed + 1)
        payload["video_pred"] = vpred.detach().to(torch.float32).cpu()
        payload["video_pred_shape"] = tuple(vpred.shape)
        payload["action_pred"] = None if apred is None else apred.detach().to(torch.float32).cpu()
        payload["action_pred_shape"] = None if apred is None else tuple(apred.shape)

    if args.mode in ("loss", "both"):
        from openwam.train.utils.seeding import seed_everything

        seed_everything(args.seed, rank=0)
        # no_grad: we never backward, so building the autograd graph only wastes
        # activation memory (OOMs the 14B I2V backbone); loss value is identical.
        with torch.no_grad():
            loss_inputs = dict(arch.prepare_inputs(batch))
            actions = loss_inputs.pop("actions", None)
            res = arch.compute_loss(**loss_inputs, actions=actions, lambda_video=1.0, lambda_action=1.0)
        payload["loss"] = float(res["loss"].item())
        payload["loss_video"] = float(res["loss_video"].item())
        payload["loss_action"] = float(res["loss_action"].item())

    torch.save(payload, args.out)
    fp = payload["action_fp"]["sum"] if payload["action_fp"] else None
    print(
        f"[{registry_name}] wrote {args.out}: "
        f"video_pred_shape={payload.get('video_pred_shape')} "
        f"loss={payload.get('loss')} action_fp_sum={fp}"
    )


if __name__ == "__main__":
    main()
