"""Differential harness: run one Wan backbone impl end-to-end and dump tensors.

Run the SAME script in two checkouts (process-isolated — they collide on the
``openwam`` package name) and compare. This harness needs the prepare/run_block/
finalize API, so both checkouts must carry it (e.g. a feature worktree vs an
earlier baseline worktree). For diffing against the main-branch reference (old
API), use the architecture-level run_arch_compare.py instead.

    ( cd <baseline-checkout> && PYTHONPATH=. python tests/wan_migration/run_backbone.py \
        --name wan22_ti2v_5b --model-dir /path/to/Wan2.2-TI2V-5B --out /tmp/old_ti2v.pt )
    ( cd <feature-checkout>  && PYTHONPATH=. python tests/wan_migration/run_backbone.py \
        --name wan22_ti2v_5b --model-dir /path/to/Wan2.2-TI2V-5B --out /tmp/new_ti2v.pt )

    python tests/wan_migration/compare.py /tmp/old_ti2v.pt /tmp/new_ti2v.pt

Inputs are produced by the backbone's own ``preprocess_input_for_train`` (so they are
shape-correct for any variant) from deterministic synthetic frames; a fixed seed
makes the noisy latents reproducible. The dumped final tensor is the bit-identical
acceptance target (training forward path).
"""

import argparse

import torch
from PIL import Image


def _synthetic_frames(batch, num_frames, height, width, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    frames = []
    for _ in range(batch):
        clip = []
        for _ in range(num_frames):
            arr = (torch.rand(height, width, 3, generator=g) * 255).to(torch.uint8).numpy()
            clip.append(Image.fromarray(arr, mode="RGB"))
        frames.append(clip)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--name", required=True, help="registry name (wan22_ti2v_5b / wan21_vace_1_3b / wan21_i2v_14b_480p)"
    )
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--num-frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--timestep", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from openwam.model.video_backbone import build_video_backbone

    device = torch.device(args.device)
    dtype = torch.bfloat16

    bb = build_video_backbone(args.name, None, source=args.model_dir, device="cpu")
    bb.set_dtype_device(dtype, device)

    frames = _synthetic_frames(args.batch, args.num_frames, args.height, args.width, seed=0)
    text = ["a robot arm picking up a red cube"] * args.batch
    # ref_images trigger first-frame conditioning on every variant.
    ref_images = [[clip[0]] for clip in frames]

    with torch.no_grad():
        pre = bb.preprocess_input_for_train(frames=frames, text=text, ref_images=ref_images)

        input_latents = pre["input_latents"]
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        noise = torch.randn(input_latents.shape, generator=g).to(dtype=dtype, device=device)
        latents = noise  # forward on pure noise — deterministic given seed
        timestep = torch.tensor([args.timestep] * input_latents.shape[0], device=device, dtype=dtype)

        loop_in = dict(
            latents=latents,
            timestep=timestep,
            context=pre["context"],
            seq_lens=pre.get("seq_lens"),
            vace_context=pre.get("vace_context"),
            vace_scale=pre.get("vace_scale", 1.0),
            clip_feature=pre.get("clip_feature"),
            y=pre.get("y"),
            fuse_vae_embedding_in_latents=pre.get("fuse_vae_embedding_in_latents", False),
            num_clean_prefix_frames=pre.get("num_clean_prefix_frames", 0),
        )
        state = bb.prepare(**loop_in)
        for i in range(bb.num_layers):
            state = bb.run_block(i, state)
        out = bb.finalize(state)

    # state_dict prefix check (migrated tree must have no _pipe.*).
    sd_keys = list(bb.state_dict().keys())
    has_pipe = any(k.startswith("_pipe.") for k in sd_keys)

    payload = {
        "name": args.name,
        "output": out.detach().to(torch.float32).cpu(),
        "output_shape": tuple(out.shape),
        "num_layers": int(bb.num_layers),
        "dim": int(bb.dim),
        "has_pipe_prefix": has_pipe,
        "sd_key_sample": sorted(sd_keys)[:8],
        "preproc_keys": {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in pre.items()},
    }
    torch.save(payload, args.out)
    print(f"[{args.name}] wrote {args.out}: output_shape={payload['output_shape']} has_pipe_prefix={has_pipe}")


if __name__ == "__main__":
    main()
