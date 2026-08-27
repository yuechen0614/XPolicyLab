"""Deploy-path differential harness: dump ``preprocess_input_for_inference`` output.

The deploy conditioning (variant branches + the WanVideoPipeline unit-runner)
lives in ``WanVideoBackbone.preprocess_input_for_inference``; the train-path
harness (``run_backbone.py``) does NOT exercise it. This harness builds a
backbone from real weights, constructs the preprocess kwargs with the
per-variant conditioning signal (first_frame_image for I2V/TI2V, vace_video for
VACE), runs ``preprocess_input_for_inference``, and dumps every tensor field of
the resulting ``inputs_shared`` dict. ``compare.py`` then asserts the deploy
conditioning is bit-identical across a refactor (e.g. removing the unit-runner /
Components rebuild).

    PYTHONPATH=. python tests/wan_migration/run_deploy.py \
        --name wan21_vace_1_3b --model-dir /path/to/Wan2.1-VACE-1.3B --out /tmp/dep_vace.pt
    python tests/wan_migration/compare.py /tmp/old_dep.pt /tmp/new_dep.pt
"""

import argparse

import torch
from PIL import Image


def _synthetic_frames(num_frames, height, width, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    clip = []
    for _ in range(num_frames):
        arr = (torch.rand(height, width, 3, generator=g) * 255).to(torch.uint8).numpy()
        clip.append(Image.fromarray(arr, mode="RGB"))
    return clip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-frames", type=int, default=21)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from openwam.model.video_backbone import build_video_backbone

    device = torch.device(args.device)
    bb = build_video_backbone(args.name, None, source=args.model_dir, device="cpu")
    bb.set_dtype_device(torch.bfloat16, device)

    clip = _synthetic_frames(args.num_frames, args.height, args.width, seed=0)
    with torch.no_grad():
        inputs_shared = bb.preprocess_input_for_inference(
            prompt="a robot arm picking up a red cube",
            first_frame_image=clip[0],  # I2V / TI2V conditioning frame
            vace_video=clip,  # VACE control video
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            seed=args.seed,
            num_inference_steps=4,
        )

    # Dump every tensor field (fp32 cpu) + scalar fields for a bit-level compare.
    tensors = {k: v.detach().to(torch.float32).cpu() for k, v in inputs_shared.items() if torch.is_tensor(v)}
    scalars = {
        k: v for k, v in inputs_shared.items() if not torch.is_tensor(v) and isinstance(v, (int, float, bool, str))
    }
    # A single concatenated tensor lets compare.py reuse its ``output`` path; we
    # also keep the per-key shapes for diagnosis.
    flat = torch.cat([t.reshape(-1) for t in (tensors[k] for k in sorted(tensors))]) if tensors else torch.zeros(1)
    sd_keys = list(bb.state_dict().keys())
    payload = {
        "name": args.name,
        "output": flat,
        "output_shape": tuple(flat.shape),
        "tensor_keys": {k: tuple(tensors[k].shape) for k in sorted(tensors)},
        "scalars": scalars,
        "has_pipe_prefix": any(k.startswith("_pipe.") for k in sd_keys),
        "sd_key_sample": sorted(sd_keys)[:8],
    }
    torch.save(payload, args.out)
    print(f"[{args.name}] deploy dump → {args.out}: tensor_keys={payload['tensor_keys']} scalars={scalars}")


if __name__ == "__main__":
    main()
