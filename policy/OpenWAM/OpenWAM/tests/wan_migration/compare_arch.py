"""Compare two run_arch_compare.py dumps for main-vs-refactor equivalence.

Gate: forward predictions bit-identical (atol=0 preferred), action-backbone init
fingerprint identical (else random-init mismatch — investigate seeding before
reading the forward diff), loss scalars within tolerance, and no ``_pipe.*``
state_dict prefix on the refactor dump.

    python tests/wan_migration/compare_arch.py OLD.pt NEW.pt [--atol 0] [--loss-atol 0]
"""

import argparse
import sys

import torch


def _cmp_tensor(name, a, b, atol):
    if a is None and b is None:
        print(f"  --   {name}: both None (skip)")
        return True
    if (a is None) != (b is None):
        print(f"  FAIL {name}: one side None (old={a is not None} new={b is not None})")
        return False
    if tuple(a.shape) != tuple(b.shape):
        print(f"  FAIL {name}: shape mismatch old={tuple(a.shape)} new={tuple(b.shape)}")
        return False
    if torch.equal(a, b):
        print(f"  OK   {name}: bit-identical (atol=0), shape={tuple(a.shape)}")
        return True
    diff = (a - b).abs()
    mx = diff.max().item()
    idx = torch.unravel_index(diff.argmax(), diff.shape)
    rel = mx / (a.abs().max().item() + 1e-12)
    n = int((diff > 0).sum().item())
    status = "OK  " if mx <= atol else "FAIL"
    print(f"  {status} {name}: max_abs={mx:.3e} @ {tuple(int(i) for i in idx)} rel={rel:.3e} ndiff={n}/{a.numel()}")
    return mx <= atol


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--atol", type=float, default=0.0, help="forward tensor tolerance")
    ap.add_argument("--loss-atol", type=float, default=0.0, help="loss scalar tolerance")
    args = ap.parse_args()

    old = torch.load(args.old, map_location="cpu")
    new = torch.load(args.new, map_location="cpu")
    ok = True

    print(f"registry: old={old.get('registry_name')} new={new.get('registry_name')}")
    print(f"geom:     old={old.get('geom')} new={new.get('geom')}")

    if new.get("has_pipe_prefix"):
        print("  FAIL no_pipe_prefix: refactor dump carries _pipe.* state_dict keys")
        ok = False
    else:
        print("  OK   no_pipe_prefix (refactor)")

    afp_o, afp_n = old.get("action_fp"), new.get("action_fp")
    if afp_o and afp_n:
        same = abs(afp_o["sum"] - afp_n["sum"]) < 1e-6 and afp_o["numel"] == afp_n["numel"]
        note = "" if same else "  <- random-init MISMATCH: forward diff likely from init, not graph"
        print(
            f"  {'OK  ' if same else 'WARN'} action_fp: old_sum={afp_o['sum']:.6f} "
            f"new_sum={afp_n['sum']:.6f} numel={afp_o['numel']}/{afp_n['numel']}{note}"
        )

    if "video_pred" in old and "video_pred" in new:
        ok &= _cmp_tensor("video_pred", old["video_pred"], new["video_pred"], args.atol)
        ok &= _cmp_tensor("action_pred", old.get("action_pred"), new.get("action_pred"), args.atol)

    for k in ("loss", "loss_video", "loss_action"):
        if k in old and k in new:
            d = abs(old[k] - new[k])
            status = "OK  " if d <= args.loss_atol else "FAIL"
            if d > args.loss_atol:
                ok = False
            print(f"  {status} {k}: old={old[k]:.8f} new={new[k]:.8f} |delta|={d:.3e}")

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
