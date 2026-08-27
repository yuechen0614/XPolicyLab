"""Compare two run_backbone.py dumps for bit-level output equivalence.

Acceptance gate (migration is done only when this passes for all 3 variants):
  - identical output shape (no layout/transpose drift),
  - element-wise equality (atol=0) preferred; otherwise a very tight tolerance
    with the max-abs-diff LOCATION reported so misalignment (transpose, token
    reorder, off-by-one) is caught rather than hidden behind a norm,
  - migrated dump must carry no ``_pipe.*`` state_dict keys.

    python tests/wan_migration/compare.py OLD.pt NEW.pt [--atol 1e-6]
"""

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--atol", type=float, default=0.0, help="fallback tolerance if not bit-identical")
    args = ap.parse_args()

    old = torch.load(args.old, map_location="cpu")
    new = torch.load(args.new, map_location="cpu")

    ok = True

    if new.get("has_pipe_prefix"):
        print(f"FAIL: migrated dump has _pipe.* state_dict keys: {new.get('sd_key_sample')}")
        ok = False
    else:
        print(f"OK: no _pipe.* prefix; sample keys={new.get('sd_key_sample')}")

    a, b = old["output"], new["output"]
    if tuple(a.shape) != tuple(b.shape):
        print(f"FAIL: output shape mismatch old={tuple(a.shape)} new={tuple(b.shape)}")
        return 1

    if torch.equal(a, b):
        print(f"OK: bit-identical output (atol=0), shape={tuple(a.shape)}")
        return 0 if ok else 1

    diff = (a - b).abs()
    max_abs = diff.max().item()
    argmax = torch.unravel_index(diff.argmax(), diff.shape)
    rel = max_abs / (a.abs().max().item() + 1e-12)
    n_diff = int((diff > 0).sum().item())
    print(
        f"NOT bit-identical: max_abs_diff={max_abs:.3e} at index={tuple(int(i) for i in argmax)}, "
        f"rel={rel:.3e}, n_differing_elems={n_diff}/{a.numel()}"
    )
    if max_abs <= args.atol:
        print(f"PASS within atol={args.atol} (investigate the nondeterminism source before relaxing further)")
        return 0 if ok else 1
    print(f"FAIL: exceeds atol={args.atol} — likely a real bug or a 错位 (misalignment).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
