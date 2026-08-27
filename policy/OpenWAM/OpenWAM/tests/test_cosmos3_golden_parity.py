"""GPU golden parity: the batched cosmos3 engine vs the native Cosmos3OmniPipeline.

Replays the exact cond-pass transformer inputs captured from a native pipeline
run (``cosmos3_golden/golden_step0.pt``, produced by the Phase-0 spike on real
Cosmos3-Edge weights) through OpenWAM's batched engine and compares:

1. mRoPE position ids built by ``text_pack`` vs the pipeline's,
2. gen-stream hidden states at the dumped layers {0, 1, 13, 27},
3. the final velocity field on noisy frames.

bf16 end-to-end. Measured bit-identical to the native forward on H20, but the
tolerances are sized for a different SDPA kernel being selected elsewhere
(FLASH vs MATH already differs by rel_max 4.4e-3 on one op at this geometry),
not for the measured zero.
"""

import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu

ASSET_PATH = Path(os.environ.get("COSMOS3_EDGE_ASSET_PATH", "/mnt/cpfs/zch/assets/Cosmos3-Edge"))
GOLDEN_PATH = Path(os.environ.get("COSMOS3_GOLDEN_PATH", "/mnt/cpfs/zch/assets/cosmos3_golden/golden_step0.pt"))


def _skip_unless_runnable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if not ASSET_PATH.exists():
        pytest.skip(f"Cosmos3-Edge bundle missing at {ASSET_PATH}.")
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Golden dump missing at {GOLDEN_PATH} (run the Phase-0 golden script).")
    pytest.importorskip("diffusers")


@pytest.fixture(scope="module")
def golden():
    _skip_unless_runnable()
    return torch.load(GOLDEN_PATH, map_location="cpu", weights_only=False)


@pytest.fixture(scope="module")
def net():
    """The transformer as PRODUCTION sees it: built through the backbone and
    moved with ``set_dtype_device``.

    Going through the backbone matters. A bare
    ``from_pretrained(torch_dtype=bf16)`` leaves the rotary ``inv_freq`` buffer
    in fp32, while the train and deploy paths both call ``set_dtype_device``,
    whose blanket ``.to(dtype=)`` used to round it — a discrepancy that let this
    harness report 0.999969 cosine while production ran degraded vision phases.
    Parity must be measured on the path that ships.
    """
    _skip_unless_runnable()
    from openwam.model.video_backbone import build_video_backbone

    cfg = {"model": {"video_backbone": {"name": "cosmos3_edge", "model_path": str(ASSET_PATH)}}}
    vb = build_video_backbone("cosmos3_edge", cfg)
    vb.set_dtype_device(torch.bfloat16, torch.device("cuda"))
    vb.eval()
    assert vb.dit.rotary_emb.inv_freq.dtype == torch.float32, (
        "set_dtype_device rounded the rotary frequency table; vision phases would be scrambled"
    )
    return vb.dit


def test_position_builder_matches_native(golden):
    from openwam.model.video_backbone.cosmos3 import text_pack

    k0 = golden["call0_kwargs"]
    und_len = int(k0["und_len"])
    grid = tuple(k0["vision_token_shapes"][0])
    text_pos, vis_pos = text_pack.build_joint_positions(
        und_len, grid, modality_margin=15000, fps=24.0, temporal_compression_factor=4
    )
    native = k0["position_ids"].float()
    mine = torch.cat([text_pos, vis_pos], dim=1)
    assert torch.allclose(mine, native, atol=1e-4), (mine - native).abs().max()


def test_engine_matches_native_on_real_weights(golden, net):
    from openwam.model.video_backbone.cosmos3 import dit_forward

    k0 = golden["call0_kwargs"]
    device = torch.device("cuda")
    dtype = torch.bfloat16

    und_len = int(k0["und_len"])
    input_ids = k0["input_ids"].to(device)
    pos = k0["position_ids"].to(device)
    text_pos, vis_pos = pos[:, :und_len], pos[:, und_len:]
    lat = k0["vision_tokens"][0].to(device=device, dtype=dtype)  # (1, 48, 8, 30, 52)
    t_val = float(k0["vision_timesteps"][0])
    noisy0 = int(k0["vision_noisy_frame_indexes"][0][0])  # first noisy frame (=1)

    layer_capture = {}
    with torch.no_grad():
        # und_mask=None is what an unpadded batch takes in production (the
        # native pipeline passes no mask either), so parity must be measured
        # on it — an all-True mask here would silently exercise a different
        # SDPA kernel than the one that ships.
        cos_und, sin_und = dit_forward.compute_rotary(net, text_pos.unsqueeze(1), device, dtype)
        _, und_kv = dit_forward.run_und_tower(net, input_ids.unsqueeze(0), None, cos_und, sin_und)
        state = dit_forward.prepare_block_loop(
            net,
            latents=lat,
            timestep=torch.tensor([t_val], device=device),
            context=torch.zeros(1, und_len, net.config.hidden_size, device=device, dtype=dtype),
            und_mask=None,
            und_kv=und_kv,
            vision_positions=vis_pos.unsqueeze(1),
            num_clean_prefix_frames=noisy0,
        )
        layer_capture["gen_in_0"] = state.hidden_states.clone()
        for i in range(len(net.layers)):
            state = dit_forward.run_block(net, i, state)
            if i in golden["layers"]:
                layer_capture[i] = state.hidden_states.clone()
        mine = dit_forward.finalize_block_loop(net, state)

    # Layer-0 input = patchify + proj_in + timestep scatter parity.
    native_gen_in0 = golden["layers"][0]["gen_in"].to(device=device, dtype=torch.float32)
    mine_gen_in0 = layer_capture["gen_in_0"][0].float()
    assert torch.allclose(mine_gen_in0, native_gen_in0, atol=5e-2), (mine_gen_in0 - native_gen_in0).abs().max()

    # Per-layer gen-stream outputs. On H20 this path is *bit-identical* to the
    # native forward at every dumped layer and on the velocity. The drift the
    # old ladder allowed (0.3% → 7.7% by depth, read at the time as bf16
    # reduction-order noise) was really the all-True und mask forcing SDPA off
    # the fused kernel the native pipeline uses; dropping it removed the last
    # numeric difference.
    #
    # The bounds are NOT set at the measured zero. A single attention op at this
    # geometry differs by rel_max 4.4e-3 / mse_ratio 6e-6 between FLASH and MATH
    # (and 9e-6 vs CUDNN) before 28 layers compound it, so bounds near zero
    # would fail on any machine that picks a different kernel and read as a real
    # regression. Calibration in the other direction: the inv_freq bug this
    # harness missed gave cosine 0.768 / mse_ratio ~1e-1, so these still catch
    # it by three orders of magnitude.
    for i in sorted(k for k in golden["layers"] if isinstance(k, int)):
        native_i = golden["layers"][i]["gen_out"].to(device=device, dtype=torch.float32)
        mine_i = layer_capture[i][0].float()
        max_diff = (mine_i - native_i).abs().max().item()
        rel = max_diff / (native_i.abs().max().item() + 1e-6)
        assert rel < 2e-2, f"layer {i}: max_diff={max_diff:.4f} rel={rel:.4f}"

    # Final velocity on noisy frames — the training-relevant hard gate.
    native_vel = golden["call0_velocity"][0].to(device=device, dtype=torch.float32)
    mine_vel = mine.float()
    m_noisy = mine_vel[:, :, noisy0:]
    n_noisy = native_vel[:, :, noisy0:]
    rel_max = (m_noisy - n_noisy).abs().max().item() / (n_noisy.abs().max().item() + 1e-6)
    cosine = torch.nn.functional.cosine_similarity(m_noisy.flatten(), n_noisy.flatten(), dim=0).item()
    mse_ratio = ((m_noisy - n_noisy).pow(2).mean() / n_noisy.pow(2).mean()).item()
    assert rel_max < 2e-2, f"velocity rel_max {rel_max:.6f}"
    assert cosine > 0.9999, f"velocity cosine {cosine:.8f}"
    assert mse_ratio < 1e-4, f"velocity mse_ratio {mse_ratio:.3e}"
    assert native_vel[:, :, :noisy0].abs().sum() == 0
