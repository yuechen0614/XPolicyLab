"""Tests for EgoDexDataset (HF lerobot v3 reader with action loss disabled).

Covers (no GPU / no real EgoDex dataset required):
  - EgoDexDataset.__init__:        schema fields loaded, len, action/state dim hardcoded
  - EgoDexDataset.__getitem__:     canonical sample-dict keys, action=0 + mask=False,
                                     proprio forced to zeros + proprio_mask all False
  - splits resolution:               info.json["splits"] honored (val_ratio removed)
  - Registry:                        deprecated reader intentionally not registered

The dataset-level tests use a minimal fake HF v3 layout (parquet + episodes
parquet + tasks parquet + info.json) and patch
``openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames`` so no real mp4 files
are needed.
"""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers (build a minimal EgoDex HF v3 layout on disk for fixture tests)
# ---------------------------------------------------------------------------


def _make_info_json(
    *,
    n_total: int,
    splits: dict[str, str] | None = None,
    fps: int = 30,
) -> dict:
    """Minimal info.json with EgoDex conversion-time schema (24-D)."""
    info = {
        "codebase_version": "v3.0",
        "robot_type": "human_hand_ego",
        "fps": fps,
        "total_episodes": n_total,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "action": {"dtype": "float32", "shape": [24], "names": None},
            "observation.state": {"dtype": "float32", "shape": [24], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    if splits is not None:
        info["splits"] = splits
    return info


def _identity_24() -> np.ndarray:
    """24-D state with identity rotation matrices and (0.1*hand_id, ...) xyz."""
    s = np.zeros(24, dtype=np.float32)
    s[0:3] = [0.1, 0.2, 0.3]  # xyz_l
    s[3:12] = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # identity rotmat_l (row-major)
    s[12:15] = [0.4, 0.5, 0.6]  # xyz_r
    s[15:24] = [1, 0, 0, 0, 1, 0, 0, 0, 1]  # identity rotmat_r
    return s


def _build_fake_egodex_root(
    base_dir: str,
    *,
    n_episodes: int = 3,
    ep_len: int = 50,
    n_tasks: int = 2,
    splits: dict[str, str] | None = None,
    target_camera: str = "observation.images.ego",
) -> str:
    """Build a minimal EgoDex HF v3 layout under ``base_dir``.

    Writes:
      meta/info.json          (with 24-D schema + optional splits)
      meta/tasks.parquet      (index = task string, column = task_index)
      meta/episodes/file-000.parquet
      data/chunk-000/file-000.parquet  (rows = sum(ep_len) per episode)

    No mp4 files; tests patch ``_decode_video_frames`` upstream.
    """
    meta_dir = os.path.join(base_dir, "meta")
    os.makedirs(os.path.join(meta_dir, "episodes"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "data", "chunk-000"), exist_ok=True)

    # --- info.json ---
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(_make_info_json(n_total=n_episodes, splits=splits), f)

    # --- tasks.parquet (HF v3 layout: index=task string, column=task_index) ---
    task_strs = [f"do task #{i}" for i in range(n_tasks)]
    tasks_df = pd.DataFrame(
        {"task_index": list(range(n_tasks))},
        index=pd.Index(task_strs, name="task"),
    )
    tasks_df.to_parquet(os.path.join(meta_dir, "tasks.parquet"))

    # --- episodes.parquet ---
    ep_rows = []
    cum_from = 0
    for ep in range(n_episodes):
        ep_rows.append(
            {
                "episode_index": ep,
                "tasks": np.array([task_strs[ep % n_tasks]], dtype=object),
                "length": ep_len,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{target_camera}/chunk_index": 0,
                f"videos/{target_camera}/file_index": 0,
                "dataset_from_index": cum_from,
                "dataset_to_index": cum_from + ep_len,
            }
        )
        cum_from += ep_len
    pd.DataFrame(ep_rows).to_parquet(os.path.join(meta_dir, "episodes", "file-000.parquet"), index=False)

    # --- data/chunk-000/file-000.parquet ---
    # Each row carries observation.state (24-D) + minimal metadata.
    # Use identity rotmat so the rot6d transform is deterministic.
    state_template = _identity_24()
    n_total_frames = n_episodes * ep_len
    rng = np.random.default_rng(42)
    state_arr = np.empty((n_total_frames, 24), dtype=np.float32)
    for i in range(n_total_frames):
        ep_id = i // ep_len
        # Perturb xyz by small per-frame offset; keep rotmat identity.
        s = state_template.copy()
        s[0:3] += 0.001 * rng.standard_normal(3).astype(np.float32) + 0.01 * ep_id
        s[12:15] += 0.001 * rng.standard_normal(3).astype(np.float32) + 0.01 * ep_id
        state_arr[i] = s

    episode_indices = np.repeat(np.arange(n_episodes), ep_len)
    frame_indices = np.tile(np.arange(ep_len), n_episodes)
    task_indices = (episode_indices % n_tasks).astype(np.int64)

    table = pa.table(
        {
            "episode_index": pa.array(episode_indices.astype(np.int64), type=pa.int64()),
            "frame_index": pa.array(frame_indices.astype(np.int64), type=pa.int64()),
            "task_index": pa.array(task_indices, type=pa.int64()),
            "observation.state": pa.FixedSizeListArray.from_arrays(
                pa.array(state_arr.reshape(-1), type=pa.float32()), 24
            ),
            # Conversion writes `action` too; we include it so the parquet schema
            # is realistic (reader will not read it because it is not in
            # ``_NEEDED_COLS``).
            "action": pa.FixedSizeListArray.from_arrays(pa.array(state_arr.reshape(-1), type=pa.float32()), 24),
        }
    )
    pq.write_table(table, os.path.join(base_dir, "data", "chunk-000", "file-000.parquet"))

    # Empty videos directory (decoder is patched in fixture tests).
    os.makedirs(os.path.join(base_dir, "videos", target_camera, "chunk-000"), exist_ok=True)

    return base_dir


@contextmanager
def _patched_video_decoder(height: int = 384, width: int = 320):
    """Patch ``_decode_video_frames`` to return black PIL frames at (h, w)."""

    def _fake_decode(path: str, frame_indices, h: int, w: int):
        return [Image.new("RGB", (w, h)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", _fake_decode):
        yield


# ---------------------------------------------------------------------------
# Registry / from_config wiring
# ---------------------------------------------------------------------------


def test_registry_does_not_register_deprecated_egodex():
    """Moving EgoDex under deprecated also removes it from active dispatch."""
    import openwam.dataloader  # noqa: F401
    from openwam.dataloader.registry import DATASET_REGISTRY

    assert "egodex" not in DATASET_REGISTRY


def test_build_dataset_rejects_deprecated_egodex():
    """The active registry must reject deprecated dataset types."""
    from openwam.dataloader import build_dataset

    with pytest.raises(ValueError, match="Unknown dataset type 'egodex'"):
        build_dataset({"type": "egodex"}, split="train")


# ---------------------------------------------------------------------------
# Dataset construction + dim contract
# ---------------------------------------------------------------------------


def test_init_loads_meta_and_reports_action_dim_20():
    """Reader hardcodes action/state dim=20 regardless of info.json's 24-D schema."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )

        assert ds.action_dim == 20
        assert ds._ACTION_DIM == 20
        assert ds._STATE_DIM == 20


def test_len_matches_window_formula():
    """train: len(ds) = sum(max(0, ep_len - (video_stride+1)) // window_stride + 1).

    Tighter than robotwin's 2-frame minimum: EgoDex requires actual_raw_len
    >= video_stride+1 so video_mask has at least 2 True slots (t=0 reference
    + t=1 first prediction). Single-REAL-frame windows are excluded because
    EgoDex is video-only supervised — they carry too weak a learning signal.
    """
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    n_eps = 3
    ep_len = 50
    num_frames = 33
    video_stride = 4
    window_stride = 1
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=n_eps, ep_len=ep_len)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=num_frames,
                window_stride=window_stride,
                video_stride=video_stride,
                height=64,
                width=64,
                split="train",
            )

        # train: max_start = max(0, ep_len - (video_stride + 1))
        expected = n_eps * (max(0, ep_len - (video_stride + 1)) // window_stride + 1)
        assert len(ds) == expected


def test_len_val_split_uses_strict_full_windows():
    """val: max_start = max(0, ep_len - num_frames) — strict full windows only."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    n_eps = 4
    ep_len = 50
    num_frames = 33
    # Use info.json splits to deterministically put all episodes into val.
    splits = {"train": "0:0", "val": f"0:{n_eps}"}
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=n_eps, ep_len=ep_len, splits=splits)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                split="val",
                num_frames=num_frames,
                video_stride=4,
                window_stride=1,
                height=64,
                width=64,
            )

        expected = n_eps * (max(0, ep_len - num_frames) + 1)
        assert len(ds) == expected


# ---------------------------------------------------------------------------
# __getitem__ contract (the load-bearing test)
# ---------------------------------------------------------------------------


_EXPECTED_KEYS = {
    "video",
    "vace_video",
    "first_frame_image",
    "action",
    "action_mask",
    "video_mask",
    "proprio",
    "proprio_mask",
    "prompt",
}


def test_getitem_dict_keys_match_canonical_schema():
    """Sample dict must carry the exact key set that mixture batch collate expects."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )
            s = ds[0]

        assert set(s.keys()) == _EXPECTED_KEYS


def test_getitem_action_is_zero_and_mask_all_false():
    """The core contract: action loss is disabled at the reader level."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    num_frames = 33
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=num_frames,
                video_stride=4,
                height=64,
                width=64,
            )
            s = ds[0]

        # action: (T-1, 20) all zeros
        assert s["action"].shape == (num_frames - 1, 20)
        assert s["action"].dtype == torch.float32
        assert s["action"].abs().sum().item() == 0.0
        # action_mask: (T-1, 20) 2D bool, all False (post 2-D mask migration —
        # EgoDex still disables action loss permanently).
        assert s["action_mask"].shape == (num_frames - 1, 20)
        assert s["action_mask"].dtype == torch.bool
        assert s["action_mask"].any().item() is False


def test_getitem_proprio_is_zero_and_mask_false():
    """EgoDex samples emit proprio=zeros(1, 20) and proprio_mask all False.

    Camera-frame state cannot be aligned with the robot base-frame EEF schema,
    so the reader's contract is: zero proprio + mask=False. The model layer
    consumes proprio_mask and isolates these samples from proprio_encoder
    gradient updates.
    """
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )
            s = ds[0]

        assert s["proprio"].shape == (1, 20)
        assert s["proprio"].dtype == torch.float32
        assert s["proprio"].abs().sum().item() == 0.0

        # proprio_mask: (1, 20) 2D bool, all False (post 2-D mask migration —
        # camera-frame 24-D state cannot be aligned with base-frame 20-D EEF).
        assert s["proprio_mask"].shape == (1, 20)
        assert s["proprio_mask"].dtype == torch.bool
        assert bool(s["proprio_mask"].any().item()) is False


def test_getitem_video_shape_and_count():
    """video: List[PIL.Image] of length ceil(num_frames/video_stride) (VACE rounding)."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    height, width = 384, 320
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=height, width=width):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=height,
                width=width,
            )
            s = ds[0]

        # num_frames=33, video_stride=4 → ceil(33/4)=9, 9 % 4 == 1 ✓ → 9 frames
        assert len(s["video"]) == 9
        assert all(isinstance(f, Image.Image) for f in s["video"])
        assert all(f.size == (width, height) for f in s["video"])
        # video_mask: shape (T_video,), all True since fixture has full-length episode
        assert s["video_mask"].shape == (9,)
        assert bool(s["video_mask"].all().item()) is True


def test_getitem_prompt_from_tasks_parquet():
    """prompt comes from tasks.parquet via task_index lookup, raw stripped text (no wrapper)."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40, n_tasks=2)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )
            s = ds[0]

        # v2.5: no wrapper prefix — emit raw tasks.parquet text as-is.
        assert s["prompt"] in ("do task #0", "do task #1")
        assert not s["prompt"].startswith("A video"), "EgoDex prompt must not carry any wrapper prefix (v2.5 decision)"


# ---------------------------------------------------------------------------
# v2.5 prompt refactor: _resolve_egodex_prompt pure function
# ---------------------------------------------------------------------------


def test_resolve_egodex_prompt_normal():
    """Normal lookup → returns the stripped raw string, no wrapper."""
    from openwam.dataloader.deprecated.egodex import _resolve_egodex_prompt

    lut = {0: "Make a sandwich.", 1: "Stack the cups."}
    assert _resolve_egodex_prompt(lut, 0) == "Make a sandwich."
    assert _resolve_egodex_prompt(lut, 1) == "Stack the cups."


def test_resolve_egodex_prompt_missing_task_index_raises():
    """Missing task_index key → KeyError (strict fail, no fallback)."""
    from openwam.dataloader.deprecated.egodex import _resolve_egodex_prompt

    lut = {0: "Make a sandwich."}
    with pytest.raises(KeyError, match="not present"):
        _resolve_egodex_prompt(lut, 99)


def test_resolve_egodex_prompt_empty_text_raises():
    """task_index maps to empty string → ValueError (strict fail, no fallback)."""
    from openwam.dataloader.deprecated.egodex import _resolve_egodex_prompt

    lut = {0: "", 1: "   "}
    with pytest.raises(ValueError, match="empty prompt"):
        _resolve_egodex_prompt(lut, 0)
    with pytest.raises(ValueError, match="empty prompt"):
        _resolve_egodex_prompt(lut, 1)


def test_resolve_egodex_prompt_strips_whitespace():
    """Surrounding whitespace in raw entry is stripped."""
    from openwam.dataloader.deprecated.egodex import _resolve_egodex_prompt

    lut = {0: "  Make a sandwich.  \n"}
    assert _resolve_egodex_prompt(lut, 0) == "Make a sandwich."


# ---------------------------------------------------------------------------
# v2.5 multiview L-shape canvas
# ---------------------------------------------------------------------------


@contextmanager
def _patched_video_decoder_colored(height: int = 384, width: int = 320, color=(180, 60, 40)):
    """Patched decoder returning a non-black colored PIL frame.

    Used by multiview tests where we need to distinguish the top (filled)
    region from the bottom (genuinely black, no wrist camera) region.
    """

    def _fake_decode(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), color) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", _fake_decode):
        yield


def test_multiview_false_outputs_single_camera_size():
    """multiview=false → frame size matches (width, height); zero-risk fallback."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=256, width=320):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=256,
                width=320,
                multiview=False,
            )
            s = ds[0]

        # PIL Image .size returns (W, H).
        assert s["video"][0].size == (320, 256)


def test_multiview_true_outputs_384x320_with_black_wrist_slots():
    """multiview=true → 384H × 320W canvas with bottom 128 rows fully black (no wrist)."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder_colored(color=(200, 100, 50)):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=384,
                width=320,
                multiview=True,
            )
            s = ds[0]

        # PIL Image .size returns (W, H).
        assert s["video"][0].size == (320, 384), (
            f"multiview canvas should be (320, 384) PIL (W, H); got {s['video'][0].size}"
        )
        arr = np.array(s["video"][0])
        # arr shape: (H=384, W=320, 3). Top 256 rows have the colored frame
        # (stretched); bottom 128 rows must be solid black (RGB == 0).
        bottom = arr[256:, :, :]
        assert bottom.max() == 0, (
            f"Bottom 128 rows must be all-black (no wrist cameras), but max pixel = {bottom.max()}"
        )
        top = arr[:256, :, :]
        # Top should be the colored fill (or near it after BILINEAR resize).
        assert top.mean() > 30, f"Top 256 rows should carry the ego frame fill (mean > 30), got {top.mean():.2f}"


def test_multiview_true_default_camera_layout():
    """When camera_layout omitted, default places target_camera at top + two missing slots."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=1, ep_len=40)
        with _patched_video_decoder_colored():
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=384,
                width=320,
                multiview=True,
                # Don't pass camera_layout — default should be applied.
            )

        assert ds._camera_layout[0] == "observation.images.ego"
        assert "__missing" in ds._camera_layout[1]
        assert "__missing" in ds._camera_layout[2]


# ---------------------------------------------------------------------------
# Boundary / tail-window mask handling
# ---------------------------------------------------------------------------


def test_tail_window_video_mask_marks_padded_positions_false():
    """Tail windows shorter than num_frames must produce a per-position mask
    where padded slots are False — matching robotwin_dataset:860-863.

    Setup: ep_len=42, num_frames=33, video_stride=4
      _video_sample_indices = [0, 4, 8, 12, 16, 20, 24, 28, 32]
      Window at offset=30 covers raw frames [30, 42), actual_raw_len=12.
      Mask = [i < 12 for i in indices]
           = [T, T, T, F, F, F, F, F, F]   (only frames 0,4,8 are real)
    """
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=1, ep_len=42)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                window_stride=1,
                height=64,
                width=64,
                split="train",
            )
            # train: max_start = ep_len - (video_stride+1) = 42 - 5 = 37, 38 windows.
            # idx=30 corresponds to offset=30 (window_stride=1); actual_raw_len = 12.
            s = ds[30]

        assert s["video_mask"].shape == (9,)
        expected_mask = torch.tensor(
            [True, True, True, False, False, False, False, False, False],
            dtype=torch.bool,
        )
        assert torch.equal(s["video_mask"], expected_mask), (
            f"Expected {expected_mask.tolist()}, got {s['video_mask'].tolist()}"
        )
        # video itself is still num_video_frames long (last real frame padded
        # to fill the slots).
        assert len(s["video"]) == 9


def test_tail_window_minimum_has_two_real_frames():
    """The most extreme tail window EgoDex emits has exactly 2 REAL video
    frames — t=0 (reference) + t=1 (first prediction step).

    Single-REAL windows (actual_raw_len < video_stride+1) are deliberately
    excluded by the train enumerator because EgoDex is video-only supervised
    and 1 real frame carries too weak a learning signal.

    Setup: ep_len=37, num_frames=33, video_stride=4, window_stride=1
      max_start = 37 - (video_stride+1) = 32  → 33 windows (offset 0..32)
      idx=32 → offset=32, actual_raw_len = min(33, 37-32) = 5
      Mask = [0<5, 4<5, 8<5, ...] = [T, T, F, F, F, F, F, F, F]
    """
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=1, ep_len=37)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                window_stride=1,
                height=64,
                width=64,
                split="train",
            )
            assert len(ds) == 33  # offsets 0..32
            s = ds[32]

        assert s["video_mask"].tolist() == [True, True, False, False, False, False, False, False, False]


def test_train_enumerator_excludes_actual_raw_len_lt_video_stride_plus_one():
    """Windows with actual_raw_len < video_stride+1 must not appear in train."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    # ep_len=35: with old policy 34 windows; with new policy max_start = 35-5 = 30 → 31 windows.
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=1, ep_len=35)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                window_stride=1,
                height=64,
                width=64,
                split="train",
            )
            assert len(ds) == 31  # offsets 0..30, never 31/32/33
            # The last emitted window must have actual_raw_len = 5 → 2 REAL slots.
            # idx=30 → offset=30 → mask [T, T, F×7]
            s = ds[30]
            assert s["video_mask"].tolist() == [True, True, False, False, False, False, False, False, False]
            # And every emitted window has at least 2 True in video_mask.
            for i in range(len(ds)):
                assert int(ds[i]["video_mask"].sum().item()) >= 2


def test_tail_window_does_not_overflow_into_next_episode_in_mp4():
    """Critical safety: tail windows must not request frame indices that
    cross the current episode's end. EgoDex packs multiple episodes into
    one mp4 (see _ep_video_frame_offset), so an out-of-bounds frame index
    would silently return the NEXT episode's frames. The fix in
    __getitem__ clamps frame_indices to actual_raw_len before calling
    _decode_video_frames.
    """
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    captured = []

    def _spy_decode(path, frame_indices, h, w):
        captured.append(list(frame_indices))
        return [Image.new("RGB", (w, h)) for _ in frame_indices]

    with tempfile.TemporaryDirectory() as tmp:
        # 3 episodes of length 40, all in the same mp4 file:
        #   ep0 covers file-frames [0, 40)
        #   ep1 covers file-frames [40, 80)
        #   ep2 covers file-frames [80, 120)
        _build_fake_egodex_root(tmp, n_episodes=3, ep_len=40)
        with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", _spy_decode):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                window_stride=1,
                height=64,
                width=64,
                split="train",
            )
            # ep0 has max_start = ep_len - (video_stride+1) = 35, i.e., 36 windows.
            # idx=35 is the last (most extreme tail under the new policy).
            # Naive strided indexing would request up to 35+32=67 — silently
            # reading ep1 frames (ep1 starts at file-frame 40). The fix
            # truncates frame_indices to actual_raw_len=5.
            ds[35]

        assert len(captured) == 1, f"expected 1 decoder call, got {len(captured)}"
        requested = captured[0]
        # offset=35 → actual_raw_len=5 → _video_sample_indices[0,1] = [0,4] are real
        # → file indices = [0+35+0, 0+35+4] = [35, 39], both inside ep0's [0, 40).
        assert all(0 <= idx < 40 for idx in requested), (
            f"Tail window leaked into next episode: requested {requested} but ep0 only owns file-frames [0, 40)."
        )


def test_getitem_vace_and_first_frame_passthrough():
    """vace_video is None (egodex has no VACE input); first_frame_image is video[0]."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )
            s = ds[0]

        assert s["vace_video"] is None
        assert len(s["first_frame_image"]) == 1
        assert s["first_frame_image"][0] is s["video"][0]


# ---------------------------------------------------------------------------
# splits resolution
# ---------------------------------------------------------------------------


def test_splits_honor_info_json():
    """When info.json carries explicit splits, the split spec drives the filter."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    n_eps = 5
    splits = {"train": "0:3", "val": "3:5"}
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=n_eps, ep_len=40, splits=splits)
        with _patched_video_decoder(height=64, width=64):
            ds_train = EgoDexDataset(
                dataset_dir=tmp,
                split="train",
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )
            ds_val = EgoDexDataset(
                dataset_dir=tmp,
                split="val",
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )

        assert len(ds_train._eps_df) == 3  # 0,1,2
        assert len(ds_val._eps_df) == 2  # 3,4
        # train: max(0, 40 - (video_stride+1)) + 1 = max(0, 40-5)+1 = 36 windows per ep
        # val:   max(0, 40 - num_frames) + 1     = max(0, 40-33)+1 = 8  windows per ep
        assert len(ds_train) == 3 * 36
        assert len(ds_val) == 2 * 8


def test_splits_fallback_train_returns_full_when_no_splits_field():
    """No info.json splits + split='train' → all episodes."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    n_eps = 5
    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=n_eps, ep_len=40, splits=None)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                split="train",
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )

        assert len(ds._eps_df) == n_eps


# ---------------------------------------------------------------------------
# pickle / multiprocessing-worker compatibility
# ---------------------------------------------------------------------------


def test_pickle_roundtrip_drops_lru_cache():
    """__getstate__ removes the lru_cache; __setstate__ rebuilds it."""
    import pickle

    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )

            # __getstate__ should not contain the lru_cache (functools wrappers
            # are not pickleable).
            state = ds.__getstate__()
            assert "_load_data_table" not in state

            # Roundtrip through pickle and confirm sample retrieval still works.
            blob = pickle.dumps(ds)
            ds2 = pickle.loads(blob)
            s = ds2[0]
            assert set(s.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Cross-source mixture compatibility — batch collate without error
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Root mode: from_config(<root>) → MultiBucketEgoDexDataset
# ---------------------------------------------------------------------------


def test_from_config_root_mode_fan_out():
    """``from_config({dataset_dir: <root>})`` discovers sub-buckets and wraps them.

    Builds a fake root containing 3 mini buckets (each with their own
    ``meta/info.json``) and confirms that:
      - ``from_config`` returns a ``MultiBucketEgoDexDataset`` (not a single
        ``EgoDexDataset``)
      - ``len(ds) == sum(len(sub_bucket))``
      - sample dict from the wrapper has the exact EgoDex key set
      - ``action_dim`` / ``normalization_stats`` are delegated to the first bucket
      - bisect-style index routing lands in the correct sub-bucket
    """
    from openwam.dataloader.deprecated.egodex import (
        EgoDexDataset,
        MultiBucketEgoDexDataset,
    )

    bucket_specs = [
        ("task_a", 3, 40),
        ("task_b", 2, 50),
        ("task_c", 4, 40),
    ]

    with tempfile.TemporaryDirectory() as tmp_root:
        for name, n_eps, ep_len in bucket_specs:
            _build_fake_egodex_root(
                os.path.join(tmp_root, name),
                n_episodes=n_eps,
                ep_len=ep_len,
            )

        cfg = {
            "type": "egodex",
            "dataset_dir": tmp_root,
            "num_frames": 33,
            "video_stride": 4,
            "height": 64,
            "width": 64,
        }
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset.from_config(cfg, split="train")

            assert isinstance(ds, MultiBucketEgoDexDataset)
            assert not isinstance(ds, EgoDexDataset)
            assert len(ds.buckets) == 3
            assert all(isinstance(b, EgoDexDataset) for b in ds.buckets)

            # len(ds) == sum(len(sub))
            assert len(ds) == sum(len(b) for b in ds.buckets)
            # train: max_start = max(0, ep_len - (video_stride+1)) = max(0, ep_len - 5)
            # → max(0, ep_len - 5) + 1 windows per episode
            expected_per_bucket = [n_eps * (max(0, ep_len - 5) + 1) for _, n_eps, ep_len in bucket_specs]
            assert [len(b) for b in ds.buckets] == expected_per_bucket

            # Sample dict is the EgoDex shape (delegated through wrapper).
            s = ds[0]
            assert set(s.keys()) == _EXPECTED_KEYS
            assert s["action"].shape == (32, 20)
            assert s["proprio"].shape == (1, 20)
            assert len(s["video"]) == 9

            # action_dim / normalization_stats come from the first bucket.
            assert ds.action_dim == 20
            assert ds.normalization_stats is None

            # Drain one idx at the boundary of each bucket; np.searchsorted
            # must route them into 3 distinct buckets (per-bucket folder name
            # is on the dataset instance, not in the sample dict).
            seen_bucket_ids = set()
            cum = 0
            for _, n_eps, ep_len in bucket_specs:
                bi = int(np.searchsorted(ds._cum_lens, cum, side="right") - 1)
                seen_bucket_ids.add(ds.buckets[bi]._dataset_id)
                cum += n_eps * (max(0, ep_len - 5) + 1)
            assert seen_bucket_ids == {"task_a", "task_b", "task_c"}


def test_from_config_root_mode_empty_raises():
    """A root directory with no sub-buckets and no meta/info.json should error clearly."""
    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "type": "egodex",
            "dataset_dir": tmp,
            "num_frames": 33,
            "video_stride": 4,
            "height": 64,
            "width": 64,
        }
        with pytest.raises(FileNotFoundError):
            EgoDexDataset.from_config(cfg, split="train")


def test_default_collate_stacks_two_egodex_samples():
    """Two EgoDex samples must stack cleanly through ``default_collate``.

    Sanity check on the tensor / list / None mix in the sample dict; the
    trainer strips list/None fields before collate the same way.
    """
    from torch.utils.data._utils.collate import default_collate

    from openwam.dataloader.deprecated.egodex import EgoDexDataset

    with tempfile.TemporaryDirectory() as tmp:
        _build_fake_egodex_root(tmp, n_episodes=2, ep_len=40)
        with _patched_video_decoder(height=64, width=64):
            ds = EgoDexDataset(
                dataset_dir=tmp,
                num_frames=33,
                video_stride=4,
                height=64,
                width=64,
            )
            s0, s1 = ds[0], ds[1]

        collate_keys = {k for k, v in s0.items() if not isinstance(v, list) and v is not None}
        batch = default_collate([{k: s0[k] for k in collate_keys}, {k: s1[k] for k in collate_keys}])

        assert batch["action"].shape == (2, 32, 20)
        # 2-D mask: (B, T_action, action_dim)
        assert batch["action_mask"].shape == (2, 32, 20)
        assert batch["action_mask"].any().item() is False
        assert batch["proprio"].shape == (2, 1, 20)
        # 2-D mask: (B, 1, action_dim)
        assert batch["proprio_mask"].shape == (2, 1, 20)
        assert batch["video_mask"].shape == (2, 9)
