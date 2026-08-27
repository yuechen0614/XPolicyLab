"""Unit tests for OxeBczDataset.

Builds a minimal LeRobot v3 BC-Z-shaped bucket on disk (state + action
parquet columns, tasks.parquet, info.json) and exercises the __init__
and __getitem__ paths. Video decoding is mocked so the tests don't need
mp4 files; the mask / value contracts are verified end-to-end.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from openwam.dataloader.deprecated.oxe_bcz import OxeBczDataset
from openwam.dataloader.utils.eef import EEF_DIM

EP_LENGTH = 60
FPS = 10.0
HEAD_CAM = "observation.images.image"


def _write_data_shard(bucket: Path, n_rows: int) -> None:
    """Write BC-Z-shaped data parquet with state[8] + action[7].

    task_index is no longer consumed by the reader (prompts come from
    tasks_annotated.parquet) but we still write it for parity with the
    real BC-Z conversion.
    """
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    # state[8] = [x, y, z, roll, pitch, yaw, pad, gripper]
    state = rng.uniform(-1, 1, size=(n_rows, 8)).astype(np.float32)
    state[:, 6] = 0.0  # pad
    state[:, 7] = rng.uniform(0, 1, size=n_rows)  # gripper
    # action[7] = [x, y, z, roll, pitch, yaw, gripper]
    action = rng.uniform(-1, 1, size=(n_rows, 7)).astype(np.float32)
    action[:, 6] = rng.uniform(0, 1, size=n_rows)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _write_episodes_parquet(bucket: Path, n_episodes: int) -> None:
    eps_dir = bucket / "meta" / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cum = 0
    for ep in range(n_episodes):
        rows.append(
            {
                "episode_index": ep,
                "length": EP_LENGTH,
                "dataset_from_index": cum,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{HEAD_CAM}/chunk_index": 0,
                f"videos/{HEAD_CAM}/file_index": 0,
            }
        )
        cum += EP_LENGTH
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), eps_dir / "chunk-000.parquet")


def _write_tasks_parquet(bucket: Path) -> None:
    """Write the legacy sparse tasks.parquet (not consumed by the reader,
    kept for parity with the real LeRobot v3 layout)."""
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick the block"], name="task"))
    df.to_parquet(bucket / "meta" / "tasks.parquet")


def _write_tasks_annotated_parquet(bucket: Path, n_episodes: int) -> None:
    """Write tasks_annotated.parquet — per-episode LLM-rewritten prompts
    the reader actually consumes."""
    df = pd.DataFrame(
        {"task": [f"pick the block — episode {i}" for i in range(n_episodes)]},
        index=pd.Index(range(n_episodes), name="episode_index"),
    )
    df.to_parquet(bucket / "meta" / "tasks_annotated.parquet")


def _write_video_placeholder(bucket: Path) -> None:
    """Touch a placeholder mp4 so video_path.format(...) resolves to an existing file."""
    vid_dir = bucket / "videos" / HEAD_CAM / "chunk-000"
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "file-000.mp4").write_bytes(b"")  # mocked decoder ignores content


def _make_eef_stats(bucket: Path) -> None:
    """Write a min-max-friendly stats file so normalize_mode=quantile and min-max
    are both exercised cleanly."""
    stats = {
        "n_samples": EP_LENGTH * 2,  # state + action
        "n_state_samples": EP_LENGTH,
        "n_action_samples": EP_LENGTH,
        "min": [-1.0] * 10,
        "max": [1.0] * 10,
        "mean": [0.0] * 10,
        "std": [0.5] * 10,
        "q01": [-0.9] * 10,
        "q99": [0.9] * 10,
    }
    (bucket / "meta" / "eef_stats.json").write_text(json.dumps(stats))


def make_bcz_bucket(tmp_path: Path, n_episodes: int = 2) -> Path:
    bucket = tmp_path / "BC-Z-Dataset"
    bucket.mkdir(parents=True, exist_ok=True)
    (bucket / "meta").mkdir(exist_ok=True)
    info = {
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))
    _write_episodes_parquet(bucket, n_episodes)
    _write_tasks_parquet(bucket)
    _write_tasks_annotated_parquet(bucket, n_episodes)
    _write_data_shard(bucket, n_rows=n_episodes * EP_LENGTH)
    _write_video_placeholder(bucket)
    _make_eef_stats(bucket)
    return bucket


@contextmanager
def _mock_video_decoder(height: int, width: int, n_frames: int = 9):
    """Patch ``_decode_video_frames`` to return PIL images of the requested size."""

    def _fake(path, frame_indices, h, w):
        # Return one PIL image per requested index at the requested h, w
        return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


class TestInit:
    def test_loads_bucket(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(384, 320):
            ds = OxeBczDataset(dataset_dir=str(b))
        assert len(ds._eps_df) == 2
        assert ds.action_dim == EEF_DIM
        assert ds._dataset_id == "BC-Z-Dataset"

    def test_normalize_mode_quantile_loads_stats(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        ds = OxeBczDataset(dataset_dir=str(b), normalize_mode="quantile")
        assert ds._normalization_stats is not None
        assert "q01" in ds._normalization_stats
        assert "q99" in ds._normalization_stats

    def test_normalize_mode_z_score_loads_stats(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        ds = OxeBczDataset(dataset_dir=str(b), normalize_mode="z-score")
        assert ds._normalization_stats is not None
        assert "mean" in ds._normalization_stats
        assert "std" in ds._normalization_stats

    def test_normalize_mode_null_skips_stats(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        ds = OxeBczDataset(dataset_dir=str(b), normalize_mode=None)
        assert ds._normalization_stats is None

    def test_missing_stats_with_quantile_mode_raises(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        (b / "meta" / "eef_stats.json").unlink()
        with pytest.raises(FileNotFoundError, match="eef_stats.json"):
            OxeBczDataset(dataset_dir=str(b), normalize_mode="quantile")


class TestGetItemShape:
    def test_multiview_true_yields_canvas(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(256, 320):
            ds = OxeBczDataset(dataset_dir=str(b), multiview=True, height=384, width=320)
            sample = ds[0]
        # 9 frames (num_video_frames = (33-1)/4 + 1 = 9), each (384, 320)
        assert len(sample["video"]) == 9
        # PIL.Image .size returns (W, H)
        assert sample["video"][0].size == (320, 384)

    def test_multiview_false_yields_head_only(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(dataset_dir=str(b), multiview=False, height=64, width=96)
            sample = ds[0]
        assert sample["video"][0].size == (96, 64)


class TestGetItemActionProprio:
    def _make_ds(self, tmp_path, **kw):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            return OxeBczDataset(dataset_dir=str(b), multiview=False, height=64, width=96, **kw)[0]

    def test_action_shape_20d_left_arm_filled(self, tmp_path):
        s = self._make_ds(tmp_path)
        assert s["action"].shape == (32, EEF_DIM)
        # Front 10 dims should NOT all be zero (real data)
        assert s["action"][:, :10].abs().sum() > 0
        # Back 10 dims all zero (right arm padding)
        assert (s["action"][:, 10:] == 0).all()

    def test_proprio_shape_20d_left_arm_filled(self, tmp_path):
        s = self._make_ds(tmp_path)
        assert s["proprio"].shape == (1, EEF_DIM)
        assert s["proprio"][0, :10].abs().sum() > 0
        assert (s["proprio"][0, 10:] == 0).all()

    def test_action_mask_2d_shape_and_left_arm_pattern(self, tmp_path):
        s = self._make_ds(tmp_path)
        assert s["action_mask"].shape == (32, EEF_DIM)
        # 32 < 60 (ep_length), so all 32 timesteps are valid
        assert s["action_mask"][:, :10].all()
        # Right arm always False
        assert not s["action_mask"][:, 10:].any()

    def test_proprio_mask_2d_left_arm_pattern(self, tmp_path):
        s = self._make_ds(tmp_path)
        assert s["proprio_mask"].shape == (1, EEF_DIM)
        assert s["proprio_mask"][0, :10].all()
        assert not s["proprio_mask"][0, 10:].any()


class TestEnableActionSupervisionFalse:
    def test_supervision_off_masks_zero(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(
                dataset_dir=str(b),
                multiview=False,
                height=64,
                width=96,
                enable_action_supervision=False,
            )
            s = ds[0]
        # Action / proprio masks should be entirely False
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
        # Action / proprio VALUES are still loaded (just masked out)
        assert s["action"][:, :10].abs().sum() > 0


class TestNormalize:
    def test_quantile_clips_to_unit(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(
                dataset_dir=str(b),
                multiview=False,
                height=64,
                width=96,
                normalize_mode="quantile",
            )
            s = ds[0]
        # All normalized values are clipped to [-1, 1]
        assert (s["action"][:, :10].abs() <= 1.0 + 1e-5).all()
        assert (s["proprio"][:, :10].abs() <= 1.0 + 1e-5).all()

    def test_min_max_bounded(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(
                dataset_dir=str(b),
                multiview=False,
                height=64,
                width=96,
                normalize_mode="min-max",
            )
            s = ds[0]
        # In-range data → all in [-1, 1] without clipping
        # (random uniform [-1, 1] → maps to [-1, 1] given min=-1, max=1)
        assert (s["action"][:, :10].abs() <= 1.0 + 1e-5).all()

    def test_z_score_normalizes(self, tmp_path):
        # stats fixture: mean=0, std=0.5 → z-score output = x / 0.5 = 2x.
        # z-score is unbounded (no clip), so we assert finite + shape only.
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(
                dataset_dir=str(b),
                multiview=False,
                height=64,
                width=96,
                normalize_mode="z-score",
            )
            s = ds[0]
        assert s["action"].shape == (32, EEF_DIM)
        assert s["action"][:, :10].abs().sum() > 0
        assert s["action"].isfinite().all()
        assert s["proprio"].isfinite().all()

    def test_normalize_null_passthrough(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(
                dataset_dir=str(b),
                multiview=False,
                height=64,
                width=96,
                normalize_mode=None,
            )
            s = ds[0]
        # No clipping; raw values can exceed [-1, 1] though our test data is within
        # the range, so this just exercises the code path doesn't blow up.
        assert s["action"].shape == (32, EEF_DIM)


class TestPrompt:
    def test_prompt_from_tasks_annotated(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(dataset_dir=str(b), multiview=False, height=64, width=96)
            s0 = ds[0]
            # Each window starting in episode 0 → annotated text for ep 0
            assert s0["prompt"] == "pick the block — episode 0"
            # Window from episode 1 (window count for ep 0 = 60 with stride 1)
            s_ep1 = ds[60]
            assert s_ep1["prompt"] == "pick the block — episode 1"

    def test_missing_tasks_annotated_raises(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        (b / "meta" / "tasks_annotated.parquet").unlink()
        import pytest

        with pytest.raises(FileNotFoundError, match="tasks_annotated.parquet"):
            OxeBczDataset(dataset_dir=str(b))


class TestPickle:
    def test_round_trip(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset(dataset_dir=str(b), multiview=False, height=64, width=96)
        import pickle

        data = pickle.dumps(ds)
        ds2 = pickle.loads(data)
        with _mock_video_decoder(64, 96):
            s = ds2[0]
        assert s["action"].shape == (32, EEF_DIM)


class TestFromConfig:
    def test_from_config_single_bucket(self, tmp_path):
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        cfg = {
            "type": "oxe_bcz",
            "dataset_dir": str(b),
            "multiview": False,
            "height": 64,
            "width": 96,
            "normalize_mode": "quantile",
            "enable_action_supervision": True,
        }
        with _mock_video_decoder(64, 96):
            ds = OxeBczDataset.from_config(cfg, split="train")
            s = ds[0]
        assert s["action"].shape == (32, EEF_DIM)
        assert s["proprio"].shape == (1, EEF_DIM)

    def test_from_config_normalize_mode_null_skips_stats(self, tmp_path):
        # Regression (B1): an explicit ``normalize_mode: null`` in the config
        # must reach the ctor. Otherwise from_config drops the None and the OXE
        # reader falls back to DEFAULT_NORMALIZE_MODE="quantile", silently
        # re-enabling the stats loading the user asked to disable.
        b = make_bcz_bucket(tmp_path, n_episodes=2)
        cfg = {
            "type": "oxe_bcz",
            "dataset_dir": str(b),
            "normalize_mode": None,
        }
        ds = OxeBczDataset.from_config(cfg, split="train")
        assert ds._normalize_mode is None
        assert ds._normalization_stats is None
