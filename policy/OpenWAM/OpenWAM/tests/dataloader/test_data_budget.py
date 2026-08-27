"""Integration tests for the optional total_hours / max_hours data budget.

Two layers:

  - Reader-level: build a minimal LeRobot v3 bucket on disk in tmp_path and
    verify that ``EgoDexDataset(..., max_hours=X)`` and
    ``RoboCOINDataset(..., max_hours=X)`` actually filter eps_df to the
    requested budget. Crucially, ``max_hours=None`` (the default for users
    who don't opt into the budget knob) must produce a byte-identical
    eps_df to the pre-budget path.

  - from_config-level: build N buckets and verify that
    ``MultiBucketEgoDexDataset.from_config(total_hours=X)`` /
    ``MultiRobotCOINDataset.from_config(total_hours=X)`` propagate
    water-filling allocations into each child, with logged warnings when
    the target exceeds the available footage.

The buckets are scaffolded with realistic-shaped parquet metadata + a tiny
``info.json`` so the reader's __init__ path is fully exercised; no mp4
decoding happens because ``__getitem__`` is not called.

To keep math obvious, all fake buckets use fps=1.0 and episodes of 60 frames
each → exactly 1 minute per episode → 60 episodes = 1 hour per bucket.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from openwam.dataloader.deprecated.egodex import EgoDexDataset, MultiBucketEgoDexDataset
from openwam.dataloader.mixture import MixtureDataset
from openwam.dataloader.robocoin import MultiRobotCOINDataset, RoboCOINDataset

# ---------------------------------------------------------------------------
# Fake LeRobot v3 bucket scaffolding
# ---------------------------------------------------------------------------

EP_LENGTH = 60  # frames per episode (at fps=1.0 → 1 min per episode)
FPS = 1.0  # Synthetic — 60 episodes per bucket = exactly 1 hour


def _write_episodes_parquet(
    bucket_dir: Path,
    n_episodes: int,
    head_cam: str,
    left_wrist: str | None = None,
    right_wrist: str | None = None,
) -> None:
    """Write a LeRobot v3 episodes parquet with all required columns.

    All episodes are packed into a single (chunk=0, file=0) shard so the
    fake bucket only needs one ``data/chunk-000/file-000.parquet`` file.
    """
    eps_dir = bucket_dir / "meta" / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cum = 0
    for ep in range(n_episodes):
        row = {
            "episode_index": ep,
            "length": EP_LENGTH,
            "dataset_from_index": cum,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{head_cam}/chunk_index": 0,
            f"videos/{head_cam}/file_index": 0,
        }
        if left_wrist is not None:
            row[f"videos/{left_wrist}/chunk_index"] = 0
            row[f"videos/{left_wrist}/file_index"] = 0
        if right_wrist is not None:
            row[f"videos/{right_wrist}/chunk_index"] = 0
            row[f"videos/{right_wrist}/file_index"] = 0
        rows.append(row)
        cum += EP_LENGTH
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), eps_dir / "chunk-000.parquet")


def _write_data_shard(bucket_dir: Path, n_rows: int) -> None:
    """Write a fake ``data/chunk-000/file-000.parquet`` with n_rows rows.

    ``RoboCOINDataset._add_data_offsets_from_files`` reads ``num_rows`` from
    each data shard's metadata. Columns are minimal — just enough to make
    pyarrow happy; the reader doesn't consume them at init time.
    """
    data_dir = bucket_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Minimal columns — pyarrow needs at least one. Use a tiny dummy column.
    df = pd.DataFrame({"_pad": np.zeros(n_rows, dtype=np.int32)})
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _write_tasks_parquet(bucket_dir: Path) -> None:
    """Single-task index → prompt mapping."""
    df = pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick the block"], name="task"))
    df.to_parquet(bucket_dir / "meta" / "tasks.parquet")


def make_egodex_bucket(tmp_path: Path, name: str, n_episodes: int) -> Path:
    """Create a minimal EgoDex-compatible bucket on disk.

    Returns the bucket directory path.
    """
    bucket = tmp_path / name
    bucket.mkdir(parents=True, exist_ok=True)
    meta = bucket / "meta"
    meta.mkdir(exist_ok=True)
    info = {
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))
    _write_episodes_parquet(bucket, n_episodes, head_cam="observation.images.ego")
    _write_tasks_parquet(bucket)
    # EgoDex's __init__ doesn't call _add_data_offsets_from_files, but
    # writing a data shard keeps the bucket layout consistent with the
    # RoboCOIN one and avoids surprises if EgoDex ever adds the scan.
    _write_data_shard(bucket, n_rows=n_episodes * EP_LENGTH)
    return bucket


def make_robocoin_bucket(tmp_path: Path, name: str, n_episodes: int) -> Path:
    """Create a minimal RoboCOIN-compatible bucket on disk."""
    bucket = tmp_path / name
    bucket.mkdir(parents=True, exist_ok=True)
    meta = bucket / "meta"
    meta.mkdir(exist_ok=True)
    head = "observation.images.cam_high_rgb"
    left = "observation.images.cam_left_wrist_rgb"
    right = "observation.images.cam_right_wrist_rgb"
    info = {
        "fps": FPS,
        "robot_type": "fake_robot",
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {head: {}, left: {}, right: {}},
    }
    (meta / "info.json").write_text(json.dumps(info))
    _write_episodes_parquet(bucket, n_episodes, head_cam=head, left_wrist=left, right_wrist=right)
    _write_tasks_parquet(bucket)
    _write_data_shard(bucket, n_rows=n_episodes * EP_LENGTH)
    return bucket


# ---------------------------------------------------------------------------
# Reader-level (single bucket) — EgoDex
# ---------------------------------------------------------------------------


class TestEgoDexMaxHoursSingleBucket:
    def test_max_hours_none_is_unchanged(self, tmp_path):
        """Regression: max_hours=None (default) → eps_df has all 60 episodes."""
        b = make_egodex_bucket(tmp_path, "b0", n_episodes=60)
        ds = EgoDexDataset(dataset_dir=str(b), num_frames=33, video_stride=4)
        assert len(ds._eps_df) == 60

    def test_max_hours_caps_eps_df(self, tmp_path):
        """max_hours=0.5h → ~30 episodes (target 1800 frames at fps=1)."""
        b = make_egodex_bucket(tmp_path, "b1", n_episodes=60)
        ds = EgoDexDataset(
            dataset_dir=str(b),
            num_frames=33,
            video_stride=4,
            max_hours=0.5,
            subsample_seed=42,
        )
        # Each episode = 60 frames at fps=1 → 60s. Greedy stops once cum
        # >= 0.5h * 3600 * 1 = 1800 frames. So actual ep count is the
        # smallest k with k*60 >= 1800 → k = 30.
        assert len(ds._eps_df) == 30
        actual_hours = ds._eps_df["length"].sum() / 1.0 / 3600.0
        assert actual_hours == pytest.approx(0.5, abs=1.0 / 60.0)  # within 1 ep

    def test_max_hours_over_budget_returns_all(self, tmp_path):
        """max_hours > available → full bucket, no error."""
        b = make_egodex_bucket(tmp_path, "b2", n_episodes=60)
        ds = EgoDexDataset(
            dataset_dir=str(b),
            num_frames=33,
            video_stride=4,
            max_hours=10.0,  # 10h vs 1h actual
        )
        assert len(ds._eps_df) == 60

    def test_max_hours_reproducible_with_same_seed(self, tmp_path):
        """Same project.seed + same max_hours → same subset."""
        b = make_egodex_bucket(tmp_path, "b3", n_episodes=60)
        d1 = EgoDexDataset(dataset_dir=str(b), num_frames=33, video_stride=4, max_hours=0.5, subsample_seed=42)
        d2 = EgoDexDataset(dataset_dir=str(b), num_frames=33, video_stride=4, max_hours=0.5, subsample_seed=42)
        assert list(d1._eps_df["episode_index"]) == list(d2._eps_df["episode_index"])

    def test_max_hours_too_small_raises(self, tmp_path):
        """max_hours so tiny that 0 episodes fit → ValueError."""
        b = make_egodex_bucket(tmp_path, "b4", n_episodes=60)
        with pytest.raises(ValueError, match="target_hours must be > 0"):
            EgoDexDataset(
                dataset_dir=str(b),
                num_frames=33,
                video_stride=4,
                max_hours=0.0,
            )


# ---------------------------------------------------------------------------
# Reader-level (single bucket) — RoboCOIN
# ---------------------------------------------------------------------------


class TestRoboCOINMaxHoursSingleBucket:
    def test_max_hours_none_is_unchanged(self, tmp_path):
        b = make_robocoin_bucket(tmp_path, "b0", n_episodes=60)
        ds = RoboCOINDataset(dataset_dir=str(b), num_frames=33, video_stride=4)
        assert len(ds._eps_df) == 60

    def test_max_hours_caps_eps_df(self, tmp_path):
        b = make_robocoin_bucket(tmp_path, "b1", n_episodes=60)
        ds = RoboCOINDataset(
            dataset_dir=str(b),
            num_frames=33,
            video_stride=4,
            max_hours=0.5,
            subsample_seed=42,
        )
        assert len(ds._eps_df) == 30
        actual_hours = ds._eps_df["length"].sum() / 1.0 / 3600.0
        assert actual_hours == pytest.approx(0.5, abs=1.0 / 60.0)

    def test_max_hours_reproducible_with_same_seed(self, tmp_path):
        b = make_robocoin_bucket(tmp_path, "b3", n_episodes=60)
        d1 = RoboCOINDataset(dataset_dir=str(b), num_frames=33, video_stride=4, max_hours=0.5, subsample_seed=42)
        d2 = RoboCOINDataset(dataset_dir=str(b), num_frames=33, video_stride=4, max_hours=0.5, subsample_seed=42)
        assert list(d1._eps_df["episode_index"]) == list(d2._eps_df["episode_index"])


# ---------------------------------------------------------------------------
# from_config (multi-bucket) — EgoDex
# ---------------------------------------------------------------------------


def _make_egodex_root(tmp_path: Path, sizes: list[int]) -> Path:
    """Build a root containing len(sizes) buckets, each with sizes[i] episodes."""
    root = tmp_path / "egodex_root"
    root.mkdir(exist_ok=True)
    for i, n in enumerate(sizes):
        make_egodex_bucket(root, f"bucket_{i:02d}", n_episodes=n)
    return root


def _make_robocoin_root(tmp_path: Path, sizes: list[int]) -> Path:
    root = tmp_path / "robocoin_root"
    root.mkdir(exist_ok=True)
    for i, n in enumerate(sizes):
        make_robocoin_bucket(root, f"bucket_{i:02d}", n_episodes=n)
    return root


def _multibucket_total_hours(ds) -> float:
    """Sum subsampled hours across all buckets of a MultiLeRobotV3Reader."""
    total_frames = 0
    for b in ds.buckets:
        total_frames += int(b._eps_df["length"].sum())
    return total_frames / FPS / 3600.0


def _multibucket_effective_hours(ds) -> float:
    """Sum the post-trim valid ranges represented by all selected leaves."""
    return sum(b.effective_hours for b in ds.buckets)


class TestEgoDexFromConfigTotalHours:
    def test_total_hours_none_loads_full(self, tmp_path):
        """total_hours unset → all episodes across all buckets."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        cfg = {"dataset_dir": str(root), "num_frames": 33, "video_stride": 4}
        ds = EgoDexDataset.from_config(cfg, split="train")
        assert isinstance(ds, MultiBucketEgoDexDataset)
        assert _multibucket_total_hours(ds) == pytest.approx(3.0, abs=1e-9)

    def test_total_hours_equal_split(self, tmp_path):
        """3 buckets × 1h each, total_hours=1.5h → 0.5h per bucket."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 1.5,
            "seed": 42,
        }
        ds = EgoDexDataset.from_config(cfg, split="train")
        actual = _multibucket_total_hours(ds)
        # Within 1 episode (1/60 h) per bucket × 3 buckets = 3/60 = 0.05h.
        assert actual == pytest.approx(1.5, abs=0.05)

    def test_total_hours_water_fill_small_bucket(self, tmp_path):
        """Small bucket (0.5h) + 2 large (1h each), budget 1.5h.
        Small uses all 0.5h; remaining 1.0h split → 0.5h per large bucket.
        """
        root = _make_egodex_root(tmp_path, sizes=[30, 60, 60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 1.5,
            "seed": 42,
        }
        ds = EgoDexDataset.from_config(cfg, split="train")
        actual = _multibucket_total_hours(ds)
        # Small bucket contributes its full 0.5h; the 2 large buckets each
        # get 0.5h (greedy). Total ≈ 1.5h ± 2 episodes worth of overshoot.
        assert actual == pytest.approx(1.5, abs=0.05)
        # Bucket 0 (small) is fully used.
        assert len(ds.buckets[0]._eps_df) == 30

    def test_water_fill_uses_effective_post_trim_capacity(self, tmp_path):
        """A raw-1h bucket with only 0.1h valid must release its surplus.

        Correct effective water-fill for a 0.6h budget is 0.1h from bucket 0
        plus 0.5h from bucket 1. Raw-manifest water-fill would allocate 0.3h
        to each, then silently realize only 0.4h after bucket 0 saturates.
        """
        root = _make_egodex_root(tmp_path, sizes=[60, 60])
        eps_path = root / "bucket_00" / "meta" / "episodes" / "chunk-000.parquet"
        eps = pd.read_parquet(eps_path)
        eps["_valid_start"] = 0
        eps["_valid_end"] = 6  # 60 eps × 6 frames @ 1 fps = 0.1 effective h
        eps.to_parquet(eps_path)

        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 0.6,
            "seed": 42,
        }
        ds = EgoDexDataset.from_config(cfg, split="train")
        assert len(ds.buckets[0]._eps_df) == 60
        assert _multibucket_effective_hours(ds) == pytest.approx(0.6, abs=1.0 / 60.0)

    def test_total_hours_over_budget_warns(self, tmp_path, caplog):
        """target > available → full dataset + warning."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 100.0,  # 100h vs 2h available
        }
        with caplog.at_level(logging.WARNING, logger="openwam.dataloader.deprecated.egodex"):
            ds = EgoDexDataset.from_config(cfg, split="train")
        assert any("exceeds available effective footage" in r.message for r in caplog.records)
        assert _multibucket_total_hours(ds) == pytest.approx(2.0, abs=1e-9)

    def test_total_hours_invalid_raises(self, tmp_path):
        root = _make_egodex_root(tmp_path, sizes=[60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 0.0,
        }
        with pytest.raises(ValueError, match=r"total_hours must be > 0"):
            EgoDexDataset.from_config(cfg, split="train")


# ---------------------------------------------------------------------------
# from_config (multi-bucket) — RoboCOIN
# ---------------------------------------------------------------------------


class TestRoboCOINFromConfigTotalHours:
    def test_total_hours_none_loads_full(self, tmp_path):
        root = _make_robocoin_root(tmp_path, sizes=[60, 60, 60])
        cfg = {"dataset_dir": str(root), "num_frames": 33, "video_stride": 4}
        ds = RoboCOINDataset.from_config(cfg, split="train")
        assert isinstance(ds, MultiRobotCOINDataset)
        assert _multibucket_total_hours(ds) == pytest.approx(3.0, abs=1e-9)

    def test_total_hours_equal_split(self, tmp_path):
        root = _make_robocoin_root(tmp_path, sizes=[60, 60, 60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 1.5,
            "seed": 42,
        }
        ds = RoboCOINDataset.from_config(cfg, split="train")
        actual = _multibucket_total_hours(ds)
        assert actual == pytest.approx(1.5, abs=0.05)

    def test_total_hours_over_budget_warns(self, tmp_path, caplog):
        root = _make_robocoin_root(tmp_path, sizes=[60, 60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 100.0,
        }
        with caplog.at_level(logging.WARNING, logger="openwam.dataloader.robocoin"):
            ds = RoboCOINDataset.from_config(cfg, split="train")
        assert any("exceeds available effective footage" in r.message for r in caplog.records)
        assert _multibucket_total_hours(ds) == pytest.approx(2.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Reproducibility across runs
# ---------------------------------------------------------------------------


class TestReproducibility:
    def test_egodex_same_seed_same_subset(self, tmp_path):
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        cfg = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 1.5,
            "seed": 42,
        }
        ds1 = EgoDexDataset.from_config(cfg, split="train")
        ds2 = EgoDexDataset.from_config(cfg, split="train")
        for b1, b2 in zip(ds1.buckets, ds2.buckets):
            assert list(b1._eps_df["episode_index"]) == list(b2._eps_df["episode_index"])

    def test_egodex_different_seed_different_subset(self, tmp_path):
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        cfg_a = {
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "total_hours": 1.5,
            "seed": 42,
        }
        cfg_b = dict(cfg_a, seed=7919)
        ds_a = EgoDexDataset.from_config(cfg_a, split="train")
        ds_b = EgoDexDataset.from_config(cfg_b, split="train")
        # At least one bucket should have a different subset.
        differs = any(
            set(ba._eps_df["episode_index"]) != set(bb._eps_df["episode_index"])
            for ba, bb in zip(ds_a.buckets, ds_b.buckets)
        )
        assert differs


# ---------------------------------------------------------------------------
# Mixture-level seed propagation: project.seed must reach sub-source subsample
# ---------------------------------------------------------------------------


class TestMixtureSeedPropagation:
    """The mixture's top-level seed must propagate into each sub-source's
    subsample, not just the mixture-level index_map shuffle.

    Regression context: before the fix ``mixture.from_config`` only forwarded
    ``seed`` to ``MixtureDataset.__init__`` (the index_map shuffle); the
    sub-source cfgs carried no ``seed`` field, so their reader-level subsample
    fell back to the ctor default 42 and was decoupled from project.seed. After
    the fix the top-level seed is pushed down into every sub-cfg (an explicit
    per-source ``seed`` still wins), so changing project.seed re-rolls *which
    episodes* each sub-source selects.
    """

    @pytest.fixture(autouse=True)
    def _register_deprecated_reader_for_mixture_unit_test(self, monkeypatch):
        """Keep this mixture plumbing test isolated from the active registry."""
        from openwam.dataloader.registry import DATASET_REGISTRY

        monkeypatch.setitem(DATASET_REGISTRY, "egodex", EgoDexDataset)

    @staticmethod
    def _cfg(root, seed, *, child_seed=None):
        ego = {
            "type": "egodex",
            "enabled": True,
            "weight": 1.0,
            "dataset_dir": str(root),
            "num_frames": 33,
            "video_stride": 4,
            "height": 384,
            "width": 320,
            "total_hours": 1.5,  # < 3h full → forces subsample across the 3 buckets
        }
        if child_seed is not None:
            ego["seed"] = child_seed
        return {"weight_strategy": "manual", "seed": seed, "datasets": {"egodex": ego}}

    @staticmethod
    def _subset(mix):
        ego = mix.get_dataset("egodex")
        return [set(b._eps_df["episode_index"]) for b in ego.buckets]

    def test_seed_propagates_to_subsample(self, tmp_path):
        """Different mixture.seed → sub-source picks a different episode subset."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        m1 = MixtureDataset.from_config(self._cfg(root, seed=42), split="train")
        m2 = MixtureDataset.from_config(self._cfg(root, seed=7919), split="train")
        assert any(a != b for a, b in zip(self._subset(m1), self._subset(m2))), (
            "mixture.seed did not reach the sub-source subsample"
        )

    def test_default_seed_matches_standalone(self, tmp_path):
        """Regression: mixture.seed=42 selects the SAME episodes a standalone
        egodex build (subsample default 42) would — so the default stays
        byte-identical to the pre-fix behaviour and existing baselines/checkpoints
        are unaffected."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        mix = MixtureDataset.from_config(self._cfg(root, seed=42), split="train")
        standalone = EgoDexDataset.from_config(
            {"dataset_dir": str(root), "num_frames": 33, "video_stride": 4, "total_hours": 1.5},
            split="train",
        )
        std_subset = [set(b._eps_df["episode_index"]) for b in standalone.buckets]
        assert self._subset(mix) == std_subset

    def test_explicit_child_seed_overrides_mixture(self, tmp_path):
        """A sub-cfg's explicit ``seed`` wins over the mixture seed, so per-dataset
        pinning for ablations still works."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        # Different mixture seeds, but both pin the child to seed=42 → identical subset.
        m1 = MixtureDataset.from_config(self._cfg(root, seed=1, child_seed=42), split="train")
        m2 = MixtureDataset.from_config(self._cfg(root, seed=2, child_seed=42), split="train")
        assert self._subset(m1) == self._subset(m2)


# ---------------------------------------------------------------------------
# Corrupt-bucket tolerance during effective-hour construction
# ---------------------------------------------------------------------------


def _truncate_episodes_parquet(bucket: Path) -> None:
    """Cut the episodes parquet in half (footer magic lost) — reproduces a
    mid-copy truncation (real-world case: AgiBotWorld-Beta bucket 351)."""
    p = next((bucket / "meta" / "episodes").rglob("*.parquet"))
    data = p.read_bytes()
    p.write_bytes(data[: len(data) // 2])


class TestCorruptBucketEffectiveBudget:
    """Effective budgeting must preserve per-bucket failure tolerance."""

    def test_corrupt_bucket_skipped_with_budget(self, tmp_path, caplog):
        root = _make_egodex_root(tmp_path, sizes=[60, 60, 60])
        _truncate_episodes_parquet(root / "bucket_01")
        cfg = {"dataset_dir": str(root), "num_frames": 33, "video_stride": 4, "total_hours": 1.0, "seed": 42}
        with caplog.at_level(logging.WARNING, logger="openwam.dataloader.utils.lerobotv3"):
            ds = EgoDexDataset.from_config(cfg, split="train")
        assert {b._dataset_id for b in ds.buckets} == {"bucket_00", "bucket_02"}
        # Budget is water-filled across the 2 healthy buckets only: 0.5h each.
        assert _multibucket_total_hours(ds) == pytest.approx(1.0, abs=0.05)
        assert any("skipping bucket_01" in r.message for r in caplog.records)

    def test_corrupt_bucket_skipped_without_budget(self, tmp_path):
        """Parity guard: the total_hours=None path tolerated this all along."""
        root = _make_egodex_root(tmp_path, sizes=[60, 60])
        _truncate_episodes_parquet(root / "bucket_00")
        cfg = {"dataset_dir": str(root), "num_frames": 33, "video_stride": 4}
        ds = EgoDexDataset.from_config(cfg, split="train")
        assert {b._dataset_id for b in ds.buckets} == {"bucket_01"}

    def test_all_buckets_corrupt_raises(self, tmp_path):
        root = _make_egodex_root(tmp_path, sizes=[60, 60])
        _truncate_episodes_parquet(root / "bucket_00")
        _truncate_episodes_parquet(root / "bucket_01")
        cfg = {"dataset_dir": str(root), "num_frames": 33, "video_stride": 4, "total_hours": 1.0}
        with pytest.raises(RuntimeError, match="buckets failed to load"):
            EgoDexDataset.from_config(cfg, split="train")
