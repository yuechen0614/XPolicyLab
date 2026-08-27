"""Tests for the full-load scanner tool (scripts/scan_dataset.py).

Covers the Round-2 review fixes that live in the scanner rather than the readers:
  * ``scan --config mixture`` fast-fails via a compose-only type peek (issue #4).
  * ``emit`` re-probes recorded failures and drops transient one-offs while
    keeping deterministic corruption (issue #3), robustly to exclusion-index
    drift between scan and emit (re-probe by episode identity, not position).

The scanner is a standalone script (not a package), so it is imported from its
file path. Video decode is monkeypatched so no mp4 is needed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

_SCAN_PATH = Path(__file__).resolve().parents[2] / "scripts" / "scan_dataset.py"
_spec = importlib.util.spec_from_file_location("scan_dataset_under_test", _SCAN_PATH)
scan_dataset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scan_dataset)

CAM = "observation.images.ego"


# ---------------------------------------------------------------------------
# Synthetic Haiyu fixture (mirrors tests/dataloader/test_haiyu.py, kept local
# so this file is self-contained).
# ---------------------------------------------------------------------------


def _make_haiyu_dir(root: Path, name: str, ep_lengths, prompts) -> Path:
    d = root / name
    (d / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (d / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 29.97,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }
        )
    )
    rows, cum = [], 0
    for ep, (length, p) in enumerate(zip(ep_lengths, prompts)):
        rows.append(
            {
                "episode_index": ep,
                "length": length,
                "tasks": [p],
                "dataset_from_index": cum,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{CAM}/chunk_index": 0,
                f"videos/{CAM}/file_index": 0,
            }
        )
        cum += length
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), d / "meta" / "episodes" / "chunk-000.parquet")
    return d


def _haiyu(root: Path):
    from openwam.dataloader.deprecated.haiyu import HaiyuDataset

    return HaiyuDataset(
        dataset_dir=str(root),
        num_frames=33,
        video_stride=4,
        window_stride=1,
        multiview=True,
        camera_layout=[CAM, "__missing_left__", "__missing_right__"],
        target_camera=CAM,
        unify_action=True,
    )


@pytest.fixture
def patch_decode_ok(monkeypatch):
    monkeypatch.setattr(
        "openwam.dataloader.deprecated.haiyu.decode_video_frames",
        lambda path, idxs, h, w: [Image.new("RGB", (w, h)) for _ in idxs],
    )


# ---------------------------------------------------------------------------
# Issue #4 — mixture fast-fail
# ---------------------------------------------------------------------------


def test_peek_config_type_distinguishes_mixture():
    assert scan_dataset._peek_config_type("mixture") == "mixture"
    assert scan_dataset._peek_config_type("worldengine") == "worldengine"


def test_cmd_scan_rejects_mixture_without_building():
    import argparse

    # cmd_scan reads only args.config before the fast-fail return; a mixture must
    # exit rc=2 up front (no slow build, no TypeError).
    assert scan_dataset.cmd_scan(argparse.Namespace(config="mixture")) == 2


# ---------------------------------------------------------------------------
# Issue #3 — re-probe drops transient, keeps deterministic
# ---------------------------------------------------------------------------


def test_reprobe_keeps_deterministic_drops_transient(tmp_path, monkeypatch):
    # dirA decodes fine (transiently-recorded failure → cleared); dirB always
    # fails (deterministic → kept).
    _make_haiyu_dir(tmp_path, "dirA", [20], ["a"])
    _make_haiyu_dir(tmp_path, "dirB", [20], ["b"])
    ds = _haiyu(tmp_path)  # dirA [0,16), dirB [16,32)
    target = scan_dataset._leaf_info(ds)[1]

    def decode(path, idxs, h, w):
        if "dirB" in str(path):
            raise RuntimeError("corrupt dirB")
        return [Image.new("RGB", (w, h)) for _ in idxs]

    monkeypatch.setattr("openwam.dataloader.deprecated.haiyu.decode_video_frames", decode)
    recs = [
        {"kind": "haiyu", "target": target, "key": ["dirA", 0], "local": 15, "err": "transient"},
        {"kind": "haiyu", "target": target, "key": ["dirB", 0], "local": 31, "err": "corrupt"},
    ]
    kept = scan_dataset._filter_reprobe_against_leaves(recs, [ds], tries=3)
    assert [r["key"] for r in kept] == [["dirB", 0]]


def test_reprobe_robust_to_exclusion_index_drift(tmp_path, monkeypatch):
    # Regression for the stale-positional-index bug: dirB is the bad episode,
    # recorded when its first window was global index 16. After dirA is excluded
    # (exclusion set grew between scan and emit), index 16 now maps to healthy
    # dirC — a positional-only re-probe would decode dirC, succeed, and WRONGLY
    # drop the bad dirB. Re-probing by episode identity relocates dirB and keeps it.
    _make_haiyu_dir(tmp_path, "dirA", [20], ["a"])
    _make_haiyu_dir(tmp_path, "dirB", [20], ["b"])
    _make_haiyu_dir(tmp_path, "dirC", [20], ["c"])
    (tmp_path / "_openwam_haiyu_excluded.json").write_text(json.dumps({"excluded": {"dirA": [0]}}))
    ds = _haiyu(tmp_path)  # dirA excluded → dirB [0,16), dirC [16,32)
    # The recorded local=16 no longer belongs to dirB (drifted onto dirC).
    assert scan_dataset._leaf_episode_key(ds, 16) == ["dirC", 0]
    target = scan_dataset._leaf_info(ds)[1]

    def decode(path, idxs, h, w):
        if "dirB" in str(path):
            raise RuntimeError("corrupt dirB")
        return [Image.new("RGB", (w, h)) for _ in idxs]

    monkeypatch.setattr("openwam.dataloader.deprecated.haiyu.decode_video_frames", decode)
    rec = {"kind": "haiyu", "target": target, "key": ["dirB", 0], "local": 16, "err": "corrupt"}
    kept = scan_dataset._filter_reprobe_against_leaves([rec], [ds], tries=2)
    assert [r["key"] for r in kept] == [["dirB", 0]]


def test_reprobe_empty_is_noop():
    assert scan_dataset._filter_reprobe_against_leaves([], [], tries=3) == []


# ---------------------------------------------------------------------------
# Round-3 review (@wayrise): lightwheel × Round-2 crossing fixes
# ---------------------------------------------------------------------------


def _make_lightwheel_leaf(root: Path, task: str, uuid: str, length: int, prompt: str) -> Path:
    d = root / task / uuid
    (d / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (d / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30.0,
                "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            }
        )
    )
    rows = [
        {
            "episode_index": 0,
            "length": length,
            "tasks": [prompt],
            "dataset_from_index": 0,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{CAM}/chunk_index": 0,
            f"videos/{CAM}/file_index": 0,
        }
    ]
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), d / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    return d


def _lightwheel(root: Path):
    from openwam.dataloader.deprecated.lightwheel import LightwheelDataset

    return LightwheelDataset(
        dataset_dir=str(root),
        num_frames=33,
        video_stride=4,
        window_stride=1,
        multiview=True,
        camera_layout=[CAM, "__missing_left__", "__missing_right__"],
        target_camera=CAM,
        unify_action=True,
    )


def test_leaf_locate_episode_handles_lightwheel(tmp_path):
    # M1: _leaf_locate_episode must special-case LightwheelDataset (it has no
    # _eps_df); the else branch would AttributeError. 20 frames, stride 4 → 16
    # windows [0,16); the episode's last window is global index 15.
    _make_lightwheel_leaf(tmp_path, "TaskA", "uuid0", 20, "a")
    ds = _lightwheel(tmp_path)
    assert scan_dataset._leaf_locate_episode(ds, ["TaskA/uuid0", 0]) == 15
    assert scan_dataset._leaf_locate_episode(ds, ["TaskA/absent", 0]) is None  # not a crash


def test_reprobe_lightwheel_relocates_by_key_on_drift(tmp_path, monkeypatch):
    # M1 end-to-end: a lightwheel record whose stale positional `local` drifted onto
    # a different episode must relocate by key through _leaf_locate_episode (the
    # lightwheel branch) rather than crash — and keep the still-bad episode.
    _make_lightwheel_leaf(tmp_path, "TaskA", "uuidA", 20, "a")
    _make_lightwheel_leaf(tmp_path, "TaskB", "uuidB", 20, "b")
    _make_lightwheel_leaf(tmp_path, "TaskC", "uuidC", 20, "c")
    (tmp_path / "_openwam_lightwheel_excluded.json").write_text(json.dumps({"excluded": {"TaskA/uuidA": [0]}}))
    ds = _lightwheel(tmp_path)  # TaskA excluded → TaskB [0,16), TaskC [16,32)
    target = scan_dataset._leaf_info(ds)[1]
    assert scan_dataset._leaf_episode_key(ds, 16) == ["TaskC/uuidC", 0]  # drifted off TaskB

    def decode(path, idxs, h, w):
        if "uuidB" in str(path):
            raise RuntimeError("corrupt B")
        return [Image.new("RGB", (w, h)) for _ in idxs]

    monkeypatch.setattr("openwam.dataloader.deprecated.lightwheel.decode_video_frames", decode)
    rec = {"kind": "lightwheel", "target": target, "key": ["TaskB/uuidB", 0], "local": 16, "err": "corrupt"}
    kept = scan_dataset._filter_reprobe_against_leaves([rec], [ds], tries=2)
    assert [r["key"] for r in kept] == [["TaskB/uuidB", 0]]


def test_reprobe_keeps_gpu_file_level_records_unprobed(tmp_path, monkeypatch):
    # M4: gpu_decode_scan writes file-level records (local=-1) after a full-file
    # decode + its own cross-GPU retry, so its verdict is deterministic. The CPU
    # re-probe must NOT last-window-decode and "clear" them (a mid-file corruption
    # would not reproduce on a sparse last-window seek). Contrast a normal window
    # record for the same (now decode-OK) episode, which IS cleared as transient.
    _make_haiyu_dir(tmp_path, "dirA", [20], ["a"])
    ds = _haiyu(tmp_path)
    target = scan_dataset._leaf_info(ds)[1]
    monkeypatch.setattr(  # every decode now succeeds
        "openwam.dataloader.deprecated.haiyu.decode_video_frames",
        lambda path, idxs, h, w: [Image.new("RGB", (w, h)) for _ in idxs],
    )
    window_rec = {"kind": "haiyu", "target": target, "key": ["dirA", 0], "local": 15, "err": "transient"}
    file_rec = {"kind": "haiyu", "target": target, "key": ["dirA", 0], "local": -1, "err": "decode_error: broken"}
    kept = scan_dataset._filter_reprobe_against_leaves([window_rec, file_rec], [ds], tries=3)
    # window record cleared (decodes fine now); file-level GPU record kept unprobed.
    assert [r["local"] for r in kept] == [-1]


# ---------------------------------------------------------------------------
# Round-3 review (@wayrise): emit-path hardening (N1 fail-loud) + shard guard
# ---------------------------------------------------------------------------


class TestExistingExclusionFailLoud:
    def test_missing_file_is_empty(self, tmp_path):
        assert scan_dataset._read_lerobot_excluded(str(tmp_path / "nope.json")) == set()
        assert scan_dataset._read_flat_excluded(str(tmp_path / "nope.json")) == {}

    def test_valid_files_parse(self, tmp_path):
        p1 = tmp_path / "excluded_episodes.json"
        p1.write_text(json.dumps({"episode_indices": [1, 2, 2, 3]}))
        assert scan_dataset._read_lerobot_excluded(str(p1)) == {1, 2, 3}
        p2 = tmp_path / "_openwam_haiyu_excluded.json"
        p2.write_text(json.dumps({"excluded": {"dirA": [0, 1]}}))
        assert scan_dataset._read_flat_excluded(str(p2)) == {"dirA": {0, 1}}

    def test_corrupt_existing_file_fails_loud(self, tmp_path):
        # A corrupt EXISTING exclusion file must abort emit, not read as empty —
        # emit then atomically overwrites, which would wipe its exclusions and
        # reload the bad episodes into training (violating only-add union).
        p1 = tmp_path / "excluded_episodes.json"
        p1.write_text("{ not valid json")
        with pytest.raises(SystemExit):
            scan_dataset._read_lerobot_excluded(str(p1))
        p2 = tmp_path / "_openwam_lightwheel_excluded.json"
        p2.write_text("{ not valid json")
        with pytest.raises(SystemExit):
            scan_dataset._read_flat_excluded(str(p2))

    @pytest.mark.parametrize(
        "payload",
        [
            "[1, 2, 3]",
            '{"episode_indices": "notalist"}',
            '{"episode_indices": [1, "x"]}',
            "{}",  # missing canonical key — a hand-edit; reading as empty would wipe it
            '{"episodes": [1, 2]}',  # misspelled key
        ],
    )
    def test_lerobot_wrong_shape_fails_loud(self, tmp_path, payload):
        # Valid JSON but wrong shape (top-level list, non-iterable / non-int
        # episode_indices, or a dict MISSING the canonical key — emit's own writes
        # always carry it) must ALSO fail loud, not read as empty — this is exactly
        # the class the training-path reader deliberately tolerates, so the emit
        # side must diverge and refuse to overwrite.
        p = tmp_path / "excluded_episodes.json"
        p.write_text(payload)
        with pytest.raises(SystemExit):
            scan_dataset._read_lerobot_excluded(str(p))

    @pytest.mark.parametrize(
        "payload",
        [
            "[1, 2, 3]",
            '{"excluded": [1, 2]}',
            '{"excluded": {"dirA": 5}}',
            "{}",  # missing canonical key
            '{"dirA": [0]}',  # old unwrapped format (pre-docstring-fix) — must die, not read empty
        ],
    )
    def test_flat_wrong_shape_fails_loud(self, tmp_path, payload):
        p = tmp_path / "_openwam_haiyu_excluded.json"
        p.write_text(payload)
        with pytest.raises(SystemExit):
            scan_dataset._read_flat_excluded(str(p))


class TestValidateShard:
    def test_valid_shards_pass(self):
        import argparse

        scan_dataset._validate_shard(argparse.Namespace(shard=0, num_shards=1, cmd="scan"))
        scan_dataset._validate_shard(argparse.Namespace(shard=3, num_shards=4, cmd="scan"))

    def test_verify_indices_are_lazy_sharded_and_limited(self):
        # Regression: verify-mixed used np.arange over the full billion-window
        # mixture before applying --limit, allocating ~10 GB for a smoke test.
        idx = scan_dataset._ShardedIndices(1_000_000_000, shard=2, num_shards=7, limit=4)
        assert len(idx) == 4
        assert [idx[i] for i in range(len(idx))] == [2, 9, 16, 23]
        assert idx[-1] == 23
        with pytest.raises(IndexError):
            _ = idx[4]

    @pytest.mark.parametrize("shard,num", [(1, 1), (4, 4), (-1, 2), (0, 0)])
    def test_out_of_range_aborts(self, shard, num):
        import argparse

        # 1-based misuse (1/1), overrun (4/4), negative, and num_shards=0 all abort
        # rather than silently scan an empty stride leaving windows uncovered.
        with pytest.raises(SystemExit):
            scan_dataset._validate_shard(argparse.Namespace(shard=shard, num_shards=num, cmd="scan"))

    def test_gpu_decode_scan_validates_shard_before_any_work(self, monkeypatch):
        # Round-4 regression: the GPU scanner shares the CPU scanner's shard
        # contract. A 1-based misuse must abort up front — BEFORE the ffmpeg
        # probe / GPU detection / (slow) dataset build — instead of every shard
        # exiting 0 while jobs[0] etc. are covered by no shard (silently clean).
        _GPU_PATH = Path(__file__).resolve().parents[2] / "scripts" / "gpu_decode_scan.py"
        spec = importlib.util.spec_from_file_location("gpu_decode_scan_under_test", _GPU_PATH)
        gpu_scan = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gpu_scan)

        def _too_far(*a, **k):  # the guard must fire before any of these
            raise AssertionError("shard guard must reject before ffmpeg/GPU/build work")

        monkeypatch.setattr(gpu_scan, "_check_ffmpeg_cuda", _too_far)
        monkeypatch.setattr(gpu_scan, "_detect_num_gpus", _too_far)
        monkeypatch.setattr(
            "sys.argv", ["gpu_decode_scan.py", "--config", "lightwheel", "--shard", "1", "--num-shards", "1"]
        )
        with pytest.raises(SystemExit) as ei:
            gpu_scan.main()
        assert "--shard must satisfy" in str(ei.value)
