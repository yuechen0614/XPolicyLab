"""Tests for the deprecated Lightwheel reader (openwam/dataloader/deprecated/lightwheel.py).

Covers the flat-reader specifics for the two-level ``<Task>/<uuid>`` single-episode
tree: relative-dir manifest keys, per-episode English prompt (+ non-Latin guard),
manifest build + caching, (dir, episode) exclusion, the flat window index,
``_nonempty_starts`` jump table, and the video-only sample-dict shape (video decode
is monkeypatched so no mp4 is needed).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

from openwam.dataloader.deprecated.lightwheel import LightwheelDataset, _lightwheel_prompt, _scan_one_leaf

CAM = "observation.images.ego"


def _make_leaf(root: Path, task: str, uuid: str, length: int, prompt: str) -> Path:
    """Build a synthetic single-episode Lightwheel leaf ``<root>/<task>/<uuid>/``."""
    d = root / task / uuid
    (d / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    info = {
        "fps": 30.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (d / "meta" / "info.json").write_text(json.dumps(info))
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


@pytest.fixture
def patch_decode(monkeypatch):
    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.deprecated.lightwheel.decode_video_frames", fake_decode)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestLightwheelPrompt:
    def test_list_cell(self):
        assert _lightwheel_prompt(["Fold the shirt."]) == "Fold the shirt."

    def test_ndarray_cell(self):
        assert _lightwheel_prompt(np.array(["Wipe the table."], dtype=object)) == "Wipe the table."

    def test_scalar(self):
        assert _lightwheel_prompt("Do it.") == "Do it."

    def test_empty_and_none(self):
        assert _lightwheel_prompt([]) is None
        assert _lightwheel_prompt(None) is None
        assert _lightwheel_prompt(["   "]) is None

    def test_none_element_is_not_literal_none(self):
        # A [None] cell must map to None, not the literal string "None" (which
        # would otherwise slip through as a training prompt).
        assert _lightwheel_prompt([None]) is None
        assert _lightwheel_prompt(np.array([None], dtype=object)) is None

    def test_latin_kept(self):
        # Legitimate typographic English / loanword accents survive the guard.
        assert _lightwheel_prompt(["Sautéed vegetables"]) == "Sautéed vegetables"

    def test_non_latin_dropped(self):
        # Any CJK / non-Latin script → None (episode dropped), guaranteeing no
        # non-English reaches training.
        assert _lightwheel_prompt(["把杯子摆好"]) is None
        assert _lightwheel_prompt(["clean 桌子"]) is None


class TestScanOneLeaf:
    def test_single_episode_offset_and_relative_dir(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "p0")
        rows = _scan_one_leaf(tmp_path / "TaskA" / "uuid0", root=tmp_path, cam=CAM)
        assert len(rows) == 1
        # (dir, episode_index, length, prompt, vchunk, vfile, voffset)
        assert rows[0][0] == "TaskA/uuid0"  # relative <Task>/<uuid> path
        assert rows[0][1] == 0
        assert rows[0][2] == 20
        assert rows[0][3] == "p0"
        assert rows[0][6] == 0  # single-episode → video-frame offset 0

    def test_empty_prompt_skipped(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuidBad", 20, "   ")
        assert _scan_one_leaf(tmp_path / "TaskA" / "uuidBad", root=tmp_path, cam=CAM) == []

    def test_non_dataset_dir(self, tmp_path):
        (tmp_path / "TaskA" / "junk").mkdir(parents=True)
        assert _scan_one_leaf(tmp_path / "TaskA" / "junk", root=tmp_path, cam=CAM) == []


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _reader(root: Path, **kw) -> LightwheelDataset:
    return LightwheelDataset(
        dataset_dir=str(root),
        num_frames=33,
        video_stride=4,
        window_stride=1,
        height=384,
        width=320,
        multiview=True,
        camera_layout=[CAM, "__missing_left__", "__missing_right__"],
        target_camera=CAM,
        unify_action=True,
        **kw,
    )


class TestLightwheelReader:
    def test_window_index_and_len(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        _make_leaf(tmp_path, "TaskA", "uuid1", 10, "a1")
        _make_leaf(tmp_path, "TaskB", "uuid2", 15, "b0")
        ds = _reader(tmp_path)
        # min_window_len = video_stride+1 = 5; windows = (L-5)//1 + 1
        # 20 -> 16, 10 -> 6, 15 -> 11  => 33
        assert len(ds) == 16 + 6 + 11

    def test_manifest_cached_and_reused(self, tmp_path, monkeypatch):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        ds1 = _reader(tmp_path)
        # Manifest filename is camera-keyed so switching target_camera never reuses
        # another camera's chunk/file/offset columns or filtered dir set.
        mpath = tmp_path / "_openwam_lightwheel_manifest_v1_observation_images_ego.parquet"
        assert mpath.exists()
        # The old camera-agnostic name must NOT be produced (it was the bug).
        assert not (tmp_path / "_openwam_lightwheel_manifest_v1.parquet").exists()
        # The camera is stamped into the parquet metadata for explicit-path reuse.
        assert pq.read_table(mpath).schema.metadata[b"openwam_head_camera"] == CAM.encode()

        # Second construction must LOAD the cache, not rescan — assert the scan
        # helper is never called (the old test only compared lengths, which passed
        # even on a full rescan).
        def _boom(*a, **k):
            raise AssertionError("cached manifest must be reused, not rescanned")

        monkeypatch.setattr("openwam.dataloader.deprecated.lightwheel._scan_one_leaf", _boom)
        ds2 = _reader(tmp_path)
        assert len(ds1) == len(ds2)

    def test_two_level_dir_key_resolves_video_path(self, tmp_path, patch_decode):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        ds = _reader(tmp_path)
        assert str(ds._dirs[0]) == "TaskA/uuid0"
        # The reader builds {dataset_dir}/{dir}/videos/{cam}/chunk-000/file-000.mp4;
        # a successful ds[0] (with decode patched) exercises that join.
        assert ds[0]["prompt"] == "a0"

    def test_sample_shape_video_only(self, tmp_path, patch_decode):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "shoelaces")
        ds = _reader(tmp_path)
        s = ds[0]
        assert len(s["video"]) == 9 and s["video"][0].size == (320, 384)
        assert tuple(s["action"].shape) == (32, 80) and s["action"].dtype == torch.float32
        assert not s["action_mask"].any() and not s["proprio_mask"].any()
        assert tuple(s["proprio"].shape) == (1, 80) and len(s["video_mask"]) == 9
        assert s["vace_video"] is None and len(s["first_frame_image"]) == 1
        assert s["prompt"] == "shoelaces"
        assert ds.action_dim == 80

    def test_prompt_per_episode_across_leaves(self, tmp_path, patch_decode):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "ep0-prompt")  # 16 windows
        _make_leaf(tmp_path, "TaskA", "uuid1", 10, "ep1-prompt")
        ds = _reader(tmp_path)
        assert ds[0]["prompt"] == "ep0-prompt"
        assert ds[16]["prompt"] == "ep1-prompt"

    def test_exclusion_drops_episode(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        _make_leaf(tmp_path, "TaskA", "uuid1", 10, "a1")
        _make_leaf(tmp_path, "TaskB", "uuid2", 15, "b0")
        # Exclude TaskA/uuid1 (10 frames -> 6 windows) => 33 - 6 = 27
        (tmp_path / "_openwam_lightwheel_excluded.json").write_text(json.dumps({"excluded": {"TaskA/uuid1": [0]}}))
        ds = _reader(tmp_path)
        assert len(ds) == 16 + 11

    def test_nonempty_starts_skips_zero_window_episodes(self, tmp_path):
        # A leaf shorter than video_stride+1 (=5) contributes 0 windows; its cum
        # start must NOT appear in _nonempty_starts (the _safe_get jump table).
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")  # 16 windows
        _make_leaf(tmp_path, "TaskA", "uuid1", 3, "tiny")  # 0 windows
        _make_leaf(tmp_path, "TaskA", "uuid2", 10, "a2")  # 6 windows
        ds = _reader(tmp_path)
        # sorted dirs: TaskA/uuid0, TaskA/uuid1, TaskA/uuid2 ; cum = [0,16,16,22]
        assert ds._nonempty_starts.tolist() == [0, 16]
        assert len(ds) == 22

    def test_english_guard_drops_non_latin_leaf(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "good english")  # kept, 16 windows
        _make_leaf(tmp_path, "TaskB", "uuidZ", 20, "把杯子摆好")  # non-Latin → dropped
        ds = _reader(tmp_path)
        assert len(ds._dirs) == 1 and str(ds._dirs[0]) == "TaskA/uuid0"
        assert len(ds) == 16

    def test_safe_get_recovers_past_zero_window_trailing_episode(self, tmp_path, monkeypatch):
        # Regression: the failing episode is the LAST non-empty one, followed by a
        # zero-window (too-short) tail. _safe_get must jump via _nonempty_starts to
        # a good episode rather than oscillate on the broken clip or index out of
        # range (a raw _cum_n_starts[ep+1] jump would land on _n_total).
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "good-ep")  # 16 windows, decodes fine
        _make_leaf(tmp_path, "TaskB", "uuidBad", 20, "bad-ep")  # 16 bad windows
        _make_leaf(tmp_path, "TaskB", "uuidTiny", 3, "tiny")  # 0-window tail
        ds = _reader(tmp_path)
        assert ds._nonempty_starts.tolist() == [0, 16]
        assert len(ds) == 32

        def decode_bad_fails(path, frame_indices, height, width):
            if "uuidBad" in str(path):
                raise RuntimeError("simulated truncated clip")
            return [Image.new("RGB", (width, height)) for _ in frame_indices]

        monkeypatch.setattr("openwam.dataloader.deprecated.lightwheel.decode_video_frames", decode_bad_fails)
        # Scanner path (direct impl) still surfaces the bad clip — retry is bypassed.
        with pytest.raises(RuntimeError):
            ds._getitem_impl(16)
        # Training path (_safe_get) recovers by jumping to the good episode.
        assert ds[16]["prompt"] == "good-ep"

    def test_safe_get_retries_transient_on_single_episode(self, tmp_path, monkeypatch):
        # With a single non-empty episode there is nowhere to jump. A transient
        # decode hiccup must be retried in place (like the base reader / haiyu)
        # rather than raising on the first attempt — otherwise one flaky NFS read
        # kills the DataLoader worker on a small / heavily-subsampled dataset.
        # (Regression for the Round-3 finding: this fix was not ported from haiyu.)
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "only-ep")
        ds = _reader(tmp_path)
        assert ds._nonempty_starts.tolist() == [0]  # single non-empty episode
        calls = {"n": 0}

        def flaky(path, frame_indices, height, width):
            calls["n"] += 1
            if calls["n"] == 1:  # first probe hiccups, then recovers
                raise RuntimeError("transient NFS hiccup")
            return [Image.new("RGB", (width, height)) for _ in frame_indices]

        monkeypatch.setattr("openwam.dataloader.deprecated.lightwheel.decode_video_frames", flaky)
        assert ds[0]["prompt"] == "only-ep"
        assert calls["n"] == 2  # attempt 0 failed, attempt 1 succeeded (not raised)

    def test_safe_get_reraises_after_budget_on_single_episode(self, tmp_path, monkeypatch):
        # A deterministic failure on the single-episode case still re-raises —
        # after exhausting the retry budget, not on the first attempt.
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "only-ep")
        ds = _reader(tmp_path)
        assert ds._nonempty_starts.tolist() == [0]

        def always_fail(path, frame_indices, height, width):
            raise RuntimeError("deterministic corruption")

        monkeypatch.setattr("openwam.dataloader.deprecated.lightwheel.decode_video_frames", always_fail)
        with pytest.raises(RuntimeError):
            ds[0]

    def test_val_split_empty(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        ds = _reader(tmp_path, split="val")
        assert len(ds) == 0

    def test_action_dim_without_unify(self, tmp_path):
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        ds = LightwheelDataset(
            dataset_dir=str(tmp_path),
            multiview=True,
            camera_layout=[CAM, "__missing_left__", "__missing_right__"],
            target_camera=CAM,
            unify_action=False,
        )
        assert ds.action_dim == 20


class TestLightwheelReviewFixes:
    """Behaviors added in response to the PR #33 review (@wayrise)."""

    def test_single_episode_video_files_flag(self, tmp_path):
        # Declared layout fact the GPU scanner reads (single mp4 == one episode)
        # so the truncation cross-check is not hardcoded via isinstance downstream.
        assert LightwheelDataset.SINGLE_EPISODE_VIDEO_FILES is True
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        assert _reader(tmp_path).SINGLE_EPISODE_VIDEO_FILES is True

    def test_head_video_path_matches_layout(self, tmp_path):
        # Single source of truth for the videos/{cam}/chunk-XXX/file-XXX.mp4 path
        # shared by the reader's decode and the external scanner.
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        ds = _reader(tmp_path)
        expected = tmp_path / "TaskA/uuid0" / "videos" / CAM / "chunk-000" / "file-000.mp4"
        assert ds.head_video_path(0) == expected

    @pytest.mark.parametrize(
        "payload",
        [
            "[1, 2, 3]",  # top-level list, not a dict → .get() AttributeError
            '{"excluded": [1, 2, 3]}',  # "excluded" a list → .items() AttributeError
            '{"excluded": {"TaskA/uuid0": 0}}',  # episode value not iterable → TypeError
        ],
    )
    def test_malformed_exclusion_does_not_crash(self, tmp_path, payload):
        # The reader promises a malformed exclusion file "must not crash
        # construction": a valid-JSON but wrong-shape file must be ignored
        # (over-include) rather than raise AttributeError/TypeError.
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")  # 16 windows
        (tmp_path / "_openwam_lightwheel_excluded.json").write_text(payload)
        ds = _reader(tmp_path)
        assert len(ds) == 16  # nothing excluded; construction survived

    def test_val_split_skips_manifest_build(self, tmp_path, monkeypatch):
        # split != "train" must NOT trigger the (expensive, ~20k-dir) manifest scan.
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")

        def _boom(self):
            raise AssertionError("val split must not build the manifest")

        monkeypatch.setattr(LightwheelDataset, "_load_or_build_manifest", _boom)
        ds = _reader(tmp_path, split="val")
        assert len(ds) == 0
        # cold-cache val must also not have written a manifest cache (camera-keyed name)
        assert not (tmp_path / "_openwam_lightwheel_manifest_v1_observation_images_ego.parquet").exists()


class TestLightwheelRound3Fixes:
    """Regression for the Round-3 review (@wayrise): haiyu's Round-2 manifest
    camera-stamp fix was not ported to this flat-reader copy."""

    def test_explicit_manifest_camera_mismatch_rebuilds(self, tmp_path):
        # An explicit manifest_path is still camera-validated: building it for one
        # camera and reusing the SAME path with a different target_camera must
        # rebuild (not silently reuse the stale camera's offsets / dir set). The
        # second camera has no columns in the fixture, so the rebuild finds no
        # episodes and raises — proving the stale manifest was NOT reused (which
        # would have succeeded with 16 windows).
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        mpath = tmp_path / "explicit_manifest.parquet"
        ds = _reader(tmp_path, manifest_path=str(mpath))
        assert len(ds) == 16 and mpath.exists()
        with pytest.raises(RuntimeError):
            LightwheelDataset(
                dataset_dir=str(tmp_path),
                multiview=True,
                camera_layout=["observation.images.other", "__missing_left__", "__missing_right__"],
                target_camera="observation.images.other",
                unify_action=True,
                manifest_path=str(mpath),
            )

    def test_unstamped_manifest_is_restamped_on_load(self, tmp_path):
        # A manifest with no camera stamp (pre-fix / hand-built) has an unknowable
        # camera, so it must NOT be trusted: loading rebuilds and re-stamps rather
        # than reusing potentially-wrong offsets.
        _make_leaf(tmp_path, "TaskA", "uuid0", 20, "a0")
        mpath = tmp_path / "unstamped.parquet"
        ds0 = _reader(tmp_path, manifest_path=str(mpath))
        tbl = pq.read_table(mpath)
        md = dict(tbl.schema.metadata or {})
        md.pop(b"openwam_head_camera", None)
        pq.write_table(tbl.replace_schema_metadata(md), mpath)
        assert b"openwam_head_camera" not in (pq.read_table(mpath).schema.metadata or {})
        ds1 = _reader(tmp_path, manifest_path=str(mpath))
        assert len(ds1) == len(ds0) == 16
        assert pq.read_table(mpath).schema.metadata[b"openwam_head_camera"] == CAM.encode()
