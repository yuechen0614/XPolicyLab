"""Tests for the deprecated Haiyu reader (openwam/dataloader/deprecated/haiyu.py).

Covers the flat-reader specifics: per-episode video-frame offset computation for
the multi-episode concatenated dirs, manifest build + caching, (dir, episode)
exclusion, the flat window index, and the video-only sample-dict shape (video
decode is monkeypatched so no mp4 is needed).
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

from openwam.dataloader.deprecated.haiyu import HaiyuDataset, _haiyu_prompt, _scan_one_dir

CAM = "observation.images.ego"


def _make_haiyu_dir(root: Path, name: str, ep_lengths: list[int], prompts: list[str]) -> Path:
    """Build a synthetic multi-episode Haiyu dir (one concatenated video/data)."""
    d = root / name
    (d / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    info = {
        "fps": 29.97,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (d / "meta" / "info.json").write_text(json.dumps(info))
    rows = []
    cum = 0
    for ep, (L, p) in enumerate(zip(ep_lengths, prompts)):
        rows.append(
            {
                "episode_index": ep,
                "length": L,
                "tasks": [p],
                "dataset_from_index": cum,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{CAM}/chunk_index": 0,
                f"videos/{CAM}/file_index": 0,
            }
        )
        cum += L
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), d / "meta" / "episodes" / "chunk-000.parquet")
    return d


@pytest.fixture
def patch_decode(monkeypatch):
    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.deprecated.haiyu.decode_video_frames", fake_decode)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestHaiyuPrompt:
    def test_list_cell(self):
        assert _haiyu_prompt(["Fold the shirt."]) == "Fold the shirt."

    def test_ndarray_cell(self):
        assert _haiyu_prompt(np.array(["Wipe the table."], dtype=object)) == "Wipe the table."

    def test_scalar(self):
        assert _haiyu_prompt("Do it.") == "Do it."

    def test_empty_and_none(self):
        assert _haiyu_prompt([]) is None
        assert _haiyu_prompt(None) is None
        assert _haiyu_prompt(["   "]) is None

    def test_none_element_is_not_literal_none(self):
        # A [None] cell must map to None, not the literal string "None" — cell-shape
        # handling is delegated to the shared ego4d._episode_tasks_string.
        assert _haiyu_prompt([None]) is None
        assert _haiyu_prompt(np.array([None], dtype=object)) is None


class TestScanOneDir:
    def test_multi_episode_offsets(self, tmp_path):
        d = _make_haiyu_dir(tmp_path, "dirA", [20, 10, 5], ["p0", "p1", "p2"])
        rows = _scan_one_dir(d, CAM)
        assert len(rows) == 3
        # (dir, episode_index, length, prompt, vchunk, vfile, voffset)
        assert [r[1] for r in rows] == [0, 1, 2]
        assert [r[2] for r in rows] == [20, 10, 5]
        assert [r[6] for r in rows] == [0, 20, 30]  # cumulative video-frame offsets
        assert [r[3] for r in rows] == ["p0", "p1", "p2"]

    def test_empty_prompt_episode_skipped(self, tmp_path):
        d = _make_haiyu_dir(tmp_path, "dirB", [20, 10], ["good", "   "])
        rows = _scan_one_dir(d, CAM)
        assert len(rows) == 1 and rows[0][3] == "good"

    def test_non_dataset_dir(self, tmp_path):
        (tmp_path / "junk").mkdir()
        assert _scan_one_dir(tmp_path / "junk", CAM) == []


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def _reader(root: Path, **kw) -> HaiyuDataset:
    return HaiyuDataset(
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


class TestHaiyuReader:
    def test_window_index_and_len(self, tmp_path):
        _make_haiyu_dir(tmp_path, "dirA", [20, 10], ["a0", "a1"])
        _make_haiyu_dir(tmp_path, "dirB", [15], ["b0"])
        ds = _reader(tmp_path)
        # min_window_len = video_stride+1 = 5; windows = (L-5)//1 + 1
        # 20 -> 16, 10 -> 6, 15 -> 11  => 33
        assert len(ds) == 16 + 6 + 11

    def test_manifest_cached_and_reused(self, tmp_path):
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])
        ds1 = _reader(tmp_path)
        # Manifest filename is camera-keyed so switching target_camera never
        # reuses another camera's chunk/file/offset columns.
        mpath = tmp_path / "_openwam_haiyu_manifest_v1_observation_images_ego.parquet"
        assert mpath.exists()
        # The old camera-agnostic name must NOT be produced (it was the bug).
        assert not (tmp_path / "_openwam_haiyu_manifest_v1.parquet").exists()
        # The camera is stamped in the parquet metadata for explicit-path reuse.
        assert pq.read_table(mpath).schema.metadata[b"openwam_head_camera"] == CAM.encode()
        # Second construction loads the cache (no rescan needed) and matches.
        ds2 = _reader(tmp_path)
        assert len(ds1) == len(ds2)

    def test_unstamped_manifest_is_restamped_on_load(self, tmp_path):
        # A manifest carrying no camera stamp (pre-fix / hand-built) has an
        # unknowable camera, so it must NOT be trusted: loading rebuilds and
        # re-stamps rather than reusing potentially-wrong offsets.
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])
        mpath = tmp_path / "unstamped.parquet"
        ds0 = _reader(tmp_path, manifest_path=str(mpath))
        # Strip the camera stamp to emulate a legacy / hand-built manifest.
        tbl = pq.read_table(mpath)
        md = dict(tbl.schema.metadata or {})
        md.pop(b"openwam_head_camera", None)
        pq.write_table(tbl.replace_schema_metadata(md), mpath)
        assert b"openwam_head_camera" not in (pq.read_table(mpath).schema.metadata or {})
        # Reload: unstamped → rebuild → re-stamped, same content.
        ds1 = _reader(tmp_path, manifest_path=str(mpath))
        assert len(ds1) == len(ds0) == 16
        assert pq.read_table(mpath).schema.metadata[b"openwam_head_camera"] == CAM.encode()

    def test_explicit_manifest_camera_mismatch_rebuilds(self, tmp_path):
        # An explicit manifest_path is still camera-validated: building it for one
        # camera and then reusing the SAME path with a different target_camera must
        # rebuild (not silently reuse the stale camera's offsets). Here the second
        # camera has no columns in the fixture, so the rebuild finds no episodes
        # and raises — proving the stale manifest was NOT reused (which would have
        # succeeded with 16 windows).
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])
        mpath = tmp_path / "explicit_manifest.parquet"
        ds = _reader(tmp_path, manifest_path=str(mpath))
        assert len(ds) == 16 and mpath.exists()
        with pytest.raises(RuntimeError):
            HaiyuDataset(
                dataset_dir=str(tmp_path),
                multiview=True,
                camera_layout=["observation.images.other", "__missing_left__", "__missing_right__"],
                target_camera="observation.images.other",
                unify_action=True,
                manifest_path=str(mpath),
            )

    def test_sample_shape_video_only(self, tmp_path, patch_decode):
        _make_haiyu_dir(tmp_path, "dirA", [20, 10], ["shoelaces", "sneakers"])
        ds = _reader(tmp_path)
        s = ds[0]
        assert len(s["video"]) == 9 and s["video"][0].size == (320, 384)
        assert tuple(s["action"].shape) == (32, 80) and s["action"].dtype == torch.float32
        assert not s["action_mask"].any() and not s["proprio_mask"].any()
        assert tuple(s["proprio"].shape) == (1, 80) and len(s["video_mask"]) == 9
        assert s["vace_video"] is None and len(s["first_frame_image"]) == 1
        assert s["prompt"] in {"shoelaces", "sneakers"}
        assert ds.action_dim == 80

    def test_second_episode_prompt_and_offset(self, tmp_path, patch_decode):
        # First dir episode 0 has 16 windows; index 16 is the first window of episode 1.
        _make_haiyu_dir(tmp_path, "dirA", [20, 10], ["ep0-prompt", "ep1-prompt"])
        ds = _reader(tmp_path)
        assert ds[0]["prompt"] == "ep0-prompt"
        assert ds[16]["prompt"] == "ep1-prompt"

    def test_exclusion_drops_episode(self, tmp_path):
        _make_haiyu_dir(tmp_path, "dirA", [20, 10], ["a0", "a1"])
        _make_haiyu_dir(tmp_path, "dirB", [15], ["b0"])
        (tmp_path / "_openwam_haiyu_excluded.json").write_text(json.dumps({"excluded": {"dirA": [1]}}))
        ds = _reader(tmp_path)
        # dirA episode 1 (10 frames -> 6 windows) excluded => 33 - 6 = 27
        assert len(ds) == 16 + 11

    def test_nonempty_starts_skips_zero_window_episodes(self, tmp_path):
        # Episodes shorter than video_stride+1 (=5) contribute 0 windows; their
        # cum-start must NOT appear in _nonempty_starts (that array is the jump
        # table for _safe_get and must only reference real, in-range windows).
        _make_haiyu_dir(tmp_path, "dirA", [20, 3, 10], ["a0", "tiny", "a2"])
        ds = _reader(tmp_path)
        # 20 -> 16 windows, 3 -> 0, 10 -> 6 ; cum = [0, 16, 16, 22]
        assert ds._nonempty_starts.tolist() == [0, 16]  # the 0-window episode is skipped
        assert len(ds) == 22

    def test_safe_get_recovers_past_zero_window_trailing_episode(self, tmp_path, monkeypatch):
        # Regression: the failing episode is the LAST non-empty one, followed by a
        # zero-window (too-short) episode. A raw _cum_n_starts[ep_local+1] jump
        # lands on _n_total (out of range) and oscillates on the same broken clip;
        # jumping via _nonempty_starts must skip the empty tail and recover on a
        # good episode instead of exhausting all retries.
        _make_haiyu_dir(tmp_path, "dirA", [20], ["good-ep"])  # 16 windows, decodes fine
        _make_haiyu_dir(tmp_path, "dirB", [20, 3], ["bad-ep", "tiny"])  # 16 bad windows + 0-window tail
        ds = _reader(tmp_path)
        assert ds._nonempty_starts.tolist() == [0, 16]
        assert len(ds) == 32

        def decode_dirB_fails(path, frame_indices, height, width):
            if "dirB" in str(path):
                raise RuntimeError("simulated truncated clip")
            return [Image.new("RGB", (width, height)) for _ in frame_indices]

        monkeypatch.setattr("openwam.dataloader.deprecated.haiyu.decode_video_frames", decode_dirB_fails)
        # Scanner path (direct impl) still surfaces the bad clip — retry is bypassed.
        with pytest.raises(RuntimeError):
            ds._getitem_impl(20)
        # Training path (_safe_get) recovers by jumping to the good episode.
        assert ds[20]["prompt"] == "good-ep"

    def test_safe_get_retries_transient_on_single_episode(self, tmp_path, monkeypatch):
        # With a single non-empty episode there is nowhere to jump. A transient
        # decode hiccup must be retried in place (like the base reader) rather
        # than raising on the first attempt — otherwise one flaky NFS read kills
        # the DataLoader worker on a small/heavily-subsampled dataset.
        _make_haiyu_dir(tmp_path, "dirA", [20], ["only-ep"])
        ds = _reader(tmp_path)
        assert ds._nonempty_starts.tolist() == [0]  # single non-empty episode
        calls = {"n": 0}

        def flaky(path, frame_indices, height, width):
            calls["n"] += 1
            if calls["n"] == 1:  # first probe hiccups, then recovers
                raise RuntimeError("transient NFS hiccup")
            return [Image.new("RGB", (width, height)) for _ in frame_indices]

        monkeypatch.setattr("openwam.dataloader.deprecated.haiyu.decode_video_frames", flaky)
        assert ds[0]["prompt"] == "only-ep"
        assert calls["n"] == 2  # attempt 0 failed, attempt 1 succeeded

    def test_safe_get_reraises_after_budget_on_single_episode(self, tmp_path, monkeypatch):
        # A deterministic failure on the single-episode case still re-raises —
        # after exhausting the retry budget, not on the first attempt.
        _make_haiyu_dir(tmp_path, "dirA", [20], ["only-ep"])
        ds = _reader(tmp_path)
        assert ds._nonempty_starts.tolist() == [0]

        def always_fail(path, frame_indices, height, width):
            raise RuntimeError("deterministic corruption")

        monkeypatch.setattr("openwam.dataloader.deprecated.haiyu.decode_video_frames", always_fail)
        with pytest.raises(RuntimeError):
            ds[0]

    def test_val_split_empty(self, tmp_path):
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])
        ds = _reader(tmp_path, split="val")
        assert len(ds) == 0

    def test_action_dim_without_unify(self, tmp_path):
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])
        ds = HaiyuDataset(
            dataset_dir=str(tmp_path),
            multiview=True,
            camera_layout=[CAM, "__missing_left__", "__missing_right__"],
            target_camera=CAM,
            unify_action=False,
        )
        assert ds.action_dim == 20


class TestHaiyuReviewFixes:
    """Parity behaviors added alongside the Lightwheel PR #33 review (@wayrise)."""

    def test_single_episode_video_files_flag(self, tmp_path):
        # Haiyu concatenates episodes per mp4 → frame-count truncation check must
        # be OFF; declared as a class attribute rather than an isinstance literal.
        assert HaiyuDataset.SINGLE_EPISODE_VIDEO_FILES is False
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])
        assert _reader(tmp_path).SINGLE_EPISODE_VIDEO_FILES is False

    def test_head_video_path_matches_layout(self, tmp_path):
        _make_haiyu_dir(tmp_path, "dirA", [20, 10], ["a0", "a1"])
        ds = _reader(tmp_path)
        expected = tmp_path / "dirA" / "videos" / CAM / "chunk-000" / "file-000.mp4"
        assert ds.head_video_path(0) == expected

    @pytest.mark.parametrize(
        "payload",
        [
            "[1, 2, 3]",
            '{"excluded": [1, 2, 3]}',
            '{"excluded": {"dirA": 0}}',
        ],
    )
    def test_malformed_exclusion_does_not_crash(self, tmp_path, payload):
        # A valid-JSON but wrong-shape exclusion file must be ignored, not crash
        # construction (same contract as Lightwheel).
        _make_haiyu_dir(tmp_path, "dirA", [20, 10], ["a0", "a1"])  # 16 + 6 = 22 windows
        (tmp_path / "_openwam_haiyu_excluded.json").write_text(payload)
        ds = _reader(tmp_path)
        assert len(ds) == 22

    def test_val_split_skips_manifest_build(self, tmp_path, monkeypatch):
        _make_haiyu_dir(tmp_path, "dirA", [20], ["a0"])

        def _boom(self):
            raise AssertionError("val split must not build the manifest")

        monkeypatch.setattr(HaiyuDataset, "_load_or_build_manifest", _boom)
        ds = _reader(tmp_path, split="val")
        assert len(ds) == 0
