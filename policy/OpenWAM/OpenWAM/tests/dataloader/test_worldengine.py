"""Tests for the WorldEngine dataloader (openwam/dataloader/worldengine.py).

WorldEngine is a SHARDED LeRobot v3 ego-hand root (``shard_000 … shard_049``,
each a standalone bucket). Its reader is a thin LeRobotV3Reader subclass whose
only WorldEngine-specific behaviours are:

  * root mode → one bucket per shard, aggregated by
    ``MultiBucketWorldEngineDataset`` (per-shard task_index namespaces);
  * plain-English prompts read from ``meta/tasks.parquet``, whose text column is
    named ``__index_level_0__`` (pyarrow placeholder — the file carries no pandas
    metadata), with fallbacks for a ``task`` column / a restored string index;
  * dropping episodes whose task text is unusable (blank / "null" / NULL /
    non-Latin);
  * single ego camera (video-only supervision) with the shared L-shape canvas;
  * an exhaustive-by-default load-time prompt-consistency probe whose failures
    raise ``DataContractError`` so root mode cannot swallow them.

Video decode is monkeypatched so no mp4 is needed.
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
import torch
from PIL import Image

from openwam.dataloader.utils.lerobotv3 import DataContractError
from openwam.dataloader.worldengine import (
    MultiBucketWorldEngineDataset,
    WorldEngineDataset,
    _ArrowPromptMap,
    normalize_prompt,
)

FPS = 1.0
EP_LEN = 40
CAM = "observation.images.ego"
CAM_R = "observation.images.ego_right"


# ---------------------------------------------------------------------------
# Synthetic WorldEngine shard on disk (no mp4; decode patched)
# ---------------------------------------------------------------------------


def _write_tasks_parquet(path: Path, task_index: list[int], texts: list[str]) -> None:
    """Write a tasks.parquet in the real drop's layout.

    pyarrow-written, NO pandas metadata → the instruction text stays an ordinary
    column literally named ``__index_level_0__`` when read back with
    ``pd.read_parquet``. Writing it via pandas instead would attach index
    metadata and change that, so the table is built directly.
    """
    table = pa.table(
        {
            "task_index": pa.array(task_index, type=pa.int64()),
            "__index_level_0__": pa.array(texts, type=pa.string()),
        }
    )
    pq.write_table(table, path)


def _make_worldengine_shard(
    root: Path,
    prompts: list[str],
    *,
    with_ego_right: bool = True,
    task_indices: list[int] | None = None,
    ep_len: int = EP_LEN,
    shard_per_episode: bool = False,
) -> Path:
    """Build a minimal single WorldEngine shard (one bucket).

    ``prompts[e]`` is the task string for episode e's per-episode ``tasks`` cell.
    Episodes are packed contiguously into one data shard and one (per view) video
    shard, mirroring the real layout. When ``with_ego_right`` the episodes table
    also carries the ego_right view columns (float dtype, some NaN) so we can
    assert the reader ignores them.

    ``task_indices`` (per-episode) decouples ``task_index`` from ``episode_index``
    like the real data (many episodes share a task). When ``None`` it defaults to
    ``range(n)``.

    ``ep_len`` is the per-episode frame count (all episodes share it); pass a
    value below ``video_stride + 1`` to build a train-window-empty short episode.

    ``shard_per_episode`` puts every episode in its OWN ``data/`` parquet, so the
    prompt-consistency probe has many shards to sample from.
    """
    n = len(prompts)
    tix = list(range(n)) if task_indices is None else list(task_indices)
    assert len(tix) == n
    meta = root / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    info = {
        "fps": FPS,
        "robot_type": "ego_hand",
        "splits": {"train": f"0:{n}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))

    # episodes parquet (with the per-episode `tasks` list<string> column)
    rows = []
    cum = 0
    for ep in range(n):
        row = {
            "episode_index": ep,
            "length": ep_len,
            "tasks": [prompts[ep]],
            "dataset_from_index": cum,
            "dataset_to_index": cum + ep_len,
            "data/chunk_index": 0,
            "data/file_index": ep if shard_per_episode else 0,
            f"videos/{CAM}/chunk_index": 0,
            f"videos/{CAM}/file_index": 0,
            f"videos/{CAM}/from_timestamp": float(cum) / FPS,
            f"videos/{CAM}/to_timestamp": float(cum + ep_len) / FPS,
        }
        if with_ego_right:
            # ego_right present for even episodes, missing (NaN) for odd — exercise
            # the float/NaN "adaptive view" columns the real data ships.
            present = ep % 2 == 0
            row[f"videos/{CAM_R}/chunk_index"] = 0.0 if present else float("nan")
            row[f"videos/{CAM_R}/file_index"] = 0.0 if present else float("nan")
        rows.append(row)
        cum += ep_len
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), meta / "episodes" / "file-000.parquet")

    # tasks.parquet built from the UNIQUE task_index -> prompt map (a shared
    # task_index reuses the first episode's prompt), so it is sparse /
    # non-identity when task_indices is supplied.
    by_ti: dict[int, str] = {}
    for ep in range(n):
        by_ti.setdefault(int(tix[ep]), prompts[ep])
    uniq = sorted(by_ti)
    _write_tasks_parquet(meta / "tasks.parquet", uniq, [by_ti[t] for t in uniq])

    # data shard(s): per-frame task_index = the episode's task_index.
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    if shard_per_episode:
        for ep in range(n):
            ti = np.full(ep_len, int(tix[ep]), dtype=np.int64)
            pq.write_table(
                pa.Table.from_pandas(pd.DataFrame({"task_index": ti})),
                data_dir / f"file-{ep:03d}.parquet",
            )
    else:
        ti = np.concatenate([np.full(ep_len, int(tix[ep]), dtype=np.int64) for ep in range(n)])
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({"task_index": ti})), data_dir / "file-000.parquet")
    return root


def _make_worldengine_root(root: Path, per_shard_prompts: list[list[str]]) -> Path:
    """Build a sharded root: ``shard_000 … shard_00N``, one bucket each."""
    root.mkdir(parents=True, exist_ok=True)
    for i, prompts in enumerate(per_shard_prompts):
        _make_worldengine_shard(root / f"shard_{i:03d}", prompts)
    return root


@pytest.fixture
def patch_decode(monkeypatch):
    """Return dummy frames so no mp4 is needed."""

    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


def _reader(root: Path, **kw) -> WorldEngineDataset:
    return WorldEngineDataset(
        dataset_dir=str(root),
        num_frames=33,
        video_stride=4,
        height=384,
        width=320,
        multiview=True,
        camera_layout=[CAM, "__missing_left__", "__missing_right__"],
        target_camera=CAM,
        unify_action=True,
        **kw,
    )


def _cfg(dataset_dir: Path, **kw) -> dict:
    cfg = {
        "type": "worldengine",
        "dataset_dir": str(dataset_dir),
        "num_frames": 33,
        "video_stride": 4,
        "height": 384,
        "width": 320,
        "multiview": True,
        "camera_layout": [CAM, "__missing_left__", "__missing_right__"],
        "target_camera": CAM,
        "unify_action": True,
        "split": "train",
    }
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------------------
# Prompt normalization
# ---------------------------------------------------------------------------


class TestNormalizePrompt:
    def test_plain_english_is_kept_verbatim(self):
        # This drop's tasks are already English — nothing is split off.
        s = "The operator folds a T-shirt at a home folding area."
        assert normalize_prompt(s) == s
        assert normalize_prompt(f"  {s}  ") == s

    def test_typographic_latin_is_kept(self):
        assert normalize_prompt("Sauté the onions — don’t burn them.") == "Sauté the onions — don’t burn them."

    def test_missing_blank_and_null_are_rejected(self):
        assert normalize_prompt(None) is None
        assert normalize_prompt("") is None
        assert normalize_prompt("   ") is None
        assert normalize_prompt("null") is None
        assert normalize_prompt("NULL") is None

    def test_nan_and_non_string_inputs_are_rejected(self):
        # A NULL parquet string cell arrives as float nan; str(nan) == "nan" is
        # non-blank and all-ASCII, so it would sail through every other check and
        # become a real training prompt.
        assert normalize_prompt(float("nan")) is None
        assert normalize_prompt(np.float64("nan")) is None
        assert normalize_prompt("nan") is None  # already stringified upstream
        assert normalize_prompt("NaN") is None
        # A numeric column that reaches us by mistake must not stringify either.
        assert normalize_prompt(0) is None
        assert normalize_prompt(np.int64(7)) is None
        assert normalize_prompt(["Do A."]) is None
        # numpy str_ subclasses str and must still be accepted.
        assert normalize_prompt(np.str_("Do A.")) == "Do A."

    def test_non_latin_script_is_rejected(self):
        assert normalize_prompt("Hold the steam铲 now.") is None
        assert normalize_prompt("擦拭杯子") is None

    def test_old_bilingual_string_is_not_special_cased(self):
        # The previous single-root drop wrapped prompts as "中文:…英文:…". That
        # format is NOT supported here: the Han glyphs trip the Latin allowlist so
        # the task is rejected outright rather than half-parsed.
        assert normalize_prompt("中文:折叠。英文:Fold the shirt.") is None


# ---------------------------------------------------------------------------
# Per-shard bucket reader
# ---------------------------------------------------------------------------


class TestWorldEngineBucket:
    def test_declares_multibucket_wrapper(self):
        # The drop is a sharded root (no meta/info.json at the top), so the reader
        # MUST declare a wrapper or from_config's root branch raises.
        assert WorldEngineDataset._multibucket_wrapper() is MultiBucketWorldEngineDataset

    def test_resolve_single_ego_camera(self, tmp_path):
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        ds = _reader(root)
        assert ds._head_camera == CAM
        assert ds._left_wrist_camera is None and ds._right_wrist_camera is None
        # ego_right, though present in the episodes table, is never resolved as a
        # decode camera.
        assert CAM_R not in ds._ep_video_frame_offsets

    def test_null_episodes_dropped(self, tmp_path):
        root = _make_worldengine_shard(tmp_path / "we", ["Do A.", "null", "Do C.", ""])
        ds = _reader(root)
        assert set(ds._eps_df["episode_index"].tolist()) == {0, 2}

    def test_non_english_episode_dropped(self, tmp_path):
        root = _make_worldengine_shard(tmp_path / "we", ["Pick up the cup.", "Hold the steam铲 now."])
        ds = _reader(root)
        assert set(ds._eps_df["episode_index"].tolist()) == {0}

    def test_prompt_read_from_index_level_0_column(self, tmp_path, patch_decode):
        # The drop's tasks.parquet has no pandas metadata, so its text column
        # surfaces as `__index_level_0__`. Pin that the reader finds it.
        root = _make_worldengine_shard(tmp_path / "we", ["Fold the shirt.", "Measure it."])
        assert "__index_level_0__" in pd.read_parquet(root / "meta" / "tasks.parquet").columns
        ds = _reader(root)
        seen = {ds[i]["prompt"] for i in range(len(ds))}
        assert seen == {"Fold the shirt.", "Measure it."}

    def test_prompt_read_from_task_column_fallback(self, tmp_path, patch_decode):
        root = _make_worldengine_shard(tmp_path / "we", ["Fold the shirt."])
        pq.write_table(
            pa.table({"task_index": pa.array([0], pa.int64()), "task": pa.array(["Fold the shirt."], pa.string())}),
            root / "meta" / "tasks.parquet",
        )
        assert {_reader(root)[i]["prompt"] for i in range(2)} == {"Fold the shirt."}

    def test_prompt_read_from_restored_string_index_fallback(self, tmp_path, patch_decode):
        # A writer that DOES emit pandas metadata makes pd.read_parquet restore the
        # text as the frame index (the Ego4D-style layout) — also accepted.
        root = _make_worldengine_shard(tmp_path / "we", ["Fold the shirt."])
        pd.DataFrame({"task_index": [0]}, index=pd.Index(["Fold the shirt."], name="task")).to_parquet(
            root / "meta" / "tasks.parquet"
        )
        assert "task" not in pd.read_parquet(root / "meta" / "tasks.parquet").columns  # it is the index
        assert {_reader(root)[i]["prompt"] for i in range(2)} == {"Fold the shirt."}

    def test_missing_task_text_column_raises(self, tmp_path):
        # No recognizable text column and a numeric index → fail loud rather than
        # silently mis-map prompts.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        pq.write_table(pa.table({"task_index": pa.array([0], pa.int64())}), root / "meta" / "tasks.parquet")
        with pytest.raises(DataContractError, match="no task-text column"):
            _reader(root)

    def test_missing_task_index_column_raises(self, tmp_path):
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        pq.write_table(
            pa.table({"__index_level_0__": pa.array(["Do A."], pa.string())}), root / "meta" / "tasks.parquet"
        )
        with pytest.raises(DataContractError, match="task_index"):
            _reader(root)

    def test_non_string_text_column_raises(self, tmp_path):
        # `__index_level_0__` means "whatever the row index was", so a conversion
        # that leaves an integer counter there must NOT be accepted: stringifying
        # it would make every prompt "0"/"1"/"2"… and nothing downstream would
        # notice. Named explicitly rather than left to the divergence probe, which
        # would blame the wrong file.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A.", "Do B."])
        pq.write_table(
            pa.table({"task_index": pa.array([0, 1], pa.int64()), "__index_level_0__": pa.array([0, 1], pa.int64())}),
            root / "meta" / "tasks.parquet",
        )
        with pytest.raises(DataContractError, match="not strings"):
            _reader(root)

    def test_string_column_with_a_null_is_not_escalated_to_a_layout_error(self, tmp_path):
        # The guard uses infer_dtype(skipna=True), not is_string_dtype: on
        # pandas 2.x one NULL cell flips is_string_dtype(object column) to False,
        # which would escalate a single bad task (normalize_prompt's job, episode
        # dropped by _filter_episodes) into a whole-source DataContractError.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A.", "null"])
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array([0, 1], pa.int64()),
                    "__index_level_0__": pa.array(["Do A.", None], pa.string()),
                }
            ),
            root / "meta" / "tasks.parquet",
        )
        ds = _reader(root)  # must not raise; episode 1 is filtered instead
        assert set(ds._eps_df["episode_index"].tolist()) == {0}

    def test_explicit_task_column_wins_over_index_placeholder(self, tmp_path, patch_decode):
        # When both exist, the self-describing `task` column is authoritative —
        # `__index_level_0__` may be any leftover index.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array([0], pa.int64()),
                    "task": pa.array(["Fold the shirt."], pa.string()),
                    "__index_level_0__": pa.array(["row-0"], pa.string()),
                }
            ),
            root / "meta" / "tasks.parquet",
        )
        assert {_reader(root)[i]["prompt"] for i in range(2)} == {"Fold the shirt."}

    def test_null_task_text_does_not_become_the_prompt_nan(self, tmp_path, patch_decode):
        # A NULL cell in a string column reads back as the float nan; a str()
        # coercion would make the literal prompt "nan" — non-blank, all-ASCII, and
        # therefore invisible to every other guard. Episode 1 must be dropped, and
        # no sample may carry "nan".
        root = _make_worldengine_shard(tmp_path / "we", ["Fold the shirt.", "Wipe the table."])
        pq.write_table(
            pa.table(
                {
                    "task_index": pa.array([0, 1], pa.int64()),
                    "__index_level_0__": pa.array(["Fold the shirt.", None], pa.string()),
                }
            ),
            root / "meta" / "tasks.parquet",
        )
        # episodes-side text for ep 1 is also NULL, so the two sources agree and
        # the divergence probe stays quiet — only the filter can catch this.
        eps_path = next((root / "meta" / "episodes").rglob("*.parquet"))
        eps = pd.read_parquet(eps_path)
        eps.loc[1, "tasks"] = [None]
        pq.write_table(pa.Table.from_pandas(eps), eps_path)

        ds = _reader(root)
        assert set(ds._eps_df["episode_index"].tolist()) == {0}
        assert {ds[i]["prompt"] for i in range(len(ds))} == {"Fold the shirt."}

    def test_prompt_resolves_by_task_index_not_episode_index(self, tmp_path, patch_decode):
        # Many episodes share a task (3,006 tasks for 71,567 episodes), so
        # task_index != episode_index. Here eps 0,1 share task_index 7 and ep 2
        # uses task_index 3 — tasks.parquet is sparse {3,7}. A lookup keyed by
        # episode_index would KeyError or resolve the wrong string.
        root = _make_worldengine_shard(
            tmp_path / "we", ["Pour the water.", "Pour the water.", "Wipe the table."], task_indices=[7, 7, 3]
        )
        ds = _reader(root)
        seen_by_ep: dict[int, set[str]] = {}
        for idx in range(len(ds)):
            ep_local = int(np.searchsorted(ds._cum_n_starts, idx, side="right") - 1)
            epi = int(ds._eps_df.iloc[ep_local]["episode_index"])
            seen_by_ep.setdefault(epi, set()).add(ds[idx]["prompt"])
        assert seen_by_ep[0] == {"Pour the water."}
        assert seen_by_ep[1] == {"Pour the water."}
        assert seen_by_ep[2] == {"Wipe the table."}

    def test_eps_df_pruned_to_needed_columns(self, tmp_path):
        # The episodes table's `tasks` list-column is CoW-dirtied per DataLoader
        # worker; the reader prunes eps_df to only the columns offset extraction /
        # _getitem_impl / the scanner read. Pin that.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A.", "Do B."])
        ds = _reader(root)
        cols = set(ds._eps_df.columns)
        # dropped — never read after init
        for c in [
            "tasks",
            "dataset_from_index",
            "dataset_to_index",
            f"videos/{CAM_R}/chunk_index",
            f"videos/{CAM}/from_timestamp",
            f"videos/{CAM}/to_timestamp",
        ]:
            assert c not in cols, f"expected {c!r} pruned"
        # kept — read by offset extraction / _getitem_impl / scanner episode key
        for c in [
            "length",
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            f"videos/{CAM}/chunk_index",
            f"videos/{CAM}/file_index",
            "_data_row_offset",
            f"_video_frame_offset/{CAM}",
        ]:
            assert c in cols, f"expected {c!r} kept"

    def test_prompt_map_is_cow_friendly_arrow_backed(self, tmp_path):
        # The reader stores prompts in an Arrow-backed _ArrowPromptMap, not a
        # {int: str} dict — so a fork-based DataLoader worker never privately
        # copies the prompt working set on lookup. Pin the contract the base
        # _resolve_prompt relies on: dict-like `in`/`[]`, KeyError on miss (sparse
        # task_index), and a *fresh* worker-local str per lookup.
        root = _make_worldengine_shard(tmp_path / "we", ["Pour the water.", "Wipe the table."], task_indices=[7, 3])
        pmap = _reader(root)._task_idx_to_text
        assert not isinstance(pmap, dict)
        assert isinstance(pmap, _ArrowPromptMap) and len(pmap) == 2
        assert 7 in pmap and 3 in pmap
        assert 0 not in pmap and 999 not in pmap and "x" not in pmap
        assert pmap[7] == "Pour the water." and pmap[3] == "Wipe the table."
        with pytest.raises(KeyError):
            pmap[999]
        # each lookup materializes a new str object → no incref of a stored str
        assert pmap[7] is not pmap[7]

    def test_sample_shape_video_only(self, tmp_path, patch_decode):
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        ds = _reader(root)
        s = ds[0]
        assert len(s["video"]) == 9 and s["video"][0].size == (320, 384)
        assert tuple(s["action"].shape) == (32, 80) and s["action"].dtype == torch.float32
        assert not s["action_mask"].any() and not s["proprio_mask"].any()
        assert tuple(s["proprio"].shape) == (1, 80)
        assert len(s["video_mask"]) == 9
        assert s["vace_video"] is None and len(s["first_frame_image"]) == 1
        assert ds.action_dim == 80

    def test_action_dim_without_unify_is_20(self, tmp_path):
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        ds = WorldEngineDataset(
            dataset_dir=str(root),
            multiview=True,
            camera_layout=[CAM, "__missing_left__", "__missing_right__"],
            target_camera=CAM,
            unify_action=False,
        )
        assert ds.action_dim == 20

    def test_window_floor_video_stride_plus_one(self, tmp_path):
        # An episode shorter than video_stride+1 (=5 here) must yield 0 train
        # windows; a length-5 episode yields exactly 1.
        short = _reader(_make_worldengine_shard(tmp_path / "we_short", ["Do A."], ep_len=4))
        assert short._train_min_window_len() == 5
        # The short episode survives prompt filtering (valid English prompt) — so
        # len(ds) == 0 is due to LENGTH, not accidental filtering...
        assert len(short._eps_df) == 1
        # ...and it contributes zero windows.
        assert len(short) == 0

        # Bracket the floor: length == video_stride+1 yields exactly one window.
        at_floor = _reader(_make_worldengine_shard(tmp_path / "we_floor", ["Do A."], ep_len=5))
        assert len(at_floor._eps_df) == 1 and len(at_floor) == 1

    def test_window_count_matches_long_episode_formula(self, tmp_path):
        # Real episodes are whole ~6-min recordings, so window count is dominated
        # by (length - video_stride) per episode. Pin the arithmetic.
        ds = _reader(_make_worldengine_shard(tmp_path / "we", ["Do A.", "Do B."], ep_len=100))
        assert len(ds) == 2 * (100 - 4)

    def test_from_config_on_a_single_shard_still_works(self, tmp_path, patch_decode):
        # Pointing dataset_dir straight at one shard (meta/info.json present) keeps
        # the single-bucket branch — useful for debugging one shard.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A.", "Do B."])
        ds = WorldEngineDataset.from_config(_cfg(root), split="train")
        assert isinstance(ds, WorldEngineDataset)
        assert len(ds) > 0 and ds.action_dim == 80


# ---------------------------------------------------------------------------
# Sharded root (multibucket) mode
# ---------------------------------------------------------------------------


class TestWorldEngineRootMode:
    def test_from_config_root_builds_one_bucket_per_shard(self, tmp_path, patch_decode):
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Do A.", "Do B."], ["Do C."], ["Do D."]])
        ds = WorldEngineDataset.from_config(_cfg(root), split="train")
        assert isinstance(ds, MultiBucketWorldEngineDataset)
        assert len(ds.buckets) == 3
        assert [b._dataset_id for b in ds.buckets] == ["shard_000", "shard_001", "shard_002"]
        assert len(ds) == sum(len(b) for b in ds.buckets)
        assert ds.action_dim == 80 and ds.normalization_stats is None

    def test_root_dispatch_keeps_task_index_inside_its_shard(self, tmp_path, patch_decode):
        # Every shard restarts task_index at 0, so a cross-shard lookup would
        # silently serve the WRONG prompt. Each shard here uses task_index 0 for a
        # different string; assert each window resolves its own shard's text.
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Fold the shirt."], ["Wipe the table."]])
        ds = WorldEngineDataset.from_config(_cfg(root), split="train")
        b0 = len(ds.buckets[0])
        assert {ds[i]["prompt"] for i in range(b0)} == {"Fold the shirt."}
        assert {ds[i]["prompt"] for i in range(b0, len(ds))} == {"Wipe the table."}

    def test_root_skips_non_bucket_subdirs(self, tmp_path, patch_decode):
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Do A."], ["Do B."]])
        (root / "scratch").mkdir()  # no meta/info.json → not a bucket
        (root / "README.md").write_text("notes")
        ds = WorldEngineDataset.from_config(_cfg(root), split="train")
        assert len(ds.buckets) == 2

    def test_root_total_hours_caps_the_load(self, tmp_path, patch_decode):
        # 2 shards x 2 episodes x 40 frames @ 1 fps = 160 s total. Cap at 80 s.
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Do A.", "Do B."], ["Do C.", "Do D."]])
        full = WorldEngineDataset.from_config(_cfg(root), split="train")
        capped = WorldEngineDataset.from_config(_cfg(root, total_hours=80.0 / 3600.0, seed=0), split="train")
        assert sum(len(b._eps_df) for b in capped.buckets) < sum(len(b._eps_df) for b in full.buckets)
        assert len(capped) > 0

    def test_root_without_any_bucket_raises(self, tmp_path):
        root = tmp_path / "WorldEngine"
        (root / "not_a_shard").mkdir(parents=True)
        with pytest.raises(FileNotFoundError):
            WorldEngineDataset.from_config(_cfg(root), split="train")

    def test_root_mode_does_not_swallow_a_diverged_shard(self, tmp_path):
        # build_multibucket deliberately tolerates a bucket that fails to build
        # (bad shard -> warning -> skip), which is right for an IO fault and WRONG
        # for proven-broken data: a silently dropped shard removes ~15M frames of
        # the real drop behind one log line. DataContractError must escape that
        # net, so root mode fails as loudly as single-bucket mode.
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Do A."], ["Do B."], ["Do C."]])
        _write_tasks_parquet(root / "shard_001" / "meta" / "tasks.parquet", [0], ["null"])
        with pytest.raises(DataContractError, match="diverged"):
            WorldEngineDataset.from_config(_cfg(root), split="train")

    def test_root_mode_does_not_swallow_a_malformed_tasks_table(self, tmp_path):
        # Same contract for the tasks.parquet layout guards in _load_prompts.
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Do A."], ["Do B."]])
        pq.write_table(
            pa.table({"task_index": pa.array([0], pa.int64())}), root / "shard_001" / "meta" / "tasks.parquet"
        )
        with pytest.raises(DataContractError, match="no task-text column"):
            WorldEngineDataset.from_config(_cfg(root), split="train")

    def test_root_mode_still_tolerates_an_environmental_bucket_failure(self, tmp_path, caplog):
        # The flip side: an unreadable bucket is an environment problem and must
        # keep being skipped, not promoted to a launch abort.
        root = _make_worldengine_root(tmp_path / "WorldEngine", [["Do A."], ["Do B."], ["Do C."]])
        (root / "shard_001" / "meta" / "info.json").write_bytes(b"{ not json")
        with caplog.at_level(logging.WARNING, logger="openwam.dataloader.utils.lerobotv3"):
            ds = WorldEngineDataset.from_config(_cfg(root), split="train")
        assert [b._dataset_id for b in ds.buckets] == ["shard_000", "shard_002"]
        assert any("shard_001" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# Load-time guards
# ---------------------------------------------------------------------------


class TestWorldEnginePromptConsistency:
    """Prompt-map input validation + the load-time two-source consistency probe."""

    def test_prompt_map_rejects_nan_task_index(self):
        # A tasks.parquet task_index column with nulls arrives as float64-with-NaN;
        # np.int64(NaN) is INT64_MIN — a silent garbage key. Must fail loud instead.
        with pytest.raises(ValueError):
            _ArrowPromptMap(np.array([1.0, float("nan")]), ["a", "b"])
        with pytest.raises(ValueError):
            _ArrowPromptMap(np.array([1.5, 2.0]), ["a", "b"])  # non-integer values
        with pytest.raises(ValueError):
            _ArrowPromptMap(np.array(["1", "2"], dtype=object), ["a", "b"])  # non-numeric
        # Integral floats (pandas' lossless float64 view of a clean int column) OK.
        pmap = _ArrowPromptMap(np.array([1.0, 2.0]), ["a", "b"])
        assert pmap[1] == "a" and pmap[2] == "b"

    def test_init_asserts_prompt_consistency_on_unusable_task_entry(self, tmp_path):
        # Divergence: the EPISODES-side tasks cell is usable English (episode
        # survives _filter_episodes) but its task_index row in tasks.parquet is
        # "null" (→ "" in the map). Every window of that episode would raise at
        # runtime and the base +1 _safe_get would kill the worker (a real episode
        # backs ~11k windows, far beyond the 64-retry budget) — construction must
        # fail loud instead.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        _write_tasks_parquet(root / "meta" / "tasks.parquet", [0], ["null"])
        with pytest.raises(DataContractError, match="diverged"):
            _reader(root)

    def test_init_asserts_prompt_consistency_on_missing_task_index(self, tmp_path):
        # Divergence: the per-frame task_index (0) is absent from tasks.parquet
        # entirely — the runtime lookup would KeyError on every window.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        _write_tasks_parquet(root / "meta" / "tasks.parquet", [99], ["Do A."])
        with pytest.raises(DataContractError, match="diverged"):
            _reader(root)

    def test_consistency_pass_skips_unreadable_shard(self, tmp_path, caplog):
        # Environmental failures must NOT block construction (the reader's standing
        # contract): an unreadable data shard is the integrity scanner's job — the
        # pass skips it with a warning and the reader still builds.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A."])
        shard = root / "data" / "chunk-000" / "file-000.parquet"
        shard.write_bytes(b"not a parquet file")  # torn/corrupt shard
        with caplog.at_level(logging.WARNING, logger="openwam.dataloader.worldengine"):
            ds = _reader(root)
        assert len(ds._eps_df) == 1  # construction survived
        assert any("prompt-consistency pass skipped" in r.message for r in caplog.records)

    def test_probe_default_is_exhaustive(self, tmp_path, caplog):
        # The default cap (0) must cover EVERY referenced shard: a divergence
        # confined to one chunk is the likely failure, and a sampled probe would
        # walk past it.
        assert WorldEngineDataset.PROMPT_CHECK_MAX_SHARDS == 0
        root = _make_worldengine_shard(tmp_path / "we", [f"Do task {i}." for i in range(10)], shard_per_episode=True)
        with caplog.at_level(logging.INFO, logger="openwam.dataloader.worldengine"):
            ds = _reader(root)
        assert len(ds._eps_df) == 10
        msg = next(r.getMessage() for r in caplog.records if "prompt consistency verified" in r.getMessage())
        assert "for 10/10 episodes across 10/10 data shards" in msg

    @pytest.mark.parametrize("broken_ep", range(10))
    def test_exhaustive_probe_catches_divergence_in_any_shard(self, tmp_path, broken_ep):
        # Parametrized over every shard so the exhaustive guarantee is pinned
        # position-independently — a regression that silently reintroduced
        # sampling would fail on whichever shards the sample skips, not pass by
        # luck of which one the test happened to break.
        root = _make_worldengine_shard(tmp_path / "we", [f"Do task {i}." for i in range(10)], shard_per_episode=True)
        broken = root / "data" / "chunk-000" / f"file-{broken_ep:03d}.parquet"
        pq.write_table(pa.Table.from_pandas(pd.DataFrame({"task_index": np.full(EP_LEN, 77, dtype=np.int64)})), broken)
        with pytest.raises(DataContractError, match="diverged"):
            _reader(root)

    @pytest.mark.parametrize("flip_to", [77, 1])  # 77: absent from tasks.parquet; 1: present, still a violation
    def test_probe_catches_task_index_flip_within_episode(self, tmp_path, flip_to):
        # The runtime prompt is resolved from each WINDOW's first row, so a
        # task_index that flips mid-episode diverges from the first-row value the
        # filter (and a first-row-only probe) saw: windows past the flip resolve a
        # different task, or raise if the value is missing — the single-chunk
        # failure the exhaustive probe exists to catch. Constancy is the
        # invariant, so a flip to a VALID task_index must raise too.
        root = _make_worldengine_shard(tmp_path / "we", ["Do A.", "Do B."], ep_len=200)
        ti = np.concatenate(
            [
                np.zeros(100, dtype=np.int64),  # episode 0: flips halfway…
                np.full(100, flip_to, dtype=np.int64),
                np.full(200, 1, dtype=np.int64),  # episode 1: constant (control)
            ]
        )
        pq.write_table(
            pa.Table.from_pandas(pd.DataFrame({"task_index": ti})),
            root / "data" / "chunk-000" / "file-000.parquet",
        )
        with pytest.raises(DataContractError, match="varies WITHIN"):
            _reader(root)

    def test_capped_probe_reports_its_reduced_coverage(self, tmp_path, caplog):
        # A positive cap trades coverage for launch latency. Pin that the log says
        # so — silent truncation would read as "everything was checked".
        root = _make_worldengine_shard(tmp_path / "we", [f"Do task {i}." for i in range(10)], shard_per_episode=True)
        with caplog.at_level(logging.INFO, logger="openwam.dataloader.worldengine"):
            _reader(root, prompt_check_max_shards=4)
        msg = next(r.getMessage() for r in caplog.records if "prompt consistency verified" in r.getMessage())
        assert "for 4/10 episodes across 4/10 data shards" in msg

    def test_capped_probe_is_configurable_from_config(self, tmp_path, patch_decode):
        # The cap is a CONFIG_KEY, so an operator can trade coverage for launch
        # time from yaml/CLI without editing code.
        assert "prompt_check_max_shards" in WorldEngineDataset.CONFIG_KEYS
        root = _make_worldengine_shard(tmp_path / "we", [f"Do task {i}." for i in range(10)], shard_per_episode=True)
        ds = WorldEngineDataset.from_config(_cfg(root, prompt_check_max_shards=3), split="train")
        assert ds.PROMPT_CHECK_MAX_SHARDS == 3
        # null in yaml → key skipped by from_config → class default (exhaustive).
        assert WorldEngineDataset.from_config(_cfg(root), split="train").PROMPT_CHECK_MAX_SHARDS == 0

    def test_probe_shard_selection_is_pure_and_spread(self):
        # Directly pin the selection function the probe delegates to: no RNG (a
        # sample that varied per launch would make a divergence appear and vanish
        # across restarts), evenly spread, and pass-through when uncapped.
        shards = [(0, i) for i in range(10)]
        assert WorldEngineDataset._sample_probe_shards(shards, 0) == shards
        assert WorldEngineDataset._sample_probe_shards(shards, -1) == shards
        assert WorldEngineDataset._sample_probe_shards(shards, 99) == shards
        picked = WorldEngineDataset._sample_probe_shards(shards, 4)
        assert picked == [(0, 0), (0, 3), (0, 6), (0, 9)]  # exact stride, not "some 4"
        # endpoints always covered, and repeated calls are byte-identical
        assert picked[0] == shards[0] and picked[-1] == shards[-1]
        assert all(WorldEngineDataset._sample_probe_shards(shards, 4) == picked for _ in range(5))
