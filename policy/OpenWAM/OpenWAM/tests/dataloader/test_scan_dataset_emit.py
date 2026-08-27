"""Tests for the scan_dataset.py emit-path hardening added in the PR #33 review
(@wayrise): unknown-kind detection (no silent drop) and the over-exclusion
guardrail that stops an environmental scan failure from zeroing a dataset.

The transaction tests inject a competing publish immediately before lock
acquisition. That deterministically catches plans built from an unlocked stale
snapshot without needing a real dataset build.
"""

from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

# scan_dataset.py lives under scripts/, not the installed package.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import scan_dataset  # noqa: E402

from openwam.dataloader.oxe_droid import (  # noqa: E402
    DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
    DROID_PROMPT_INPUTS_DIGEST_KEY,
    OxeDroidDataset,
    compute_droid_prompt_inputs_digest,
    load_droid_prompt_exclusions,
)


class TestGroupFailures:
    def test_splits_known_and_collects_unknown(self):
        recs = [
            {"kind": "lerobot", "target": "/a/meta/excluded_episodes.json", "key": 5},
            {"kind": "lerobot", "target": "/a/meta/excluded_episodes.json", "key": 5},  # dedup
            {"kind": "haiyu", "target": "/h/_openwam_haiyu_excluded.json", "key": ["dirA", 1]},
            {"kind": "lightwheel", "target": "/l/_openwam_lightwheel_excluded.json", "key": ["T/u", 0]},
            {"kind": "mystery", "target": "/x", "key": 0},
            {"kind": "mystery", "target": "/x", "key": 1},
        ]
        lerobot, flat, unknown = scan_dataset._group_failures(recs)
        assert lerobot == {"/a/meta/excluded_episodes.json": {5}}
        assert flat["/h/_openwam_haiyu_excluded.json"] == {"dirA": {1}}
        assert flat["/l/_openwam_lightwheel_excluded.json"] == {"T/u": {0}}
        assert unknown == {"mystery": 2}


class TestExclusionRatioViolations:
    def test_flags_over_threshold(self):
        planned = [{"target": "/t", "n_final": 60, "existing": 0, "n_new": 60}]
        v = scan_dataset._exclusion_ratio_violations(planned, {"/t": 100}, 0.05)
        assert len(v) == 1 and v[0][0] == "/t"

    def test_no_violation_under_threshold(self):
        planned = [{"target": "/t", "n_final": 3, "existing": 0, "n_new": 3}]
        assert scan_dataset._exclusion_ratio_violations(planned, {"/t": 100}, 0.05) == []

    def test_no_new_exclusions_never_violates(self):
        # Idempotent re-emit (nothing new) must not abort even when already large.
        planned = [{"target": "/t", "n_final": 100, "existing": 100, "n_new": 0}]
        assert scan_dataset._exclusion_ratio_violations(planned, {"/t": 0}, 0.05) == []

    def test_unknown_total_is_skipped(self):
        planned = [{"target": "/t", "n_final": 99, "existing": 0, "n_new": 99}]
        assert scan_dataset._exclusion_ratio_violations(planned, {}, 0.05) == []

    def test_universe_is_the_pre_exclusion_population(self):
        # The reader supplies this stable total before applying exclusions. A
        # concurrently changed exclusion file must not alter the denominator.
        planned = [{"target": "/t", "n_final": 100, "existing": 100, "n_new": 10}]
        v = scan_dataset._exclusion_ratio_violations(planned, {"/t": 100}, 0.05)
        assert len(v) == 1 and v[0][1] == 100 and v[0][2] == 100

    def test_target_totals_ignore_post_exclusion_length(self, monkeypatch):
        class Leaf:
            _n_episodes_before_exclusions = 100
            _eps_df = [object()] * 3

        monkeypatch.setattr(scan_dataset, "_leaf_info", lambda _leaf: ("lerobot", "/t"))

        assert scan_dataset._target_episode_totals([Leaf()]) == {"/t": 100}


class TestCmdEmitUnknownKind:
    def test_raises_on_unknown_kind(self, tmp_path):
        out_dir = tmp_path / "scan_out" / "myconfig"
        out_dir.mkdir(parents=True)
        (out_dir / "failures.shard0-of-1.jsonl").write_text(
            json.dumps({"kind": "mystery", "target": "/x", "key": 0}) + "\n"
        )

        class Args:
            config = "myconfig"
            out_dir = str(tmp_path / "scan_out")
            dry_run = True
            max_exclude_frac = 0.05
            force = False

        # Unknown kind must abort loudly (was previously a silent drop). cmd_emit
        # attempts a single dataset build first, but for this bogus config the
        # build fails soft (→ leaves=None, re-probe skipped), so no real Hydra
        # config / on-disk dataset is required and the unknown-kind abort still fires.
        with pytest.raises(SystemExit):
            scan_dataset.cmd_emit(Args())


class TestCmdEmitLockedTransaction:
    @staticmethod
    def _write_failures(out_dir: Path, target: Path, episode_indices: list[int]) -> None:
        out_dir.mkdir(parents=True)
        records = [
            {
                "kind": "lerobot",
                "target": str(target),
                "key": episode_index,
                "local": -1,
                "err": "truncated video",
            }
            for episode_index in episode_indices
        ]
        (out_dir / "failures.shard0-of-1.jsonl").write_text("".join(json.dumps(record) + "\n" for record in records))

    @staticmethod
    def _inject_publish_before_lock(monkeypatch, target: Path, episode_indices: list[int]) -> None:
        real_lock = scan_dataset.locked_exclusion_files

        @contextmanager
        def competing_publish(paths):
            # Models another cooperating writer completing after cmd_emit's
            # slow build/reprobe phase but before this transaction gets the lock.
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({"episode_indices": episode_indices}))
            with real_lock(paths):
                yield

        monkeypatch.setattr(scan_dataset, "locked_exclusion_files", competing_publish)

    def test_rereads_and_merges_after_acquiring_lock(self, tmp_path, monkeypatch):
        scan_root = tmp_path / "scan_out"
        out_dir = scan_root / "dataset"
        target = tmp_path / "dataset" / "meta" / "excluded_episodes.json"
        self._write_failures(out_dir, target, [2])
        self._inject_publish_before_lock(monkeypatch, target, [1])
        monkeypatch.setattr(scan_dataset, "_build_leaves", lambda _config: None)

        class Args:
            config = "dataset"
            out_dir = str(scan_root)
            dry_run = False
            max_exclude_frac = 0.05
            force = False

        assert scan_dataset.cmd_emit(Args()) == 0
        assert json.loads(target.read_text())["episode_indices"] == [1, 2]

    def test_guardrail_rechecks_fresh_cumulative_population(self, tmp_path, monkeypatch):
        scan_root = tmp_path / "scan_out"
        out_dir = scan_root / "dataset"
        target = tmp_path / "dataset" / "meta" / "excluded_episodes.json"
        self._write_failures(out_dir, target, [3, 4, 5])
        self._inject_publish_before_lock(monkeypatch, target, [0, 1, 2])
        monkeypatch.setattr(scan_dataset, "_build_leaves", lambda _config: [object()])
        monkeypatch.setattr(scan_dataset, "_target_episode_totals", lambda _leaves: {str(target): 100})

        class Args:
            config = "dataset"
            out_dir = str(scan_root)
            dry_run = False
            max_exclude_frac = 0.05
            force = False
            reprobe = False

        assert scan_dataset.cmd_emit(Args()) == 3
        # The competing writer's valid update remains byte-for-byte logical
        # state; this emit must not publish the freshly merged 6% plan.
        assert json.loads(target.read_text())["episode_indices"] == [0, 1, 2]


class TestCmdEmitPreservesDroidOwnership:
    def test_overlapping_generic_failure_is_recorded_as_independently_owned(self, tmp_path, monkeypatch):
        out_dir = tmp_path / "scan_out" / "droid"
        out_dir.mkdir(parents=True)
        target = tmp_path / "Droid" / "meta" / "excluded_episodes.json"
        target.parent.mkdir(parents=True)
        (target.parent / "episodes").mkdir()
        (target.parent / "info.json").write_text(
            json.dumps(
                {
                    "fps": 10,
                    "total_frames": 1,
                    "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
                }
            )
        )
        pd.DataFrame(
            {
                "episode_index": [0],
                "length": [1],
                "dataset_from_index": [0],
                "data/chunk_index": [0],
                "data/file_index": [0],
            }
        ).to_parquet(target.parent / "episodes" / "chunk-000.parquet")
        data_dir = target.parents[1] / "data" / "chunk-000"
        data_dir.mkdir(parents=True)
        pd.DataFrame({"task_index": [0]}, index=pd.Index([""], name="task")).to_parquet(target.parent / "tasks.parquet")
        pd.DataFrame(
            {
                "episode_index": [0],
                "task_index": [0],
                **{column: [""] for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
            }
        ).to_parquet(data_dir / "file-000.parquet")
        prompt = {
            "schema_version": DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
            "fallback_chain": list(OxeDroidDataset.PROMPT_FALLBACK_COLS),
            "episode_indices": [0],
            "independently_owned_episode_indices": [],
            "latest_scan": {
                "episode_indices": [0],
                "stats": {
                    "rows_scanned": 1,
                    "unresolved_rows": 1,
                    "episodes_all_unresolved": 1,
                    "episodes_partially_unresolved": 0,
                    "task_index_missing_from_tasks_parquet": 0,
                    "fallback_chain": list(OxeDroidDataset.PROMPT_FALLBACK_COLS),
                },
                DROID_PROMPT_INPUTS_DIGEST_KEY: compute_droid_prompt_inputs_digest(target.parents[1]),
            },
        }
        target.write_text(
            json.dumps(
                {
                    "episode_indices": [0],
                    "reason": "preserve me",
                    "droid_prompt_exclusions": prompt,
                }
            )
        )
        (out_dir / "failures.shard0-of-1.jsonl").write_text(
            json.dumps(
                {
                    "kind": "lerobot",
                    "target": str(target),
                    "key": 0,
                    "local": -1,
                    "err": "truncated video",
                }
            )
            + "\n"
        )
        monkeypatch.setattr(scan_dataset, "_build_leaves", lambda _config: None)

        class Args:
            config = "droid"
            out_dir = str(tmp_path / "scan_out")
            dry_run = False
            max_exclude_frac = 0.05
            force = False

        assert scan_dataset.cmd_emit(Args()) == 0
        payload = json.loads(target.read_text())
        assert payload["episode_indices"] == [0]
        assert payload["reason"] == "preserve me"
        assert payload["droid_prompt_exclusions"]["latest_scan"] == prompt["latest_scan"]
        assert payload["droid_prompt_exclusions"]["independently_owned_episode_indices"] == [0]
        _, canonical = load_droid_prompt_exclusions(target.parents[1])
        assert canonical == {0}

    def test_null_droid_namespace_is_rejected(self):
        payload = {
            "episode_indices": [],
            "droid_prompt_exclusions": None,
        }

        with pytest.raises(ValueError, match="must be an object"):
            scan_dataset._merge_lerobot_payload(payload, {0})
