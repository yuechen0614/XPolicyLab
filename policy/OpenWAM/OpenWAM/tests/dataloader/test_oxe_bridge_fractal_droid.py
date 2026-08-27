"""Unit tests for Bridge / Fractal / DROID OXE readers.

All three readers share the LeRobotV3Reader scaffolding; their differences
are: head camera key, optional wrist camera, state schema (BC-Z-style
Euler(8) / Fractal quat(8) / DROID gripper-mount Euler(6) + closedness), action
source column, and prompt
source (Bridge / Fractal read tasks_annotated.parquet by episode_index; DROID
reads tasks.parquet by task_index plus a per-frame fallback chain).
This file exercises each through synthetic LeRobot v3 buckets with
mocked video decode.
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
from scipy.spatial.transform import Rotation as _R

from openwam.dataloader.deprecated.oxe_bridge import OxeBridgeDataset
from openwam.dataloader.deprecated.oxe_fractal import OxeFractalDataset
from openwam.dataloader.oxe_droid import (
    DROID_DATA_POPULATION_DIGEST_KEY,
    DROID_EEF_STATS_CONTRACT,
    DROID_EEF_STATS_CONTRACT_KEY,
    DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
    DROID_PROMPT_INPUTS_DIGEST_KEY,
    DROID_STATS_POPULATION_KEY,
    OxeDroidDataset,
    _clean_text,
    compute_droid_prompt_inputs_digest,
    digest_droid_prompt_shard,
    resolve_droid_stats_population,
)
from openwam.dataloader.utils.eef import EEF_DIM
from openwam.dataloader.utils.lerobotv3 import (
    digest_lerobot_v3_data_population,
    resolve_lerobot_v3_data_population,
)
from openwam.dataloader.utils.normalization import ROT6D_DIMS_ARM10
from openwam.dataloader.utils.oxe_schema import droid_euler7_to_arm10
from openwam.dataloader.utils.stats_computation.oxe_stats_computation import compute_dataset_stats

EP_LENGTH = 60


def _write_episodes(
    bucket: Path,
    n_episodes: int,
    head_cam: str,
    wrist_cam: str | None = None,
    extra_cams: tuple[str, ...] = (),
) -> None:
    eps_dir = bucket / "meta" / "episodes"
    eps_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    cum = 0
    for ep in range(n_episodes):
        r = {
            "episode_index": ep,
            "length": EP_LENGTH,
            "dataset_from_index": cum,
            "data/chunk_index": 0,
            "data/file_index": 0,
            f"videos/{head_cam}/chunk_index": 0,
            f"videos/{head_cam}/file_index": 0,
        }
        if wrist_cam is not None:
            r[f"videos/{wrist_cam}/chunk_index"] = 0
            r[f"videos/{wrist_cam}/file_index"] = 0
        for cam in extra_cams:
            r[f"videos/{cam}/chunk_index"] = 0
            r[f"videos/{cam}/file_index"] = 0
        rows.append(r)
        cum += EP_LENGTH
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), eps_dir / "chunk-000.parquet")


def _write_tasks(bucket: Path) -> None:
    pd.DataFrame({"task_index": [0]}, index=pd.Index(["pick the block"], name="task")).to_parquet(
        bucket / "meta" / "tasks.parquet"
    )


def _write_tasks_annotated(bucket: Path, n_episodes: int) -> None:
    """Per-episode LLM-rewritten prompts (what the reader actually consumes)."""
    df = pd.DataFrame(
        {"task": [f"pick the block — episode {i}" for i in range(n_episodes)]},
        index=pd.Index(range(n_episodes), name="episode_index"),
    )
    df.to_parquet(bucket / "meta" / "tasks_annotated.parquet")


def _write_video_placeholder(bucket: Path, cam: str) -> None:
    vid_dir = bucket / "videos" / cam / "chunk-000"
    vid_dir.mkdir(parents=True, exist_ok=True)
    (vid_dir / "file-000.mp4").write_bytes(b"")


def _write_eef_stats(bucket: Path, excluded_episode_indices: list[int] | None = None) -> None:
    stats = {
        "n_samples": EP_LENGTH * 2,
        "min": [-1.0] * 10,
        "max": [1.0] * 10,
        "mean": [0.0] * 10,
        "std": [0.5] * 10,
        "q01": [-0.9] * 10,
        "q99": [0.9] * 10,
    }
    if excluded_episode_indices is not None:
        stats["excluded_episode_indices"] = excluded_episode_indices
        population = resolve_lerobot_v3_data_population(bucket)
        stats[DROID_DATA_POPULATION_DIGEST_KEY] = digest_lerobot_v3_data_population(population)
        _, stats_population = resolve_droid_stats_population(
            population,
            json.loads((bucket / "meta" / "info.json").read_text()),
            excluded_episode_indices,
            split="train",
        )
        stats[DROID_STATS_POPULATION_KEY] = stats_population
        stats[DROID_EEF_STATS_CONTRACT_KEY] = dict(DROID_EEF_STATS_CONTRACT)
    (bucket / "meta" / "eef_stats.json").write_text(json.dumps(stats))


@contextmanager
def _mock_decoder():
    def _fake(path, frame_indices, h, w):
        return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


def _write_bcz_style_data(bucket: Path, n_rows: int) -> None:
    """Write data parquet matching BC-Z / Bridge schema (8-D state + 7-D action)."""
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(7)
    state = rng.uniform(-0.5, 0.5, size=(n_rows, 8)).astype(np.float32)
    state[:, 6] = 0.0
    state[:, 7] = rng.uniform(0, 1, size=n_rows)
    action = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    action[:, 6] = rng.uniform(0, 1, size=n_rows)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _write_fractal_data(bucket: Path, n_rows: int) -> None:
    """Write Fractal data: state[8] with quat xyzw at [3:7]."""
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(11)
    state = np.zeros((n_rows, 8), dtype=np.float32)
    state[:, :3] = rng.uniform(-0.5, 0.5, size=(n_rows, 3))
    state[:, 3:7] = _R.random(n_rows, random_state=rng).as_quat().astype(np.float32)
    state[:, 7] = rng.uniform(0, 1, size=n_rows)
    action = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    action[:, 6] = rng.uniform(0, 1, size=n_rows)
    df = pd.DataFrame(
        {
            "task_index": np.zeros(n_rows, dtype=np.int64),
            "observation.state": list(state),
            "action": list(action),
        }
    )
    pq.write_table(pa.Table.from_pandas(df), data_dir / "file-000.parquet")


def _write_droid_data(bucket: Path, n_rows: int, fallback_texts: dict[str, str] | None = None) -> None:
    """Write DROID arm-side pose streams plus physically distinct sentinels.

    ``fallback_texts`` overrides the per-frame instruction / annotation columns
    the reader falls back to when the tasks.parquet text is blank; by default
    every one of them is empty so only the tasks.parquet path is exercised.
    """
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(13)
    state = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    state[:, :6] = 77.0  # achieved task-TCP sentinel: only state[6] is valid here
    state[:, 6] = rng.uniform(0, 1, size=n_rows)  # gripper
    state_gripper_pose = rng.uniform(-0.5, 0.5, size=(n_rows, 6)).astype(np.float32)
    action_wrist = rng.uniform(-0.5, 0.5, size=(n_rows, 7)).astype(np.float32)
    action_wrist[:, 6] = rng.uniform(0, 1, size=n_rows)
    cols = {
        "episode_index": np.arange(n_rows, dtype=np.int64) // EP_LENGTH,
        "task_index": np.zeros(n_rows, dtype=np.int64),
        "state": list(state),
        "other_information.observation_gripper_pose6d": list(state_gripper_pose),
        "other_information.action_wrist_pose": list(action_wrist),
        # Physically distinct task-TCP stream the reader must NOT consume.
        "other_information.action_tcp_pose": list(np.full((n_rows, 7), 88.0, dtype=np.float32)),
        # The clipped-delta column the reader must NOT read — filled with a
        # sentinel so a regression that picks it up is obvious.
        "action": list(np.full((n_rows, 7), 99.0, dtype=np.float32)),
    }
    fallback_texts = fallback_texts or {}
    for col in OxeDroidDataset.PROMPT_FALLBACK_COLS:
        cols[col] = [fallback_texts.get(col, "")] * n_rows
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(cols)), data_dir / "file-000.parquet")


def _make_info(bucket: Path) -> None:
    info = {
        "fps": 10.0,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (bucket / "meta" / "info.json").write_text(json.dumps(info))


# ---------------------------------------------------------------------------
# Bridge tests
# ---------------------------------------------------------------------------


def _make_bridge_bucket(tmp_path: Path, n_episodes: int = 2) -> Path:
    b = tmp_path / "Bridge-Dataset"
    b.mkdir(parents=True, exist_ok=True)
    (b / "meta").mkdir(exist_ok=True)
    _make_info(b)
    _write_episodes(b, n_episodes, head_cam="observation.images.image_0")
    _write_tasks(b)
    _write_tasks_annotated(b, n_episodes)
    _write_bcz_style_data(b, n_episodes * EP_LENGTH)
    _write_video_placeholder(b, "observation.images.image_0")
    _write_eef_stats(b)
    return b


class TestBridge:
    def test_loads(self, tmp_path):
        b = _make_bridge_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeBridgeDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"].shape == (32, EEF_DIM)
        assert sample["action_mask"].shape == (32, EEF_DIM)
        assert sample["proprio_mask"].shape == (1, EEF_DIM)

    def test_left_arm_filled(self, tmp_path):
        b = _make_bridge_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeBridgeDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"][:, :10].abs().sum() > 0
        assert (sample["action"][:, 10:] == 0).all()
        assert sample["proprio_mask"][0, :10].all()
        assert not sample["proprio_mask"][0, 10:].any()

    def test_uses_image_0_camera(self, tmp_path):
        b = _make_bridge_bucket(tmp_path)
        ds = OxeBridgeDataset(dataset_dir=str(b))
        assert ds.HEAD_CAMERA == "observation.images.image_0"


# ---------------------------------------------------------------------------
# Fractal tests
# ---------------------------------------------------------------------------


def _make_fractal_bucket(tmp_path: Path, n_episodes: int = 2) -> Path:
    b = tmp_path / "Fractal-Dataset"
    b.mkdir(parents=True, exist_ok=True)
    (b / "meta").mkdir(exist_ok=True)
    _make_info(b)
    _write_episodes(b, n_episodes, head_cam="observation.images.image")
    _write_tasks(b)
    _write_tasks_annotated(b, n_episodes)
    _write_fractal_data(b, n_episodes * EP_LENGTH)
    _write_video_placeholder(b, "observation.images.image")
    _write_eef_stats(b)
    return b


class TestFractal:
    def test_loads(self, tmp_path):
        b = _make_fractal_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeFractalDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"].shape == (32, EEF_DIM)
        # quat-based proprio still produces valid 20-D
        assert sample["proprio"].shape == (1, EEF_DIM)

    def test_quat_sanity_check_passes_on_unit_quats(self, tmp_path):
        # _post_init reads sample state and asserts ‖q‖≈1; our fake data
        # uses scipy random_state which produces unit quats — should not raise.
        b = _make_fractal_bucket(tmp_path)
        OxeFractalDataset(dataset_dir=str(b))  # no raise

    def test_quat_sanity_check_fails_on_non_unit_quats(self, tmp_path):
        b = _make_fractal_bucket(tmp_path)
        # Overwrite data parquet with non-unit quats (0.5 norm)
        data_path = b / "data" / "chunk-000" / "file-000.parquet"
        df = pd.read_parquet(data_path)
        states = np.stack(df["observation.state"].values).copy()
        states[:, 3:7] = 0.5  # all 0.5 → norm sqrt(0.25*4)=1.0 actually — make it
        states[:, 3] = 0.1
        states[:, 4] = 0.1
        states[:, 5] = 0.1
        states[:, 6] = 0.1  # norm sqrt(0.04) ≈ 0.2 — way off
        df["observation.state"] = list(states)
        pq.write_table(pa.Table.from_pandas(df), data_path)
        import pytest

        with pytest.raises(ValueError, match="Quaternion norm check failed"):
            OxeFractalDataset(dataset_dir=str(b))


# ---------------------------------------------------------------------------
# DROID tests
# ---------------------------------------------------------------------------


_PRIMARY = "observation.images.primary"
_WRIST = "observation.images.wrist"
# Synthetic second head key. The re-converted DROID bucket has ONE exterior
# column (the two exterior lenses are separate episodes), so this exists only to
# keep the base class's head_camera_choices feature under test — see
# TestHeadViewSampling. Deliberately NOT a superstring of _PRIMARY so the
# path-matching tinted decoder below can tell the two apart.
_ALT_HEAD = "observation.images.alt_head_for_test"


def _write_droid_exclusions(
    bucket: Path,
    episode_indices: list[int] | None = None,
    prompt_episode_indices: list[int] | None = None,
    independently_owned_episode_indices: list[int] | None = None,
) -> None:
    canonical = episode_indices or []
    prompt_owned = prompt_episode_indices or []
    independently_owned = (
        independently_owned_episode_indices
        if independently_owned_episode_indices is not None
        else sorted(set(canonical) - set(prompt_owned))
    )
    rows_scanned = resolve_lerobot_v3_data_population(bucket).total_rows
    scan_stats = {
        "rows_scanned": rows_scanned,
        "unresolved_rows": len(prompt_owned) * EP_LENGTH,
        "episodes_all_unresolved": len(prompt_owned),
        "episodes_partially_unresolved": 0,
        "task_index_missing_from_tasks_parquet": 0,
        "fallback_chain": list(OxeDroidDataset.PROMPT_FALLBACK_COLS),
    }
    payload = {
        "episode_indices": canonical,
        "droid_prompt_exclusions": {
            "schema_version": DROID_PROMPT_EXCLUSION_SCHEMA_VERSION,
            "fallback_chain": list(OxeDroidDataset.PROMPT_FALLBACK_COLS),
            "episode_indices": prompt_owned,
            "independently_owned_episode_indices": independently_owned,
            "latest_scan": {
                "episode_indices": prompt_owned,
                "stats": scan_stats,
                DROID_PROMPT_INPUTS_DIGEST_KEY: compute_droid_prompt_inputs_digest(bucket),
            },
        },
    }
    (bucket / "meta" / "excluded_episodes.json").write_text(json.dumps(payload))


def _make_droid_bucket(
    tmp_path: Path,
    n_episodes: int = 2,
    with_alt_head: bool = False,
    task_text: str = "pick the block",
    fallback_texts: dict[str, str] | None = None,
) -> Path:
    b = tmp_path / "DROID-Dataset"
    b.mkdir(parents=True, exist_ok=True)
    (b / "meta").mkdir(exist_ok=True)
    _make_info(b)
    _write_episodes(
        b,
        n_episodes,
        head_cam=_PRIMARY,
        wrist_cam=_WRIST,
        extra_cams=(_ALT_HEAD,) if with_alt_head else (),
    )
    # Prompts come from meta/tasks.parquet by task_index (no tasks_annotated.parquet).
    pd.DataFrame({"task_index": [0]}, index=pd.Index([task_text], name="task")).to_parquet(b / "meta" / "tasks.parquet")
    _write_droid_data(b, n_episodes * EP_LENGTH, fallback_texts=fallback_texts)
    _write_video_placeholder(b, _PRIMARY)
    _write_video_placeholder(b, _WRIST)
    if with_alt_head:
        _write_video_placeholder(b, _ALT_HEAD)
    _write_droid_exclusions(b)
    _write_eef_stats(b, excluded_episode_indices=[])
    return b


class TestDroid:
    def test_loads(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            sample = ds[0]
        assert sample["action"].shape == (32, EEF_DIM)
        assert sample["proprio"].shape == (1, EEF_DIM)

    def test_loader_forces_rot6d_passthrough_for_unpinned_stats(self, tmp_path):
        # The synthetic eef_stats file deliberately uses q01/q99=±0.9 on every
        # field.  The reader must still preserve rot6d exactly rather than rely
        # on a generator-time convention.
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            raw = OxeDroidDataset(dataset_dir=str(b), normalize_mode=None)[0]
            norm = OxeDroidDataset(dataset_dir=str(b), normalize_mode="quantile")[0]
        dims = list(ROT6D_DIMS_ARM10)
        np.testing.assert_allclose(
            norm["action"][:, dims].numpy(),
            raw["action"][:, dims].numpy(),
            atol=1e-6,
        )
        np.testing.assert_allclose(
            norm["proprio"][:, dims].numpy(),
            raw["proprio"][:, dims].numpy(),
            atol=1e-6,
        )

    def test_uses_absolute_wrist_pose_not_tcp_or_delta_action(self, tmp_path):
        # The shared frame is the rigid arm-side wrist / gripper mount. The
        # moving task-TCP and top-level clipped delta must stay excluded.
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b), normalize_mode=None)
        assert "other_information.action_wrist_pose" in ds.NEEDED_COLS
        assert "other_information.action_tcp_pose" not in ds.NEEDED_COLS
        assert "action" not in ds.NEEDED_COLS
        win = ds._load_data_table(0, 0).slice(0, EP_LENGTH).to_pandas()
        raw_wrist = np.stack(win["other_information.action_wrist_pose"].values)
        action = ds._action_20d(win)
        np.testing.assert_allclose(action[:, :10], droid_euler7_to_arm10(raw_wrist), atol=1e-7)
        # Neither excluded stream's 88/99 sentinel reaches the raw payload.
        assert np.abs(action).max() < 10.0

    def test_uses_gripper_mount_state_pose_plus_state_gripper(self, tmp_path):
        # Pose comes from observation_gripper_pose6d, while state contributes
        # only closedness at index 6; state[:6] is the excluded task-TCP.
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b), normalize_mode=None)
        assert "state" in ds.NEEDED_COLS
        assert "other_information.observation_gripper_pose6d" in ds.NEEDED_COLS
        assert "observation.state.cartesian_position" not in ds.NEEDED_COLS
        assert "observation.state.gripper_position" not in ds.NEEDED_COLS
        win = ds._load_data_table(0, 0).slice(0, EP_LENGTH).to_pandas()
        proprio = ds._proprio_20d(win)
        expected_pose = np.stack(win["other_information.observation_gripper_pose6d"].values[:1])
        np.testing.assert_allclose(proprio[:, :3], expected_pose[:, :3], atol=1e-7)
        assert np.abs(proprio[:, :3]).max() < 10.0  # state[:6] sentinel is 77

    def test_action_and_proprio_grippers_use_canonical_open_scale(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b), normalize_mode=None)
        win = ds._load_data_table(0, 0).slice(0, EP_LENGTH).to_pandas()
        raw_action = np.stack(win["other_information.action_wrist_pose"].values)
        raw_state = np.stack(win["state"].values[:1])

        action = ds._action_20d(win)
        proprio = ds._proprio_20d(win)

        np.testing.assert_allclose(action[:, 9], 1.0 - raw_action[:, 6], atol=1e-7)
        np.testing.assert_allclose(proprio[:, 9], 1.0 - raw_state[:, 6], atol=1e-7)

    @pytest.mark.parametrize(
        "bad_contract",
        [
            None,
            {**DROID_EEF_STATS_CONTRACT, "schema_version": True},
            {**DROID_EEF_STATS_CONTRACT, "action_pose_source": "other_information.action_tcp_pose"},
            {**DROID_EEF_STATS_CONTRACT, "gripper_transform": "raw"},
        ],
    )
    def test_stale_gripper_stats_contract_fails_at_construction(self, tmp_path, bad_contract):
        b = _make_droid_bucket(tmp_path)
        stats_path = b / "meta" / "eef_stats.json"
        stats = json.loads(stats_path.read_text())
        if bad_contract is None:
            stats.pop(DROID_EEF_STATS_CONTRACT_KEY)
        else:
            stats[DROID_EEF_STATS_CONTRACT_KEY] = bad_contract
        stats_path.write_text(json.dumps(stats))

        with pytest.raises(ValueError, match="normalization provenance|gripper transform"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_camera_layout_has_left_wrist(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        ds = OxeDroidDataset(dataset_dir=str(b), multiview=True)
        # multiview=True → [head, left_wrist, right_wrist_or_missing]
        assert ds._camera_layout[0] == _PRIMARY
        assert ds._camera_layout[1] == _WRIST
        # right_wrist is None → "__missing_right__"
        assert ds._camera_layout[2] == "__missing_right__"

    def test_prompt_from_tasks_parquet(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            assert ds[0]["prompt"] == "pick the block"

    # Every distinct placeholder text actually present in the bucket, from a full
    # scan of all 46,259,014 rows (counts are episodes-worth of rows collapsed to
    # the distinct strings). _clean_text must reject all of them, else the junk
    # text is trained on as a prompt instead of falling through to the chain.
    OBSERVED_PLACEHOLDERS = [
        "No action",
        "No action.",
        "Not action",
        "Not Action",
        "no action",
        "NULL",
        "OP",
        "Pree",
        "No Action",
        "N/A",
        "Pm",
        "No instruction",
    ]

    @pytest.mark.parametrize("text", OBSERVED_PLACEHOLDERS)
    def test_every_observed_placeholder_is_rejected(self, text):
        assert _clean_text(text) == "", f"{text!r} leaked through as a prompt"

    @pytest.mark.parametrize("blank", ["", "   ", "No action", "NULL", "n/a", "Pm"])
    def test_blank_or_placeholder_task_falls_back(self, tmp_path, blank):
        """13.96% of episodes carry a blank task and 532 carry a placeholder;
        both must fall through to the per-frame annotation columns rather than
        training on the junk text."""
        b = _make_droid_bucket(
            tmp_path,
            task_text=blank,
            fallback_texts={"annotation.substask": "pull the fabric to the left"},
        )
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            assert ds[0]["prompt"] == "pull the fabric to the left"

    def test_fallback_chain_order(self, tmp_path):
        # language_instruction_2 precedes substask in the chain (matching the
        # dataset's own cleaning/config.yaml R20 fallback_chain).
        b = _make_droid_bucket(
            tmp_path,
            task_text="",
            fallback_texts={
                "other_information.language_instruction_2": "from instruction 2",
                "annotation.substask": "from substask",
            },
        )
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            assert ds[0]["prompt"] == "from instruction 2"

    def test_all_prompt_sources_blank_raises(self, tmp_path):
        # 426/152,986 episodes bottom out here. _safe_get retries onto the next
        # window; the underlying error must be a loud ValueError, not an empty
        # prompt silently entering training.
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            # Simulate a bucket mutation after construction. A valid scan
            # artifact must not claim this blank episode was scanned clean.
            ds._task_idx_to_text[0] = ""
            row = ds._eps_df.iloc[0]
            win = ds._load_data_table(0, 0).slice(0, 33).to_pandas()
            with pytest.raises(ValueError, match="blank prompt"):
                ds._resolve_prompt(row, win)

    def test_safe_get_cannot_retry_past_a_no_prompt_episode(self, tmp_path):
        """Why meta/excluded_episodes.json is REQUIRED for this bucket.

        _safe_get recovers from a raising window by retrying at ``idx + 1``, but
        with window_stride=1 that is the next frame of the SAME episode, and an
        episode with no resolvable prompt has none on any of its rows. The error
        therefore reaches the DataLoader worker and kills it — the documented
        route to a DDP deadlock. On the shipped bucket 336 of the 426 affected
        episodes are longer than _GETITEM_MAX_RETRIES, and a 16-worker shuffled
        loader died at sample 264 before the exclusion list existed.
        """
        b = _make_droid_bucket(tmp_path, n_episodes=1)
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            ds._task_idx_to_text[0] = ""
            with pytest.raises(ValueError, match="blank prompt"):
                ds[0]  # __getitem__ -> _safe_get: the retries do NOT save it

    def test_excluded_episodes_json_drops_them_at_construction(self, tmp_path):
        """The remedy: the base reader drops listed episodes in __init__, after
        the offsets are computed, so the survivors stay alignment-safe."""
        b = _make_droid_bucket(tmp_path, n_episodes=3)
        plain = OxeDroidDataset(dataset_dir=str(b))
        _write_droid_exclusions(b, episode_indices=[1])
        _write_eef_stats(b, excluded_episode_indices=[1])
        ds = OxeDroidDataset(dataset_dir=str(b))
        assert len(ds._eps_df) == len(plain._eps_df) - 1
        assert 1 not in set(ds._eps_df["episode_index"].astype(int))
        assert len(ds) == len(plain) - EP_LENGTH
        with _mock_decoder():
            assert ds[0]["prompt"] == "pick the block"

    def test_missing_prompt_exclusion_artifact_fails_at_construction(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        (b / "meta" / "excluded_episodes.json").unlink()

        with pytest.raises(FileNotFoundError, match="write_droid_prompt_exclusions"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_generic_exclusions_without_prompt_provenance_fail_at_construction(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        (b / "meta" / "excluded_episodes.json").write_text(json.dumps({"episode_indices": [1]}))
        _write_eef_stats(b, excluded_episode_indices=[1])

        with pytest.raises(ValueError, match="stale or malformed"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_validated_exclusion_snapshot_is_not_read_twice(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        exclusion_path = b / "meta" / "excluded_episodes.json"

        import builtins

        real_open = builtins.open

        def _reject_second_open(path, *args, **kwargs):
            try:
                candidate = Path(path)
            except TypeError:
                candidate = None
            if candidate == exclusion_path:
                raise AssertionError("excluded_episodes.json was reopened after certificate validation")
            return real_open(path, *args, **kwargs)

        with patch("builtins.open", side_effect=_reject_second_open):
            OxeDroidDataset(dataset_dir=str(b))

    @pytest.mark.parametrize(
        "payload",
        [
            "{",
            "[]",
        ],
    )
    def test_malformed_prompt_exclusion_artifact_fails_at_construction(self, tmp_path, payload):
        b = _make_droid_bucket(tmp_path)
        (b / "meta" / "excluded_episodes.json").write_text(payload)

        with pytest.raises(ValueError, match="stale or malformed"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_prompt_exclusion_without_completed_scan_fails_at_construction(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        path = b / "meta" / "excluded_episodes.json"
        payload = json.loads(path.read_text())
        del payload["droid_prompt_exclusions"]["latest_scan"]
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="stale or malformed"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_prompt_scan_row_count_must_match_bucket_metadata(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        info_path = b / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["total_frames"] = EP_LENGTH * 2 + 1
        info_path.write_text(json.dumps(info))

        with pytest.raises(ValueError, match="episodes manifest addresses"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_recorded_prompt_scan_row_count_must_match_manifest(self, tmp_path):
        b = _make_droid_bucket(tmp_path)
        exclusion_path = b / "meta" / "excluded_episodes.json"
        payload = json.loads(exclusion_path.read_text())
        payload["droid_prompt_exclusions"]["latest_scan"]["stats"]["rows_scanned"] += 1
        exclusion_path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="data manifest addresses"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_task_text_change_with_same_row_count_invalidates_prompt_scan(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=1)
        pd.DataFrame(
            {"task_index": [0]},
            index=pd.Index([""], name="task"),
        ).to_parquet(b / "meta" / "tasks.parquet")

        with pytest.raises(ValueError, match="prompt_inputs_digest"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_manifest_episode_mapping_change_invalidates_prompt_scan(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=2)
        manifest_path = b / "meta" / "episodes" / "chunk-000.parquet"
        manifest = pd.read_parquet(manifest_path)
        manifest["episode_index"] = manifest["episode_index"].iloc[::-1].to_numpy()
        manifest.to_parquet(manifest_path)

        with pytest.raises(ValueError, match="prompt source population"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_manifest_address_metadata_change_invalidates_prompt_digest(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=1)
        manifest_path = b / "meta" / "episodes" / "chunk-000.parquet"
        manifest = pd.read_parquet(manifest_path)
        manifest["dataset_from_index"] = 7
        manifest.to_parquet(manifest_path)

        with pytest.raises(ValueError, match="prompt_inputs_digest"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_data_path_change_to_identical_rows_invalidates_prompt_digest(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=1)
        custom_dir = b / "custom"
        custom_dir.mkdir()
        source = b / "data" / "chunk-000" / "file-000.parquet"
        source.replace(custom_dir / "shard-0.parquet")
        info_path = b / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["data_path"] = "custom/shard-{file_index}.parquet"
        info_path.write_text(json.dumps(info))

        with pytest.raises(ValueError, match="prompt_inputs_digest"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_unreferenced_data_parquet_does_not_change_prompt_digest(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=1)
        before = compute_droid_prompt_inputs_digest(b)
        frame_data = pd.read_parquet(b / "data" / "chunk-000" / "file-000.parquet")
        frame_data["episode_index"] = 999
        frame_data.to_parquet(b / "data" / "chunk-000" / "backup.parquet")

        assert compute_droid_prompt_inputs_digest(b) == before
        OxeDroidDataset(dataset_dir=str(b))

    def test_missing_manifest_data_file_is_not_replaced_by_backup(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=1)
        data_path = b / "data" / "chunk-000" / "file-000.parquet"
        data_path.rename(data_path.with_name("backup.parquet"))

        with pytest.raises(ValueError, match="prompt source population"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_prompt_shard_digest_normalizes_dictionary_chunks_with_null_values(self):
        dictionary_text = pa.chunked_array(
            [
                pa.DictionaryArray.from_arrays(pa.array([0, 1]), pa.array(["first", None])),
                pa.DictionaryArray.from_arrays(pa.array([0, 1]), pa.array([None, "second"])),
            ]
        )
        plain_text = pa.chunked_array([pa.array(["first", None]), pa.array([None, "second"])])
        columns = {
            "episode_index": pa.array([0, 0, 1, 1]),
            "task_index": pa.array([0, 0, 0, 0]),
            **{column: pa.array([""] * 4) for column in OxeDroidDataset.PROMPT_FALLBACK_COLS},
        }
        dictionary_table = pa.table(
            {
                **columns,
                OxeDroidDataset.PROMPT_FALLBACK_COLS[0]: dictionary_text,
            }
        )
        plain_table = pa.table(
            {
                **columns,
                OxeDroidDataset.PROMPT_FALLBACK_COLS[0]: plain_text,
            }
        )

        assert digest_droid_prompt_shard(dictionary_table) == digest_droid_prompt_shard(plain_table)

    @pytest.mark.parametrize(
        "column",
        ["episode_index", "task_index", *OxeDroidDataset.PROMPT_FALLBACK_COLS],
    )
    def test_prompt_shard_content_change_with_same_row_count_invalidates_scan(self, tmp_path, column):
        fallback_texts = {column: "valid fallback"} if column in OxeDroidDataset.PROMPT_FALLBACK_COLS else None
        b = _make_droid_bucket(
            tmp_path,
            n_episodes=1,
            task_text="" if fallback_texts else "valid task",
            fallback_texts=fallback_texts,
        )
        data_path = b / "data" / "chunk-000" / "file-000.parquet"
        frame_data = pd.read_parquet(data_path)
        if column == "episode_index":
            frame_data.loc[0, column] = 99
        elif column == "task_index":
            frame_data.loc[0, column] = 12345
        else:
            frame_data.loc[0, column] = ""
        frame_data.to_parquet(data_path)

        error = "prompt source population" if column == "episode_index" else "prompt_inputs_digest"
        with pytest.raises(ValueError, match=error):
            OxeDroidDataset(dataset_dir=str(b))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("algorithm", "md5"),
            ("format_version", 0),
            ("format_version", 1),
            ("format_version", True),
            ("format_version", 1.0),
            ("value", "not-a-sha256"),
        ],
    )
    def test_malformed_prompt_inputs_digest_fails_at_construction(self, tmp_path, field, value):
        b = _make_droid_bucket(tmp_path)
        path = b / "meta" / "excluded_episodes.json"
        payload = json.loads(path.read_text())
        payload["droid_prompt_exclusions"]["latest_scan"][DROID_PROMPT_INPUTS_DIGEST_KEY][field] = value
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="stale or malformed"):
            OxeDroidDataset(dataset_dir=str(b))

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("schema_version", 1),
            ("schema_version", 2),
            ("fallback_chain", list(reversed(OxeDroidDataset.PROMPT_FALLBACK_COLS))),
        ],
    )
    def test_stale_prompt_exclusion_artifact_fails_at_construction(self, tmp_path, field, value):
        b = _make_droid_bucket(tmp_path)
        path = b / "meta" / "excluded_episodes.json"
        payload = json.loads(path.read_text())
        payload["droid_prompt_exclusions"][field] = value
        path.write_text(json.dumps(payload))

        with pytest.raises(ValueError, match="stale or malformed"):
            OxeDroidDataset(dataset_dir=str(b))

    @pytest.mark.parametrize("stats_episode_indices", [None, [0]])
    def test_stale_stats_exclusion_population_fails_at_construction(self, tmp_path, stats_episode_indices):
        b = _make_droid_bucket(tmp_path, n_episodes=2)
        _write_droid_exclusions(b, episode_indices=[1])
        _write_eef_stats(b, excluded_episode_indices=stats_episode_indices)

        with pytest.raises(ValueError, match="excluded_episode_indices"):
            OxeDroidDataset(dataset_dir=str(b))

    @pytest.mark.parametrize("population_digest", [None, "not-a-digest", "0" * 64])
    def test_stale_stats_data_population_fails_at_construction(self, tmp_path, population_digest):
        b = _make_droid_bucket(tmp_path, n_episodes=2)
        stats_path = b / "meta" / "eef_stats.json"
        stats = json.loads(stats_path.read_text())
        if population_digest is None:
            stats.pop(DROID_DATA_POPULATION_DIGEST_KEY)
        else:
            stats[DROID_DATA_POPULATION_DIGEST_KEY] = population_digest
        stats_path.write_text(json.dumps(stats))

        with pytest.raises(ValueError, match="population"):
            OxeDroidDataset(dataset_dir=str(b))

    def test_stats_scan_and_reader_are_bound_to_train_split(self, tmp_path):
        b = _make_droid_bucket(tmp_path, n_episodes=3)
        info_path = b / "meta" / "info.json"
        info = json.loads(info_path.read_text())
        info["splits"] = {"train": "0:1", "val": "1:3"}
        info_path.write_text(json.dumps(info))

        # The pre-change stats file still describes all three episodes. The raw
        # data-population digest did not change, but the effective train
        # certificate must make construction fail closed.
        with pytest.raises(ValueError, match="train split/exclusion population"):
            OxeDroidDataset(dataset_dir=str(b))

        stats, n_state, n_action = compute_dataset_stats(b, "DROID")
        assert n_state == EP_LENGTH
        assert n_action == EP_LENGTH
        assert stats[DROID_STATS_POPULATION_KEY]["split"] == "train"
        assert stats[DROID_STATS_POPULATION_KEY]["num_episodes"] == 1
        assert stats[DROID_STATS_POPULATION_KEY]["num_rows"] == EP_LENGTH

    def test_validated_stats_snapshot_is_materialized_without_second_read(self, tmp_path):
        b = _make_droid_bucket(tmp_path)

        with patch(
            "openwam.dataloader.bases.lerobot_v3_reader.LeRobotV3Reader._load_stats",
            side_effect=AssertionError("validated stats were reopened"),
        ):
            OxeDroidDataset(dataset_dir=str(b))

    def test_task_index_absent_from_tasks_parquet_raises(self, tmp_path):
        # A data row pointing at a task_index tasks.parquet doesn't define is a
        # bucket-integrity bug; it must surface as a KeyError naming the index,
        # not as a silent fabricated prompt. (Zero occurrences in the shipped
        # bucket — all 46,259,014 rows resolve — so this guards regressions in
        # the lookup, not a live data condition.)
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b))
            row = ds._eps_df.iloc[0]
            win = ds._load_data_table(0, 0).slice(0, 33).to_pandas()
            win["task_index"] = 12345
            with pytest.raises(KeyError, match="task_index=12345"):
                ds._resolve_prompt(row, win)

    def test_wrist_decode_failure_is_tolerated(self, tmp_path):
        """When the wrist mp4 raises (missing / corrupt clip on disk), the reader
        must return a usable sample with an empty wrist slot rather than
        propagating the error. Propagating it triggers a DataLoader-worker death
        → rank-0 early iterator EOF → DDP deadlock (rank 0 in
        destroy_process_group, peer ranks still in backward NCCL allreduce)."""
        from unittest.mock import patch

        b = _make_droid_bucket(tmp_path)

        def _decoder(path, frame_indices, h, w):
            # head slot (256x320) succeeds; wrist slot (128x160) raises.
            if (h, w) == (128, 160):
                raise FileNotFoundError(path)
            return [Image.new("RGB", (w, h), (0, 0, 0)) for _ in frame_indices]

        with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_decoder):
            ds = OxeDroidDataset(dataset_dir=str(b), multiview=True)
            sample = ds[0]
        # Sample is usable: head slot present, wrist slot fell back to black.
        assert sample["action"].shape == (32, EEF_DIM)
        assert len(sample["video"]) > 0
        # The fail counter recorded the wrist failures (one per video frame).
        assert ds._wrist_fail_count >= 1


# ---------------------------------------------------------------------------
# Head-view sampling (Octo-style): head_camera_choices makes each TRAIN window
# decode the head slot from one uniformly sampled camera. This is a
# LeRobotV3Reader base feature; no shipped config uses it since the re-converted
# DROID bucket exposes its two exterior lenses as separate EPISODES rather than
# separate camera columns. Exercised through the DROID reader with a synthetic
# second head key (_ALT_HEAD) so the base machinery stays covered.
# ---------------------------------------------------------------------------

_TINTS = {_PRIMARY: (255, 0, 0), _ALT_HEAD: (0, 255, 0), _WRIST: (0, 0, 255)}


@contextmanager
def _camera_tinted_decoder():
    """Mock decoder returning a solid per-camera color (keyed off the mp4 path),
    so tests can tell WHICH camera a frame came from by reading one pixel."""

    def _fake(path, frame_indices, h, w):
        color = next((c for cam, c in _TINTS.items() if cam in str(path)), (255, 255, 255))
        return [Image.new("RGB", (w, h), color) for _ in frame_indices]

    with patch("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", side_effect=_fake):
        yield


class TestHeadViewSampling:
    CHOICES = [_PRIMARY, _ALT_HEAD]

    def test_train_samples_both_heads(self, tmp_path):
        import random as _random

        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        with _camera_tinted_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b), head_camera_choices=self.CHOICES)
            _random.seed(0)
            # Single-view mode: the sample frames ARE the head camera.
            colors = {ds[0]["video"][0].getpixel((5, 5)) for _ in range(20)}
        assert colors == {_TINTS[_PRIMARY], _TINTS[_ALT_HEAD]}

    def test_multiview_head_slot_varies_wrist_fixed(self, tmp_path):
        import random as _random

        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        with _camera_tinted_decoder():
            ds = OxeDroidDataset(
                dataset_dir=str(b), multiview=True, height=384, width=320, head_camera_choices=self.CHOICES
            )
            _random.seed(0)
            top_colors = set()
            for _ in range(20):
                canvas = ds[0]["video"][0]
                top_colors.add(canvas.getpixel((5, 5)))  # head slot (top)
                assert canvas.getpixel((5, 380)) == _TINTS[_WRIST]  # bot-left wrist slot
        assert top_colors == {_TINTS[_PRIMARY], _TINTS[_ALT_HEAD]}
        # Slot names stay the resolved trio — sampling swaps content, not layout.
        assert ds._camera_layout[0] == _PRIMARY

    def test_window_count_unchanged(self, tmp_path):
        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        ds_plain = OxeDroidDataset(dataset_dir=str(b))
        ds_sampled = OxeDroidDataset(dataset_dir=str(b), head_camera_choices=self.CHOICES)
        assert len(ds_sampled) == len(ds_plain)

    def test_val_split_stays_on_head_camera(self, tmp_path):
        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        info = json.loads((b / "meta" / "info.json").read_text())
        info["splits"] = {"train": "0:1", "val": "1:2"}
        (b / "meta" / "info.json").write_text(json.dumps(info))
        # A val reader intentionally uses train-derived normalization stats.
        _write_eef_stats(b, excluded_episode_indices=[])
        with _camera_tinted_decoder():
            ds = OxeDroidDataset(dataset_dir=str(b), split="val", head_camera_choices=self.CHOICES)
            assert ds._head_camera_choices is None
            colors = {ds[0]["video"][0].getpixel((5, 5)) for _ in range(5)}
        assert colors == {_TINTS[_PRIMARY]}

    def test_choices_must_include_head_camera(self, tmp_path):
        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        with pytest.raises(ValueError, match="must include the resolved head camera"):
            OxeDroidDataset(dataset_dir=str(b), head_camera_choices=[_ALT_HEAD])

    def test_choice_without_video_columns_raises(self, tmp_path):
        # Bucket lacks the alt-head index columns → fail fast at init, not as an
        # opaque decode-retry storm in _safe_get.
        b = _make_droid_bucket(tmp_path, with_alt_head=False)
        with pytest.raises(ValueError, match="have no videos/"):
            OxeDroidDataset(dataset_dir=str(b), head_camera_choices=self.CHOICES)

    def test_single_or_duplicate_choice_is_inert(self, tmp_path):
        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        ds = OxeDroidDataset(dataset_dir=str(b), head_camera_choices=[_PRIMARY, _PRIMARY])
        assert ds._head_camera_choices is None

    def test_from_config_passthrough(self, tmp_path):
        b = _make_droid_bucket(tmp_path, with_alt_head=True)
        cfg = {
            "type": "oxe_droid",
            "dataset_dir": str(b),
            "head_camera_choices": list(self.CHOICES),
        }
        ds = OxeDroidDataset.from_config(cfg, split="train")
        assert ds._head_camera_choices == self.CHOICES
        # Both head cameras got per-episode video frame offsets.
        assert _PRIMARY in ds._ep_video_frame_offsets and _ALT_HEAD in ds._ep_video_frame_offsets


# ---------------------------------------------------------------------------
# Normalization modes — every reader must support min-max / z-score / quantile
# (+ null passthrough). All three share LeRobotV3Reader stats loading +
# apply_normalization, so this matrix locks the contract across the readers.
# ---------------------------------------------------------------------------

_READERS = [
    (_make_bridge_bucket, OxeBridgeDataset),
    (_make_fractal_bucket, OxeFractalDataset),
    (_make_droid_bucket, OxeDroidDataset),
]


class TestNormalizeModes:
    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "fractal", "droid"])
    @pytest.mark.parametrize("mode", ["min-max", "z-score", "quantile"])
    def test_mode_loads_stats_and_normalizes(self, tmp_path, make_bucket, cls, mode):
        b = make_bucket(tmp_path)
        with _mock_decoder():
            ds = cls(dataset_dir=str(b), normalize_mode=mode)
            sample = ds[0]
        # Every non-null mode loads the per-dataset stats dict.
        assert ds._normalization_stats is not None
        assert sample["action"].shape == (32, EEF_DIM)
        assert sample["action"][:, :10].abs().sum() > 0
        # All modes must yield finite values.
        assert sample["action"].isfinite().all()
        assert sample["proprio"].isfinite().all()
        # The bounded modes clip the active left-arm dims into [-1, 1];
        # z-score is unbounded by design, so only finiteness is asserted there.
        if mode in ("min-max", "quantile"):
            assert (sample["action"][:, :10].abs() <= 1.0 + 1e-5).all()
            assert (sample["proprio"][:, :10].abs() <= 1.0 + 1e-5).all()

    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "fractal", "droid"])
    def test_null_mode_skips_stats(self, tmp_path, make_bucket, cls):
        b = make_bucket(tmp_path)
        with _mock_decoder():
            ds = cls(dataset_dir=str(b), normalize_mode=None)
            sample = ds[0]
        assert ds._normalization_stats is None
        assert sample["action"].shape == (32, EEF_DIM)


# ---------------------------------------------------------------------------
# enable_action_supervision=False — Bridge / Fractal / DROID.
# Mirrors TestEnableActionSupervisionFalse in test_oxe_bcz.py. Locks the
# video-only-auxiliary contract across the three remaining OXE readers:
# both masks integer-zero everywhere, while action/proprio VALUES and the
# video/prompt payload stay intact so the source still contributes frames.
# ---------------------------------------------------------------------------


class TestEnableActionSupervisionFalse:
    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "fractal", "droid"])
    def test_supervision_off_masks_zero(self, tmp_path, make_bucket, cls):
        b = make_bucket(tmp_path)
        with _mock_decoder():
            ds = cls(dataset_dir=str(b), enable_action_supervision=False)
            s = ds[0]
        # Masks entirely False — no action/proprio supervision signal.
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
        # Shapes unchanged.
        assert s["action_mask"].shape == (32, EEF_DIM)
        assert s["proprio_mask"].shape == (1, EEF_DIM)
        # Values still loaded (just masked out), video/prompt still present.
        assert s["action"][:, :10].abs().sum() > 0
        assert s["proprio"][:, :10].abs().sum() > 0
        assert len(s["video"]) > 0
        assert isinstance(s["prompt"], str) and len(s["prompt"]) > 0

    def test_supervision_off_droid_multiview(self, tmp_path):
        # DROID carries a left wrist camera → exercises the L-shape multiview
        # assembly path with supervision off (head + wrist slots both decoded).
        b = _make_droid_bucket(tmp_path)
        with _mock_decoder():
            ds = OxeDroidDataset(
                dataset_dir=str(b),
                multiview=True,
                height=384,
                width=320,
                enable_action_supervision=False,
            )
            s = ds[0]
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
        assert s["action"].shape == (32, EEF_DIM)
        assert len(s["video"]) > 0

    @pytest.mark.parametrize("make_bucket, cls", _READERS, ids=["bridge", "fractal", "droid"])
    def test_from_config_supervision_off(self, tmp_path, make_bucket, cls):
        # The mixture/training path reads the flag from yaml via from_config;
        # `enable_action_supervision: false` must reach the reader as False
        # (False is not None → forwarded), not get dropped as a falsy value.
        b = make_bucket(tmp_path)
        cfg = {
            "type": cls.DATASET_NAME,
            "dataset_dir": str(b),
            "normalize_mode": "quantile",
            "enable_action_supervision": False,
        }
        with _mock_decoder():
            ds = cls.from_config(cfg, split="train")
            s = ds[0]
        assert ds._enable_action_supervision is False
        assert not s["action_mask"].any()
        assert not s["proprio_mask"].any()
