"""Tests for the deprecated Ego4D reader (openwam/dataloader/deprecated/ego4d.py).

Covers the two Ego4D-specific behaviours on top of the shared LeRobotV3Reader
machinery: English-only prompt extraction from the combined bilingual string,
and dropping episodes whose prompt is the ``"null"`` placeholder. Plus an
end-to-end sample-dict shape check on a synthetic bucket (video decode is
monkeypatched so no mp4 is needed).
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

from openwam.dataloader.deprecated.ego4d import Ego4DDataset, extract_english_prompt

FPS = 1.0
EP_LEN = 40
CAM = "observation.images.ego"


# ---------------------------------------------------------------------------
# Pure English-extraction function
# ---------------------------------------------------------------------------


class TestExtractEnglish:
    def test_ascii_colon(self):
        assert extract_english_prompt("中文:折叠衣服。英文:Fold the clothes.") == "Fold the clothes."

    def test_fullwidth_colon(self):
        # 52 tasks in the real data use the full-width colon after 英文.
        assert extract_english_prompt("中文:测量。英文：Measure with the ruler.") == "Measure with the ruler."

    def test_null_placeholder(self):
        assert extract_english_prompt("null") is None

    def test_no_english_marker(self):
        assert extract_english_prompt("中文:只有中文没有英文标记") is None

    def test_empty_english_half(self):
        assert extract_english_prompt("中文:x。英文:   ") is None

    def test_none_and_empty(self):
        assert extract_english_prompt(None) is None
        assert extract_english_prompt("") is None

    def test_english_may_contain_more_text(self):
        # Regex is greedy to end-of-string, so trailing content stays.
        assert extract_english_prompt("中文:a。英文:Do A, then B.") == "Do A, then B."

    def test_duplicate_marker_takes_last(self):
        # A stray/duplicated 英文 marker inside the Chinese half must not leak
        # Chinese: anchor to the LAST 英文 marker (real data: group_00 ti=4147).
        s = "中文:用右手编织浅蓝色针织物。英文:右手编织浅蓝色针织物。英文:Knit the light blue knitted fabric with the right hand."
        assert extract_english_prompt(s) == "Knit the light blue knitted fabric with the right hand."

    def test_reversed_order_strips_chinese_tail(self):
        # Reversed 英文:…中文:… order (real data: group_11 ti=5188) → drop the
        # trailing Chinese clause.
        s = "英文:Reach for the glass jar of water with the right hand.中文:右手伸向装水的玻璃罐。"
        assert extract_english_prompt(s) == "Reach for the glass jar of water with the right hand."

    def test_stray_han_char_in_english_is_dropped(self):
        # Malformed: a lone Han glyph embedded in the English (real: group_00
        # ti=9928 "…the steam铲…"). No Chinese may reach training → drop it.
        assert extract_english_prompt("中文:x。英文:Hold the steam铲 with left hand.") is None

    def test_english_clause_is_actually_chinese_is_dropped(self):
        # The 英文: clause itself is Chinese (real: group_09 ti=12560) → drop.
        assert extract_english_prompt("中文:摘绿色叶子。英文:摘绿色叶子入不锈钢盆并丢弃茎秆。") is None

    def test_residual_cjk_punctuation_is_dropped(self):
        # Defense-in-depth: a stray CJK full stop 。/ fullwidth colon ： in the
        # English residue → drop (guarantees the emitted prompt is pure English).
        assert extract_english_prompt("英文:put the cup down。中文:放下杯子。") is None
        assert extract_english_prompt("中文:x。英文:Do it：now") is None

    def test_typographic_english_is_kept(self):
        # em-dash / accented / curly quotes are Latin/punctuation, not another
        # script — legitimate English stays (real data: "sautéed", "Español",
        # "car's" with a curly apostrophe).
        assert extract_english_prompt("中文:x。英文:Pick up the cup—slowly.") == "Pick up the cup—slowly."
        assert extract_english_prompt("中文:x。英文:Café table.") == "Café table."
        assert extract_english_prompt("中文:x。英文:Open the car’s door.") == "Open the car’s door."

    def test_other_script_contamination_is_dropped(self):
        # Non-Latin script leaking into the English (real: group_11 "Rinseام
        # glass…" has Arabic) → drop; only Chinese is not the only contaminant.
        assert extract_english_prompt("中文:x。英文:Rinseام glass under the tap.") is None


# ---------------------------------------------------------------------------
# Synthetic Ego4D bucket on disk (no mp4; decode monkeypatched)
# ---------------------------------------------------------------------------


def _make_ego4d_bucket(bucket: Path, combined_prompts: list[str]) -> Path:
    """Build a minimal Ego4D bucket. One task/episode; ``combined_prompts[e]`` is
    the bilingual (or "null") string for episode e."""
    n = len(combined_prompts)
    meta = bucket / "meta"
    (meta / "episodes").mkdir(parents=True, exist_ok=True)
    (meta).mkdir(exist_ok=True)
    info = {
        "fps": FPS,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
    }
    (meta / "info.json").write_text(json.dumps(info))

    # episodes parquet (with the per-episode `tasks` list<string> column)
    rows = []
    cum = 0
    for ep in range(n):
        rows.append(
            {
                "episode_index": ep,
                "length": EP_LEN,
                "tasks": [combined_prompts[ep]],
                "dataset_from_index": cum,
                "data/chunk_index": 0,
                "data/file_index": 0,
                f"videos/{CAM}/chunk_index": 0,
                f"videos/{CAM}/file_index": 0,
            }
        )
        cum += EP_LEN
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), meta / "episodes" / "chunk-000.parquet")

    # tasks.parquet: index = combined string, task_index col = ep
    tdf = pd.DataFrame({"task_index": list(range(n))}, index=pd.Index(combined_prompts, name="task"))
    tdf.to_parquet(meta / "tasks.parquet")

    # data shard: task_index constant within each episode
    data_dir = bucket / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    ti = np.concatenate([np.full(EP_LEN, ep, dtype=np.int64) for ep in range(n)])
    pq.write_table(pa.Table.from_pandas(pd.DataFrame({"task_index": ti})), data_dir / "file-000.parquet")
    return bucket


@pytest.fixture
def patch_decode(monkeypatch):
    """Return dummy frames so no mp4 is needed."""

    def fake_decode(path, frame_indices, height, width):
        return [Image.new("RGB", (width, height)) for _ in frame_indices]

    monkeypatch.setattr("openwam.dataloader.bases.lerobot_v3_reader._decode_video_frames", fake_decode)


def _reader(bucket: Path, **kw) -> Ego4DDataset:
    return Ego4DDataset(
        dataset_dir=str(bucket),
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


class TestEgo4DReader:
    def test_null_episodes_dropped(self, tmp_path):
        prompts = ["中文:a。英文:Do A.", "null", "中文:c。英文:Do C.", "null"]
        bucket = _make_ego4d_bucket(tmp_path / "g", prompts)
        ds = _reader(bucket)
        # 2 null episodes dropped → 2 kept.
        assert len(ds._eps_df) == 2
        kept = set(ds._eps_df["episode_index"].tolist())
        assert kept == {0, 2}

    def test_prompt_is_english_only(self, tmp_path, patch_decode):
        prompts = ["中文:折叠。英文:Fold the shirt.", "中文:测量。英文：Measure it."]
        bucket = _make_ego4d_bucket(tmp_path / "g", prompts)
        ds = _reader(bucket)
        seen = {ds[i]["prompt"] for i in range(0, len(ds), max(1, len(ds) // 4))}
        for p in seen:
            assert "英文" not in p and "中文" not in p
            assert all(ord(c) < 128 for c in p)
        assert "Fold the shirt." in seen or "Measure it." in seen

    def test_sample_shape_video_only(self, tmp_path, patch_decode):
        prompts = ["中文:a。英文:Do A."]
        bucket = _make_ego4d_bucket(tmp_path / "g", prompts)
        ds = _reader(bucket)
        s = ds[0]
        assert len(s["video"]) == 9 and s["video"][0].size == (320, 384)
        assert tuple(s["action"].shape) == (32, 80) and s["action"].dtype == torch.float32
        assert not s["action_mask"].any() and not s["proprio_mask"].any()
        assert tuple(s["proprio"].shape) == (1, 80)
        assert len(s["video_mask"]) == 9
        assert s["vace_video"] is None and len(s["first_frame_image"]) == 1
        assert ds.action_dim == 80

    def test_action_dim_without_unify_is_20(self, tmp_path):
        bucket = _make_ego4d_bucket(tmp_path / "g", ["中文:a。英文:Do A."])
        ds = Ego4DDataset(dataset_dir=str(bucket), multiview=True,
                          camera_layout=[CAM, "__missing_left__", "__missing_right__"], target_camera=CAM,
                          unify_action=False)
        assert ds.action_dim == 20
