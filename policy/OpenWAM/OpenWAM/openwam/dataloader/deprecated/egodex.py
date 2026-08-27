"""EgoDex dataloader for lerobot v3 standard datasets.

Reads /oss/worldmodel/lerobot3/EgoDex (produced by data_conform/egodex_conform)
through the shared :class:`~openwam.dataloader.bases.lerobot_v3_reader.LeRobotV3Reader`
machinery, emitting the canonical sample-dict shape (video / vace_video /
first_frame_image / action / action_mask / video_mask / proprio / proprio_mask
/ prompt) so it batch-collates cleanly with the other robotic sources in
``configs/dataloader/mixture.yaml``.

Design contract (see data_conform/EGODEX_DATALOADER_DESIGN.md):

  - Action loss is **disabled**. The reader emits
        action       = zeros(T-1, 20)
        action_mask  = zeros(T-1, 20, dtype=bool)
    (``_action_20d`` returns None → the base finalizer emits zeros + all-False
    mask). The training pipeline computes ``action_is_pad = ~action_mask`` and
    skips the action term for every EgoDex sample. Only video / image-prediction
    loss is supervised.

  - Proprio is **masked out at the model level**. EgoDex's raw
    ``observation.state`` is a 24-D camera-frame double-wrist pose, which
    cannot be aligned with the robot base-frame 20-D EEF schema used by the
    mixture's other readers. So every EgoDex sample emits
        proprio      = zeros(1, 20)
        proprio_mask = zeros(1, 20, dtype=bool)
    (``_proprio_20d`` returns None). ``observation.state`` is dropped from
    ``NEEDED_COLS`` (no IO). The 24->20 EEF conversion path is intentionally
    absent — rollback goes through ``git revert``.

  - Schema knobs are hardcoded, not read from info.json: action_dim = 20,
    state_dim = 20. info.json declares 24-D (the conversion-time schema); the
    mixture consumes 20-D EEF, hence the dim is fixed in the reader.

  - splits resolution priority (see ``apply_info_splits``):
        info.json["splits"][split]  → conversion-time partition
        else                        → split=train returns full eps_df;
                                      split=val returns empty.

  - Tail-window handling (``_train_min_window_len``): train uses
    ``max_start = length - (video_stride + 1)`` so every emitted window has at
    least 2 real video frames (t0 reference + first supervised prediction step).
    val uses full ``num_frames`` windows to keep val loss comparable. This is a
    tighter train floor than robotwin's 2-frame minimum — intentional, because
    EgoDex is video-only-supervised and a single-REAL-frame window is too weak
    a signal. Padded tail positions are marked ``video_mask=False``.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt resolution (pure function, mirrors robotwin._resolve_prompt)
# ---------------------------------------------------------------------------


def _resolve_egodex_prompt(
    task_idx_to_text: Dict[int, str],
    task_index: int,
) -> str:
    """Resolve EgoDex training-time prompt from ``tasks.parquet`` lookup.

    Mirrors :func:`openwam.dataloader.robotwin._resolve_prompt` in
    shape (pure function, independently testable) but adapted to EgoDex's
    1:1 ``task_index → prompt`` schema:

      * no paraphrase pool (each ``task_index`` maps to exactly 1 string)
      * **no fallback default** — a missing entry or empty string is treated
        as a data bug and raises (rather than silently fabricating a
        template prompt that would slip into training unnoticed)
      * **no wrapper prefix** — the raw stripped string is returned as-is.
    """
    if int(task_index) not in task_idx_to_text:
        raise KeyError(
            f"EgoDex prompt lookup failed: task_index={task_index} not present "
            "in this bucket's tasks.parquet. Indicates a data inconsistency "
            "between the episode's parquet row and the bucket's tasks.parquet."
        )
    base_prompt = task_idx_to_text[int(task_index)].strip()
    if not base_prompt:
        raise ValueError(
            f"EgoDex task_index={task_index} maps to an empty prompt string. "
            "Indicates a tasks.parquet entry with blank text — data must be "
            "fixed upstream."
        )
    return base_prompt


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class EgoDexDataset(LeRobotV3Reader):
    """lerobot v3 reader for EgoDex with action loss disabled.

    Video-only supervised: ``_action_20d`` / ``_proprio_20d`` inherit the base
    ``None`` default, so the base finalizer emits zero action/proprio with
    all-False masks. See module docstring + ``EGODEX_DATALOADER_DESIGN.md``.
    """

    DATASET_NAME = "EgoDex"
    # Reader-emit dims; matches the EEF mixture schema (action_dim=20).
    _ACTION_DIM = 20
    _STATE_DIM = 20
    ACTION_DIM = 20
    # Only task_index is read (for prompt resolution via tasks.parquet); action
    # / observation.state / *_world / finger_poses / etc. are all ignored.
    NEEDED_COLS = ("task_index",)
    # EgoDex's per-bucket tasks.parquet is always present (conversion writes it).
    PROMPT_FILE_REQUIRED = True

    def _resolve_cameras(self, info: dict):
        # Single ego camera; no wrist cameras. Default matches the historical
        # ``target_camera`` ctor default so direct construction still works.
        return (self._target_camera or "observation.images.ego", None, None)

    def _train_min_window_len(self) -> int:
        # video_stride+1 is the smallest actual_raw_len under which
        # _video_sample_indices[1] = video_stride is still < actual_raw_len,
        # i.e. video_mask has >= 2 True slots (t0 reference + first prediction).
        return self._video_stride + 1

    def _resolve_prompt(self, row, win) -> str:
        return _resolve_egodex_prompt(self._task_idx_to_text, int(win["task_index"].iloc[0]))

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiBucketEgoDexDataset


# ---------------------------------------------------------------------------
# Multi-bucket wrapper (root mode)
# ---------------------------------------------------------------------------


class MultiBucketEgoDexDataset(MultiLeRobotV3Reader):
    """Aggregate of N EgoDex per-task buckets.

    Used in root mode: ``EgoDexDataset.from_config({dataset_dir: <root>})``
    where ``<root>`` contains N subdirs each with ``meta/info.json``. All
    EgoDex sub-buckets share ``action_dim=20`` and ``normalization_stats=None``
    by design contract; the base class's defaults match directly.
    """

    def __init__(self, buckets: List["EgoDexDataset"]):
        super().__init__(buckets)
        logger.info(
            "MultiBucketEgoDexDataset: %d buckets, %d total windows (action_dim=%d)",
            len(self._buckets),
            len(self),
            self.action_dim,
        )

    @property
    def buckets(self) -> List["EgoDexDataset"]:
        return self._buckets


__all__ = ["EgoDexDataset", "MultiBucketEgoDexDataset"]
