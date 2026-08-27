"""Haiyu dataloader for lerobot v3 egocentric human-hand data.

Reads ``/mnt/data/wangyuran/Haiyu`` — 87,801 source-video directories
(``<md5>_<start>_<end>/``), each a self-contained multi-episode LeRobot v3
dataset (``robot_type: "ego_hand"``). A directory holds one concatenated
egocentric mp4 and one concatenated per-frame parquet; the ~1–57 episodes in it
are time segments located by a per-episode video-frame offset (like the standard
LeRobot v3 packing). Emits the canonical sample-dict shape (video / vace_video /
first_frame_image / action / action_mask / video_mask / proprio / proprio_mask
/ prompt) so it batch-collates cleanly with the other sources in
``configs/dataloader/mixture.yaml``.

Why a bespoke *flat* reader instead of the EgoDex-style multibucket path
--------------------------------------------------------------------------
EgoDex / RoboCOIN build one full ``LeRobotV3Reader`` per bucket — fine at
~100–700 buckets. Haiyu has **87,801** directories; instantiating one heavyweight
reader per directory (info.json + episodes-parquet read + offset cumsums + LRU
install, ×87k, every process, every run) is far too slow / memory-heavy.

Instead this reader scans the root **once** into a small cached manifest with one
row per *episode* — ``(dir, episode_index, length, prompt, video_chunk,
video_file, video_frame_offset)`` — and builds a single flat window index over
all episodes. Subsequent runs load the manifest instantly. There is no
per-episode reader object and no per-getitem data-parquet read: a sample only
needs its directory, its length, its prompt, and its video-frame offset — all in
the manifest. The video-frame offset is computed with the same
``compute_file_local_offsets`` groupby-cumsum the base reader uses, so seeking is
identical to the standard path.

Design contract (video-only supervised, mirrors EgoDex byte-for-byte)
---------------------------------------------------------------------
  - Action loss is **disabled**: every sample emits ``action = zeros(T-1, dim)``
    all-False mask and ``proprio = zeros(1, dim)`` all-False. Haiyu's per-frame
    payload is bimanual *human* MANO hand pose (not a robot action), so only
    video / image-prediction loss is supervised. With ``unify_action: true`` the
    width ``dim`` is the shared 80-D ``UNIFY_DIM`` (all zeros, all masked) so it
    collates with the other 80-D-head sources. This is exactly what EgoDex's
    ``_finalize_action(None)`` / ``_finalize_proprio(None)`` produce.

  - **Prompt**: the fine-grained per-*episode* English instruction from that
    episode's ``meta/episodes/*.parquet`` ``tasks`` cell (a ``list<string>`` of
    length 1), NOT the coarse ``meta/tasks.parquet`` category label (the two do
    not match; the fine one describes the actual clip segment).

  - **Multiview canvas**: identical to EgoDex — a 3-slot L-shape
    (``multiview: true``, ``camera_layout: [ego, __missing_left__,
    __missing_right__]``) with the ego view on top (256×320) and the two wrist
    slots black, producing the shared 384×320 canvas.

  - **Mixed source resolution** (≈91% 1920×1080, ≈9% 1920×1680) is a non-issue:
    every frame is force-resized into its canvas slot, same as EgoDex.

  - **Bad-episode exclusion**: an optional ``_openwam_haiyu_excluded.json`` at
    the dataset root maps ``{"excluded": {dir: [episode_index, ...], ...}}`` of
    episodes to skip (the flat-dataset analogue of
    ``meta/excluded_episodes.json``, same wrapper shape). The full-load scanner
    (``scripts/scan_dataset.py``) writes/unions it after finding truncated /
    undecodable clips.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from openwam.dataloader.bases.dataset import BaseDataset
from openwam.dataloader.deprecated.ego4d import _episode_tasks_string
from openwam.dataloader.transforms.multiview import assemble_multiview_layout
from openwam.dataloader.transforms.video import VideoColorJitter
from openwam.dataloader.utils import get_cfg as _get_cfg
from openwam.dataloader.utils.eef import EEF_DIM
from openwam.dataloader.utils.lerobotv3 import compute_file_local_offsets, subsample_episodes_by_hours
from openwam.dataloader.utils.unify_action import UNIFY_DIM
from openwam.dataloader.utils.video_io import decode_video_frames

logger = logging.getLogger(__name__)

# Multiview L-shape top (head) slot size (must match assemble_multiview_layout
# / LeRobotV3Reader defaults so canvases are byte-identical to EgoDex).
_HEAD_SLOT_H, _HEAD_SLOT_W = 256, 320

_MANIFEST_VERSION = "v1"
_MANIFEST_COLUMNS = ("dir", "episode_index", "length", "prompt", "video_chunk", "video_file", "video_frame_offset")
# Parquet schema-metadata key recording which camera the cached chunk/file/offset
# columns were built for (see _sanitize_cam / manifest loading).
_MANIFEST_CAMERA_META_KEY = b"openwam_head_camera"
_EXCLUDED_FILENAME = "_openwam_haiyu_excluded.json"
_GETITEM_MAX_RETRIES = 64


def _sanitize_cam(cam: str) -> str:
    """Filesystem-safe token for a camera key, for the manifest filename.

    ``observation.images.ego`` → ``observation_images_ego``.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", str(cam)).strip("_") or "cam"


def _haiyu_prompt(tasks_cell: Any) -> Optional[str]:
    """Extract the fine-grained instruction string from an episodes ``tasks`` cell.

    ``tasks`` is a ``list<string>`` of length 1. Returns the stripped string, or
    None when missing / empty (the caller drops such episodes). Cell-shape
    handling (list/ndarray/scalar, ``[None]`` → None not the literal "None") is
    the shared :func:`~openwam.dataloader.deprecated.ego4d._episode_tasks_string`, so a
    future cell-shape quirk is fixed in one place.
    """
    s = _episode_tasks_string(tasks_cell)
    if s is None:
        return None
    s = s.strip()
    return s or None


def _scan_one_dir(dir_path: Path, cam: str) -> List[tuple]:
    """Return one ``(dir, episode_index, length, prompt, vchunk, vfile, voffset)``
    tuple per usable episode in ``dir_path`` (a multi-episode LeRobot v3 dir).

    Empty on a directory that is not a valid dataset. Episodes with a
    non-positive length or no usable prompt are skipped.
    """
    epm = sorted((dir_path / "meta" / "episodes").rglob("*.parquet"))
    if not epm:
        return []
    vchunk_col = f"videos/{cam}/chunk_index"
    vfile_col = f"videos/{cam}/file_index"
    # Wrap the WHOLE body: a single malformed dir among 87k (unreadable parquet,
    # a missing column, a compute error) must be skipped-and-logged, not abort
    # the entire manifest build.
    try:
        eps = pa.concat_tables([pq.read_table(p) for p in epm]).to_pandas()
        if len(eps) == 0 or vchunk_col not in eps.columns:
            return []
        eps = eps.sort_values("episode_index").reset_index(drop=True)
        # Per-episode video-frame offset within its (chunk, file) video shard —
        # same groupby-cumsum the base reader uses. Handles the multi-episode
        # concatenated mp4 (episode k starts after the summed lengths of earlier
        # episodes).
        voff = compute_file_local_offsets(eps, vchunk_col, vfile_col)
        out: List[tuple] = []
        for i in range(len(eps)):
            length = int(eps["length"].iloc[i])
            prompt = _haiyu_prompt(eps["tasks"].iloc[i])
            if length <= 0 or prompt is None:
                continue
            out.append(
                (
                    dir_path.name,
                    int(eps["episode_index"].iloc[i]),
                    length,
                    prompt,
                    int(eps[vchunk_col].iloc[i]),
                    int(eps[vfile_col].iloc[i]),
                    int(voff[i]),
                )
            )
        return out
    except Exception as e:  # noqa: BLE001 — one bad dir must not kill the scan
        logger.warning("Haiyu manifest: skipping %s (%s: %s)", dir_path.name, type(e).__name__, e)
        return []


class HaiyuDataset(BaseDataset):
    """Flat single-index reader over all Haiyu episodes (87k concatenated dirs).

    Emits the same canonical, video-only-supervised sample dict as
    :class:`~openwam.dataloader.deprecated.egodex.EgoDexDataset`. See module docstring for
    why this is a bespoke flat reader rather than an 87k-bucket multibucket.
    """

    DATASET_NAME = "Haiyu"

    # Layout fact consumed by the external scanners (scripts/gpu_decode_scan.py):
    # Haiyu concatenates several episodes into one mp4, so a decoded frame count is
    # NOT comparable to any single episode's length — the truncation cross-check
    # must be skipped (exit-code check only). Declared here rather than hardcoded
    # via isinstance in the scanner.
    SINGLE_EPISODE_VIDEO_FILES = False

    CONFIG_KEYS: Tuple[str, ...] = (
        "num_frames",
        "video_stride",
        "window_stride",
        "height",
        "width",
        "multiview",
        "target_camera",
        "camera_layout",
        "unify_action",
        "color_jitter",
        "manifest_path",
        "rebuild_manifest",
    )

    def __init__(
        self,
        dataset_dir: str,
        *,
        num_frames: int = 33,
        video_stride: int = 4,
        window_stride: int = 1,
        height: int = 384,
        width: int = 320,
        split: str = "train",
        multiview: bool = True,
        target_camera: str = "observation.images.ego",
        camera_layout: Optional[List[str]] = None,
        unify_action: bool = False,
        color_jitter: Optional[Any] = None,
        max_hours: Optional[float] = None,
        subsample_seed: int = 42,
        manifest_path: Optional[str] = None,
        rebuild_manifest: bool = False,
        dataset_id: Optional[str] = None,
        **_unused: Any,
    ):
        self._dataset_dir = Path(dataset_dir)
        self._dataset_id = dataset_id or self._dataset_dir.name
        self._num_frames = int(num_frames)
        self._video_stride = max(1, int(video_stride))
        self._window_stride = max(1, int(window_stride))
        self._height = int(height)
        self._width = int(width)
        self._split = split
        self._multiview = bool(multiview)
        self._head_camera = target_camera or "observation.images.ego"
        self._camera_layout = (
            list(camera_layout)
            if camera_layout
            else [
                self._head_camera,
                "__missing_left__",
                "__missing_right__",
            ]
        )
        self._max_hours = max_hours
        self._subsample_seed = int(subsample_seed)
        self._rebuild_manifest = bool(rebuild_manifest)
        self._fail_count = 0  # rate-limited retry logging (see _safe_get)
        # The manifest bakes in the target camera's chunk/file/frame-offset
        # columns, so it is camera-specific: the default filename carries the
        # (sanitized) camera and the parquet stamps it in metadata (validated on
        # load). Otherwise changing ``target_camera`` and reusing the old cache
        # would decode the wrong frames (or thrash retries) with no warning.
        self._manifest_path = (
            Path(manifest_path)
            if manifest_path
            else self._dataset_dir
            / f"_openwam_haiyu_manifest_{_MANIFEST_VERSION}_{_sanitize_cam(self._head_camera)}.parquet"
        )

        # ── unified action space (zeros payload; only ACTION_DIM matters) ──
        # Haiyu emits all-zero action/proprio with all-False masks, so the full
        # unify scatter machinery is unnecessary — only the emitted width needs
        # to become UNIFY_DIM under unify so it collates with the 80-D-head
        # mixture sources. Byte-identical to EgoDex's finalized zeros.
        self._unify = bool(unify_action)
        self.ACTION_DIM = int(UNIFY_DIM) if self._unify else int(EEF_DIM)

        # ── load-time video augmentation (train split only, like EgoDex) ──
        self._color_jitter = None
        if color_jitter and split == "train":
            cj_get = color_jitter.get if hasattr(color_jitter, "get") else (lambda k, d: d)
            self._color_jitter = VideoColorJitter(
                brightness=float(cj_get("brightness", 0.2)),
                contrast=float(cj_get("contrast", 0.2)),
                saturation=float(cj_get("saturation", 0.2)),
                hue=float(cj_get("hue", 0.0)),
            )

        # ── video sub-sampling (identical formula to LeRobotV3Reader) ──
        self._video_sample_indices = np.arange(0, self._num_frames, self._video_stride, dtype=np.int64)
        self._num_video_frames = int(self._video_sample_indices.size)

        # ── manifest: one row per episode ──
        # Pretraining convention: split=="train" uses the full manifest; any other
        # split (val) is empty (Haiyu carries no explicit val partition). Build the
        # (expensive, cold) manifest ONLY for train — a val instantiation on a cold
        # cache would otherwise scan the whole tree just to discard the result.
        if split != "train":
            logger.info("Haiyu(%s): split=%s → empty (no val partition)", self._dataset_id, split)
            manifest = pd.DataFrame(columns=list(_MANIFEST_COLUMNS))
        else:
            manifest = self._load_or_build_manifest()

        # ── optional bad-episode exclusion (scanner output) ──
        # Stable universe for scan_dataset's over-exclusion guardrail. This
        # must describe the reader population before the mutable exclusion
        # artifact is applied, not the number left by an arbitrary snapshot.
        self._n_episodes_before_exclusions = int(len(manifest))
        manifest = self._apply_exclusions(manifest)

        # ── optional hour-budget subsample (fps ~29.97, homogeneous) ──
        if self._max_hours is not None and len(manifest) > 0:
            manifest = subsample_episodes_by_hours(
                manifest, target_hours=float(self._max_hours), fps=29.97, seed=self._subsample_seed
            )
            logger.info(
                "Haiyu(%s): subsampled to %d episodes (max_hours=%.3f, seed=%d)",
                self._dataset_id,
                len(manifest),
                self._max_hours,
                self._subsample_seed,
            )

        manifest = manifest.reset_index(drop=True)
        self._dirs = manifest["dir"].to_numpy()
        self._episode_index = manifest["episode_index"].to_numpy().astype(np.int64)
        self._lengths = manifest["length"].to_numpy().astype(np.int64)
        self._prompts = manifest["prompt"].tolist()
        self._vchunk = manifest["video_chunk"].to_numpy().astype(np.int64)
        self._vfile = manifest["video_file"].to_numpy().astype(np.int64)
        self._voffset = manifest["video_frame_offset"].to_numpy().astype(np.int64)

        # ── flat window index (identical formula to LeRobotV3Reader) ──
        min_window_len = self._num_frames if split == "val" else self._video_stride + 1
        n_starts = np.where(
            self._lengths >= min_window_len,
            (self._lengths - min_window_len) // self._window_stride + 1,
            0,
        ).astype(np.int64)
        self._cum_n_starts = np.concatenate([[0], np.cumsum(n_starts)]).astype(np.int64)
        self._n_total = int(self._cum_n_starts[-1])
        # First global window index of every NON-EMPTY episode. Episodes shorter
        # than min_window_len contribute 0 windows (the np.where 0 branch above),
        # so their cum-start coincides with the next episode's. _safe_get jumps
        # between these anchors on failure so it always lands on a real window of
        # a DIFFERENT episode — never a zero-window episode, never _n_total (out
        # of range). nonempty_starts[0] is always 0 when any episode is kept.
        self._nonempty_starts = self._cum_n_starts[:-1][n_starts > 0].astype(np.int64)

        logger.info(
            "Haiyu(%s, %s): %d episodes, %d windows, multiview=%s, action_dim=%d",
            self._dataset_id,
            split,
            len(self._dirs),
            self._n_total,
            self._multiview,
            self.ACTION_DIM,
        )

    # ----- manifest ---------------------------------------------------------

    def _load_or_build_manifest(self) -> pd.DataFrame:
        """Load the cached manifest, or scan the root once and cache it.

        The cache is only reused when it matches BOTH the expected columns and
        the current ``target_camera`` (stamped in parquet metadata) — a manifest
        built for a different camera holds that camera's chunk/file/offset and
        would silently decode wrong frames, so a camera mismatch triggers a
        rebuild rather than a reuse.
        """
        if self._manifest_path.exists() and not self._rebuild_manifest:
            tbl = pq.read_table(self._manifest_path)
            cols = list(tbl.column_names)
            if cols != list(_MANIFEST_COLUMNS):
                raise ValueError(
                    f"Haiyu({self._dataset_id}): cached manifest {self._manifest_path} has columns "
                    f"{cols}, expected {list(_MANIFEST_COLUMNS)}. Delete it or set rebuild_manifest=true."
                )
            stamped = (tbl.schema.metadata or {}).get(_MANIFEST_CAMERA_META_KEY)
            stamped_cam = stamped.decode("utf-8") if stamped is not None else None
            # Reuse only when the stamped camera matches. A mismatch OR a missing
            # stamp (a pre-fix / hand-built manifest whose camera is unknowable)
            # forces a rebuild — trusting an unverifiable camera would decode the
            # wrong frames when an explicit manifest_path is reused across cameras.
            if stamped_cam != self._head_camera:
                logger.warning(
                    "Haiyu(%s): cached manifest %s camera stamp (%r) != target_camera (%r) — "
                    "rebuilding (an unverified camera would decode wrong frames).",
                    self._dataset_id,
                    self._manifest_path,
                    stamped_cam,
                    self._head_camera,
                )
            else:
                m = tbl.to_pandas()
                logger.info(
                    "Haiyu(%s): loaded cached manifest (%d episodes) from %s",
                    self._dataset_id,
                    len(m),
                    self._manifest_path,
                )
                return m

        logger.info(
            "Haiyu(%s): scanning %s for episodes (first-run manifest build)…", self._dataset_id, self._dataset_dir
        )
        sub_dirs = [d for d in sorted(self._dataset_dir.iterdir()) if d.is_dir() and not d.name.startswith((".", "_"))]
        rows: List[tuple] = []
        n_empty_dirs = 0
        scan = partial(_scan_one_dir, cam=self._head_camera)
        with ThreadPoolExecutor(max_workers=16) as pool:
            for res in pool.map(scan, sub_dirs):
                if res:
                    rows.extend(res)
                else:
                    n_empty_dirs += 1
        if not rows:
            raise RuntimeError(f"Haiyu({self._dataset_id}): no valid episodes found under {self._dataset_dir}")
        m = pd.DataFrame(rows, columns=list(_MANIFEST_COLUMNS))
        # Stable, reproducible order across runs/machines.
        m = m.sort_values(["dir", "episode_index"]).reset_index(drop=True)
        logger.info(
            "Haiyu(%s): scanned %d dirs → %d episodes (%d dirs skipped as non-dataset/empty)",
            self._dataset_id,
            len(sub_dirs),
            len(m),
            n_empty_dirs,
        )
        self._write_manifest_atomic(m)
        return m

    def _write_manifest_atomic(self, m: pd.DataFrame) -> None:
        try:
            # Per-process temp name: under multi-rank DDP first-run every rank
            # scans and writes concurrently. A shared temp path would let two
            # processes stream into the same file and promote a torn parquet;
            # a PID-unique temp + atomic replace makes each writer isolated (the
            # replace is atomic, so the final manifest is one rank's complete file).
            tmp = self._manifest_path.with_suffix(self._manifest_path.suffix + f".{os.getpid()}.tmp")
            # Stamp the camera into schema metadata so a later load can detect a
            # target_camera change even when an explicit manifest_path is reused.
            tbl = pa.Table.from_pandas(m, preserve_index=False)
            md = dict(tbl.schema.metadata or {})
            md[_MANIFEST_CAMERA_META_KEY] = self._head_camera.encode("utf-8")
            pq.write_table(tbl.replace_schema_metadata(md), tmp)
            tmp.replace(self._manifest_path)
            logger.info("Haiyu(%s): cached manifest → %s", self._dataset_id, self._manifest_path)
        except Exception as e:
            # A read-only dataset dir is not fatal — we just rescan next time.
            logger.warning(
                "Haiyu(%s): could not cache manifest to %s (%s); will rescan next run.",
                self._dataset_id,
                self._manifest_path,
                e,
            )

    def _apply_exclusions(self, manifest: pd.DataFrame) -> pd.DataFrame:
        """Drop (dir, episode_index) pairs listed in ``_openwam_haiyu_excluded.json``.

        File format: ``{"excluded": {"<dir>": [ep_idx, ...], ...}}``. Absent /
        empty → no-op.
        """
        excl_path = self._dataset_dir / _EXCLUDED_FILENAME
        if not excl_path.exists() or len(manifest) == 0:
            return manifest
        try:
            with open(excl_path) as f:
                excluded = json.load(f).get("excluded", {})
            if not excluded:
                return manifest
            # Parse the shape INSIDE the guard: a valid-JSON but wrong-shape file
            # (top-level list, ``excluded`` a list, or a non-iterable episode value)
            # raises AttributeError/TypeError/ValueError here — which must not crash
            # construction any more than a truncated file does.
            excl_pairs = {(d, int(e)) for d, eps in excluded.items() for e in eps}
        except (json.JSONDecodeError, OSError, AttributeError, TypeError, ValueError) as e:
            # A corrupt / truncated / malformed exclusion file must not crash
            # construction; log and load everything (safer to over-include than to
            # abort training).
            logger.warning("Haiyu(%s): ignoring unreadable/malformed %s (%s)", self._dataset_id, excl_path.name, e)
            return manifest
        if not excl_pairs:
            return manifest
        keys = list(zip(manifest["dir"].tolist(), manifest["episode_index"].tolist()))
        keep = np.array([k not in excl_pairs for k in keys], dtype=bool)
        n_drop = int((~keep).sum())
        if n_drop:
            logger.info(
                "Haiyu(%s): excluded %d/%d episodes via %s", self._dataset_id, n_drop, len(manifest), excl_path.name
            )
        return manifest[keep].reset_index(drop=True)

    # ----- Dataset ----------------------------------------------------------

    def __len__(self) -> int:
        return self._n_total

    def __getitem__(self, idx: int) -> dict:
        return self._safe_get(idx)

    def _safe_get(self, idx: int) -> dict:
        """Retry a failed sample like ``LeRobotV3Reader._safe_get`` so a transient
        decode hiccup (flaky NFS read, momentary PyAV error) does not kill the
        DataLoader worker — and thus training — the way an unguarded call would
        (``MixtureDataset.__getitem__`` has no guard of its own).

        Unlike the base reader's +1 advance, a failure here jumps to the FIRST
        window of a DIFFERENT non-empty episode: with ``window_stride=1`` adjacent
        windows come from the same clip, so +1 would keep hitting the same broken
        video. Jumping via ``_nonempty_starts`` (not raw ``_cum_n_starts``) skips
        zero-window episodes and never produces ``_n_total`` (an out-of-range
        index) — a raw ``_cum_n_starts[ep_local+1]`` jump would do both when the
        failing episode is the last non-empty one followed by short episodes,
        oscillating on the same broken clip instead of advancing. When only one
        (or zero) non-empty episode remains there is nowhere to jump, so retries
        fall back to the same index in place (matching the base reader) — a
        transient error still gets the full retry budget rather than raising on
        the first attempt.
        The scanner (``scripts/scan_dataset.py``) calls ``_getitem_impl`` directly
        to bypass this retry, so deterministic bad episodes are still found and
        excluded; this guard only covers the transient / post-scan cases.
        """
        n_ne = int(self._nonempty_starts.size)
        for attempt in range(_GETITEM_MAX_RETRIES):
            try:
                return self._getitem_impl(idx)
            except Exception as e:  # noqa: BLE001
                if attempt == _GETITEM_MAX_RETRIES - 1:
                    raise
                self._fail_count += 1
                if self._fail_count == 1 or self._fail_count % 100 == 0:
                    logger.warning(
                        "Haiyu(%s): %d cumulative __getitem__ failures (latest idx=%d, %s: %s)",
                        self._dataset_id,
                        self._fail_count,
                        idx,
                        type(e).__name__,
                        e,
                    )
                if n_ne > 1:
                    # Jump to the first window of a DIFFERENT non-empty episode
                    # (window_stride=1 → +1 stays on the same broken clip).
                    pos = int(np.searchsorted(self._nonempty_starts, idx, side="right") - 1)
                    idx = int(self._nonempty_starts[(pos + 1) % n_ne])
                # else: only one (or zero) non-empty episode — nowhere to jump, so
                # retry the SAME idx in place (like the base reader) to ride out a
                # transient hiccup on the full retry budget instead of dying on the
                # first attempt.
        raise RuntimeError("unreachable")

    def _getitem_impl(self, idx: int) -> dict:
        """Assemble one sample. Head-video decode failure raises (the scanner
        relies on this to find truncated clips); there is no wrist camera."""
        if not 0 <= idx < self._n_total:
            raise IndexError(f"Haiyu({self._dataset_id}) idx {idx} out of range [0, {self._n_total})")
        ep_local = int(np.searchsorted(self._cum_n_starts, idx, side="right") - 1)
        offset = (idx - int(self._cum_n_starts[ep_local])) * self._window_stride
        ep_len = int(self._lengths[ep_local])
        actual_raw_len = min(self._num_frames, ep_len - offset)

        # Video frame indices within this episode's (possibly concatenated) clip:
        # v_base = the episode's frame offset in its video file + the window
        # offset within the episode.
        real_local_indices = self._video_sample_indices[self._video_sample_indices < actual_raw_len]
        v_base = int(self._voffset[ep_local]) + offset
        frame_indices = (real_local_indices + v_base).tolist()

        video = self._decode_video(ep_local, frame_indices, idx)
        if self._color_jitter is not None:
            video = self._color_jitter.apply({"video": video})["video"]
        video_mask = torch.from_numpy(self._video_sample_indices < actual_raw_len)

        # Video-only supervision: zeros + all-False masks (== EgoDex output).
        t_action = self._num_frames - 1
        action = torch.zeros((t_action, self.ACTION_DIM), dtype=torch.float32)
        action_mask = torch.zeros((t_action, self.ACTION_DIM), dtype=torch.bool)
        proprio = torch.zeros((1, self.ACTION_DIM), dtype=torch.float32)
        proprio_mask = torch.zeros((1, self.ACTION_DIM), dtype=torch.bool)

        return {
            "video": video,
            "vace_video": None,
            "first_frame_image": [video[0]] if video else [],
            "action": action,
            "action_mask": action_mask,
            "video_mask": video_mask,
            "proprio": proprio,
            "proprio_mask": proprio_mask,
            "prompt": self._prompts[ep_local],
        }

    def head_video_path(self, ep_local: int) -> Path:
        """Absolute path of the ego (head) mp4 backing episode ``ep_local``.

        Single source of truth for the ``{dir}/videos/{cam}/chunk-XXX/file-XXX.mp4``
        layout: both this reader's decode path and the external integrity scanner
        (``scripts/gpu_decode_scan.py``) resolve the *same* file through this
        method, so "the scanner checks the file training decodes" holds by
        construction instead of by hand-synced copies of the path formula.
        """
        return (
            self._dataset_dir
            / str(self._dirs[ep_local])
            / "videos"
            / self._head_camera
            / f"chunk-{int(self._vchunk[ep_local]):03d}"
            / f"file-{int(self._vfile[ep_local]):03d}.mp4"
        )

    def _decode_video(self, ep_local: int, frame_indices: List[int], idx: int) -> List:
        """Decode the ego view into a list of PIL images (single-view) or L-shape
        multiview canvases, with last-real-frame tail padding — matches
        ``LeRobotV3Reader._decode_window_video`` for a head-only source."""
        video_path = self.head_video_path(ep_local)
        head_h, head_w = (_HEAD_SLOT_H, _HEAD_SLOT_W) if self._multiview else (self._height, self._width)
        head_frames = decode_video_frames(str(video_path), frame_indices, head_h, head_w)
        if not head_frames:
            raise RuntimeError(f"empty head-video decode for Haiyu({self._dataset_id}) at idx={idx} ({video_path})")

        n_real = len(head_frames)
        if n_real < self._num_video_frames:
            head_frames = head_frames + [head_frames[-1]] * (self._num_video_frames - n_real)

        if not self._multiview:
            return head_frames

        return [
            assemble_multiview_layout(
                {self._head_camera: head_frames[fi]}, self._camera_layout, out_h=self._height, out_w=self._width
            )
            for fi in range(self._num_video_frames)
        ]

    # ----- BaseDataset metadata --------------------------------------------

    @property
    def action_dim(self) -> int:
        return self.ACTION_DIM

    @property
    def normalization_stats(self) -> Optional[dict]:
        # Samples carry no action supervision; nothing to normalize.
        return None

    # ----- from_config ------------------------------------------------------

    @classmethod
    def from_config(cls, config, split: str = "train") -> "HaiyuDataset":
        dataset_dir = _get_cfg(config, "dataset_dir")
        if dataset_dir is None:
            raise ValueError(f"{cls.__name__}: missing dataset_dir")

        kwargs: dict = {"split": split}
        for key in cls.CONFIG_KEYS:
            v = _get_cfg(config, key)
            if v is None:
                continue
            kwargs[key] = v

        total_hours = _get_cfg(config, "total_hours")
        if total_hours is not None:
            kwargs["max_hours"] = float(total_hours)
            kwargs["subsample_seed"] = int(_get_cfg(config, "seed", 42))

        dataset_id = _get_cfg(config, "dataset_id")
        if dataset_id is not None:
            kwargs["dataset_id"] = dataset_id

        return cls(dataset_dir=str(dataset_dir), **kwargs)


__all__ = ["HaiyuDataset"]
