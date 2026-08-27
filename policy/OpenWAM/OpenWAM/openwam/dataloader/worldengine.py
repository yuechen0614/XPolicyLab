"""WorldEngine dataloader for the lerobot v3 egocentric human-hand dataset.

Reads ``/mnt/data/wangyuran/pretrain_dataset/WorldEngine`` — a **sharded** root
holding 50 standalone LeRobot v3 datasets (``shard_000 … shard_049``, each a
complete ``meta/`` + ``data/`` + ``videos/`` tree with its own
``meta/info.json``, ``robot_type: "ego_hand"``). Aggregate size at this drop:
71,567 episodes, ~744.9 M frames @ 30 fps (~6,897 h of raw ego footage) and
3,006 tasks. Because ``meta/info.json`` does **not** exist at the root,
``from_config`` takes the **root (multi-bucket) branch**: one
:class:`WorldEngineDataset` per ``shard_XXX``, aggregated by
:class:`MultiBucketWorldEngineDataset` via the shared multi-bucket reader.
Each bucket emits the canonical sample-dict (video / vace_video /
first_frame_image / action / action_mask / video_mask / proprio / proprio_mask
/ prompt) so it batch-collates cleanly with the other sources in
``configs/dataloader/mixture.yaml``.

.. note::
   This reader targets the **pretrain_dataset drop only**. The earlier
   single-root ``/mnt/data/wangyuran/WorldEngine`` (1,198,957 atomic 0.5–4 s
   clips, bilingual ``中文:…英文:…`` prompts in a ``task`` column) is a different
   dataset and is no longer supported here — its layout assumptions have been
   removed rather than kept as a fallback.

Layout facts this reader depends on (verified against the drop)
---------------------------------------------------------------
  - **Sharded root, per-shard namespaces.** Each ``shard_XXX`` restarts
    ``episode_index`` at 0 and ``task_index`` at 0, so prompt lookup MUST stay
    inside a shard. That is automatic in root mode: every bucket reads only its
    own ``meta/tasks.parquet``.

  - **Long episodes.** An episode is a whole ~6-minute recording (median 11,150
    frames ≈ 372 s), not an atomic action clip — ~1,430 episodes per shard. At
    ``window_stride: 1`` the source contributes ~744.6 M training windows
    uncapped. The shipped yaml applies an effective-hour budget; see
    ``configs/dataloader/mixture.yaml`` for the active balance and per-process
    index-map memory requirement.

  - **Plain-English prompts, stored in the parquet index column.**
    ``meta/tasks.parquet`` has an ``int64 task_index`` column plus the
    instruction text. The file is written by pyarrow **without pandas
    metadata**, so the text arrives as an ordinary column literally named
    ``__index_level_0__`` (a writer that emits pandas metadata would restore it
    as the DataFrame index instead — :meth:`_load_prompts` handles both, plus a
    plain ``task`` column). The text is already English (0/3006 tasks contain a
    Han glyph, 0 are blank), so there is **no bilingual split step**; only the
    shared non-Latin-script guard runs — see :func:`normalize_prompt`.

  - **One task per episode.** Every ``meta/episodes/*.parquet`` ``tasks`` cell
    is a ``list<string>`` of length 1, and the per-frame ``task_index`` is
    constant across an episode and agrees with that cell. Prompt resolution
    still goes through the per-frame ``task_index`` (the family convention, and
    the base :meth:`_resolve_prompt` path), with the episodes-side text used for
    filtering and for the load-time consistency check.

Design contract (mirrors Ego4D / EgoDex — video-only supervised)
---------------------------------------------------------------
  - **Action loss is disabled.** ``_action_20d`` / ``_proprio_20d`` inherit the
    base ``None`` default, so the finalizer emits ``action = zeros(T-1, dim)``
    with an all-False mask and ``proprio = zeros(1, dim)`` all-False.
    WorldEngine's per-frame payload is bimanual *human* MANO hand pose
    (``left/right_transl_world[3]`` + ``orient_world[9]`` + ``hand_pose[135]``)
    plus a 122-D ``observation.state`` and a 2-D ``state_mask`` (== per-hand
    ``left_kept``/``right_kept``) — human hand pose, NOT a robot action. The
    ``data/*.parquet`` carries **no materialized ``action`` column at all**
    (``meta/info.json`` lists an ``action[102]`` feature, but it is aspirational
    — the frames store the decomposed hand pose instead). So only video /
    image-prediction loss is supervised. With ``unify_action: true`` the emitted
    width becomes the shared 80-D ``UNIFY_DIM`` (all zeros, all masked) so it
    collates with the other 80-D-head sources.

  - **Unusable-prompt episodes are dropped.** An episode whose ``tasks`` cell is
    NULL, empty, a placeholder token like ``"null"``, or carries non-Latin script
    is excluded via :meth:`_filter_episodes` before the window index is built, so
    it never reaches training. At this drop that filter removes nothing — it is a
    guard against a future re-conversion, not a workaround.

  - **A broken data contract aborts the launch, an IO fault does not.** Root mode
    goes through ``build_multibucket``, which deliberately turns a bucket that
    fails to construct into a warning + a dropped bucket. That is right for a
    truncated shard and wrong for proven-broken data — silently dropping one
    shard here costs ~15 M frames. So every guard that has PROVEN a data problem
    (prompt-table divergence, an unusable ``tasks.parquet`` layout, a missing
    ``tasks`` column) raises
    :class:`~openwam.dataloader.utils.lerobotv3.DataContractError`, which
    ``build_multibucket`` re-raises instead of swallowing. Environmental failures
    keep using ordinary exceptions and stay tolerated.

  - **Single ego camera.** Each shard ships TWO egocentric views —
    ``observation.images.ego`` and ``observation.images.ego_right``. For canvas
    consistency with the rest of the ego-hand family (ego4d / haiyu / egodex
    each render a single ego view on the top L-shape slot with two black wrist
    slots) and clean mixture collation, this reader uses **only** the primary
    ``ego`` view. ``ego_right`` is intentionally not rendered; wiring it into a
    second slot is a known follow-up.

  - **Video seeking is the stock base path.** Videos are LeRobot-v3 concatenated
    per-view mp4 (2–7 episodes per file). The base reader seeks each episode via
    its per-episode ``videos/observation.images.ego/{chunk,file}_index`` plus a
    groupby-cumsum frame offset; that offset was verified to equal
    ``round(from_timestamp * fps)`` for every episode in the drop, and the
    analogous per-file data-row offset was verified to land on the episode's own
    rows (``episode_index`` at the offset matches). So no WorldEngine-specific
    seeking is needed.

  - **Bad-episode exclusion (operational).** Per-shard
    ``meta/excluded_episodes.json`` (written by ``scripts/scan_dataset.py`` /
    ``scripts/scan_all.sh`` phase-1, consumed at
    ``bases/lerobot_v3_reader.py`` __init__) blacklists episodes that live in a
    truncated parquet / mp4. None ship with this drop; run the scanner to
    populate them. Training is robust either way — the base ``_safe_get`` retries
    past a bad clip — but note that a ~11 k-window episode far exceeds the
    64-retry budget, so a systematically bad episode must be excluded, not
    retried around.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, List, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openwam.dataloader.bases import LeRobotV3Reader, MultiLeRobotV3Reader
from openwam.dataloader.utils.lerobotv3 import DataContractError

logger = logging.getLogger(__name__)

# Shared annotation shape with the now-dormant Ego4D reader.  Keep these two
# tiny prompt helpers local so the active WorldEngine path never imports a
# module from ``openwam.dataloader.deprecated``.
_NON_ENGLISH_RE = re.compile(r"[^\x00-\x7f\u00c0-\u024f\u2000-\u206f]")


def _episode_tasks_string(tasks_cell) -> Optional[str]:
    """Extract one prompt from an episodes-parquet ``tasks`` cell."""
    if tasks_cell is None:
        return None
    if isinstance(tasks_cell, (list, tuple, np.ndarray)):
        first = tasks_cell[0] if len(tasks_cell) else None
        return None if first is None else str(first)
    return str(tasks_cell)


# ---------------------------------------------------------------------------
# Prompt normalization (pure function, independently testable)
# ---------------------------------------------------------------------------

# Non-instruction tokens that must never become a prompt. ``"nan"`` earns its
# place the hard way: a NULL cell in a string column comes back from pandas as
# the float ``nan``, and ``str(nan)`` is ``"nan"`` — non-blank, all-ASCII, so
# every other guard here waves it through. Compared case-insensitively against
# the stripped text.
_PLACEHOLDER_TASKS = frozenset({"null", "none", "nan", "n/a", "na", "-", "--"})


def normalize_prompt(text: Optional[str]) -> Optional[str]:
    """Return the usable English instruction for a WorldEngine task, or None.

    This drop's task strings are already plain English — unlike the older
    single-root WorldEngine / Ego4D conversions there is no ``中文:…英文:…``
    wrapper to split, so nothing is stripped. What remains is the safety net:

      * anything that is not a ``str`` → None. This is the load-bearing case,
        not a type nicety: a NULL cell in a parquet string column arrives as the
        float ``nan``, and a ``str()`` coercion would turn it into the literal
        prompt ``"nan"`` — non-blank, all-ASCII, and therefore invisible to every
        other check here and to the two callers downstream. Same for a numeric
        column that reaches us by mistake (``0`` → ``"0"``);
      * blank, or a placeholder token like ``"null"`` (see
        :data:`_PLACEHOLDER_TASKS`) → None;
      * any character outside the Latin allowlist shared with Ego4D
        (``_NON_ENGLISH_RE``: ASCII + Latin-1 Supplement / Latin Extended-A,B +
        General Punctuation) → None, i.e. a CJK / Cyrillic / Arabic glyph that
        leaked in from a future re-conversion marks the task unusable.

    A None result makes :meth:`WorldEngineDataset._filter_episodes` drop the
    episode, guaranteeing no non-English text reaches training. At the current
    drop 0 / 3006 tasks are rejected.
    """
    # numpy.str_ subclasses str, so the real drop's pandas/Arrow values pass.
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s or s.lower() in _PLACEHOLDER_TASKS:
        return None
    if _NON_ENGLISH_RE.search(s):
        return None
    return s


class _ArrowPromptMap:
    """CoW-friendly ``task_index -> English prompt`` lookup.

    Stores the prompts in a single contiguous Arrow ``string`` array (one data +
    one offsets buffer), keyed by a sorted ``int64`` numpy array — instead of a
    ``{int: str}`` dict of distinct Python ``str`` objects. Under the fork-based
    ``DataLoader`` a dict lookup increfs the hit ``str``, dirtying that object's
    memory page into a per-worker private copy; across an epoch every prompt is
    visited → the whole prompt working set turns private per worker. This is the
    same fork-CoW mechanism the ``eps_df`` column prune addresses. A lookup on
    this map instead only *reads* the shared buffers and materializes a fresh
    worker-local ``str`` via ``as_py()`` — no per-prompt refcount is touched, so
    the shared pages stay clean and resident memory does not grow with prompts
    visited. (Per shard this drop holds only ~60 tasks, so the saving is small
    in absolute terms; the map is kept because it also gives the load-time
    validation below and costs nothing.)

    Duck-types the ``{int: str}`` mapping the base
    :meth:`LeRobotV3Reader._resolve_prompt` consumes (``in`` + ``[]`` with
    ``KeyError`` on miss), so that method — and its empty-string fail-loud guard
    for an episode that slipped past ``_filter_episodes`` — is reused unchanged.
    Handles a sparse / non-contiguous ``task_index`` (real WorldEngine shards are
    dense ``0..n-1``, but this does not assume it) via ``searchsorted`` over the
    sorted keys.
    """

    __slots__ = ("_keys", "_texts")

    def __init__(self, task_idx: np.ndarray, texts: list[str]) -> None:
        # Validate BEFORE the int64 cast: a tasks.parquet task_index column with
        # nulls arrives from pandas as float64-with-NaN, and np.int64(NaN) is
        # INT64_MIN — a silent garbage key that would shadow the real one and
        # surface only as a runtime KeyError. Fail loud at load time instead.
        raw = np.asarray(task_idx)
        if raw.dtype.kind == "f":
            if not np.all(np.isfinite(raw)) or not np.all(raw == np.floor(raw)):
                raise ValueError(
                    "task_index column contains null/NaN or non-integer values — "
                    "casting would produce garbage int64 keys (NaN → INT64_MIN). "
                    "The tasks.parquet is malformed; fix it upstream."
                )
            idx = raw.astype(np.int64)
        elif raw.dtype.kind in "iu":
            idx = raw.astype(np.int64)
        else:
            raise ValueError(f"task_index column has non-numeric dtype {raw.dtype!r}; expected integers.")
        order = np.argsort(idx, kind="stable")
        self._keys = idx[order]
        # Reorder the prompts to match the sorted keys, then freeze into one
        # contiguous Arrow buffer (no per-element Python str objects retained).
        self._texts = pa.array([texts[i] for i in order.tolist()], type=pa.string())

    def _pos(self, key: int) -> int:
        i = int(np.searchsorted(self._keys, key))
        if 0 <= i < self._keys.shape[0] and int(self._keys[i]) == key:
            return i
        return -1

    def __contains__(self, key: object) -> bool:
        try:
            return self._pos(int(key)) >= 0  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __getitem__(self, key: int) -> str:
        i = self._pos(int(key))
        if i < 0:
            raise KeyError(key)
        return self._texts[i].as_py()

    def __len__(self) -> int:
        return int(self._keys.shape[0])


class WorldEngineDataset(LeRobotV3Reader):
    """Per-shard lerobot v3 reader for WorldEngine (video-only, English prompts).

    One instance reads exactly one ``shard_XXX`` bucket; the 50 shards of the
    dataset root are aggregated by :class:`MultiBucketWorldEngineDataset` in
    ``from_config`` root mode. Video-only supervised (like Ego4D / EgoDex):
    ``_action_20d`` / ``_proprio_20d`` inherit the base ``None`` default → zero
    action/proprio with all-False masks. See the module docstring for the full
    contract.
    """

    DATASET_NAME = "WorldEngine"
    # Reader-emit width; matches the EEF mixture schema (action_dim=20 pre-unify,
    # 80 under unify_action=true).
    ACTION_DIM = 20
    # Only task_index is read from the per-frame parquet (for the inherited
    # task_index prompt path); the MANO hand-pose / state columns are ignored
    # (video-only supervision).
    NEEDED_COLS = ("task_index",)
    # Prompt convention: task_index → text via meta/tasks.parquet, English-only.
    PROMPT_SOURCE = "task_index"
    PROMPT_FILE_REQUIRED = True

    # Column names meta/tasks.parquet may carry the instruction text under, in
    # precedence order. ``task`` is checked FIRST because it is an explicit,
    # self-describing name; ``__index_level_0__`` is pyarrow's placeholder for
    # "whatever the row index happened to be", which this drop's writer leaves
    # behind (it emits no pandas metadata) but which a different conversion could
    # just as well fill with an integer row counter — so it is the fallback, and
    # whichever column wins is dtype-checked in :meth:`_load_prompts`. A writer
    # that DOES emit pandas metadata makes pandas restore the text as the
    # DataFrame index instead — handled separately there too.
    TASK_TEXT_COLUMNS: ClassVar[tuple[str, ...]] = ("task", "__index_level_0__")

    # Load-time prompt-consistency pass: max data shards probed per bucket.
    # 0 (the default) / negative → probe EVERY referenced shard; a positive value
    # caps the probe at that many evenly spread shards. Overridable per-run via
    # the ``prompt_check_max_shards`` config key. See
    # :meth:`_assert_prompt_consistency` for the coverage-vs-latency trade.
    PROMPT_CHECK_MAX_SHARDS: ClassVar[int] = 0

    # Expose the probe cap to yaml / CLI (`dataloader.prompt_check_max_shards=8`)
    # so the coverage-vs-launch-latency trade is the operator's to make without
    # editing code.
    CONFIG_KEYS: ClassVar[tuple[str, ...]] = LeRobotV3Reader.CONFIG_KEYS + ("prompt_check_max_shards",)

    def __init__(self, *args, prompt_check_max_shards: Optional[int] = None, **kwargs):
        # Set BEFORE super().__init__: it runs _post_init → the probe, which
        # reads this value. An instance attribute shadows the class default.
        if prompt_check_max_shards is not None:
            self.PROMPT_CHECK_MAX_SHARDS = int(prompt_check_max_shards)
        super().__init__(*args, **kwargs)

    def _resolve_cameras(self, info: dict):
        # Single ego camera; no wrist cameras. ego_right is available in the data
        # but intentionally unused (see module docstring).
        return (self._target_camera or "observation.images.ego", None, None)

    def _train_min_window_len(self) -> int:
        # video_stride+1 guarantees >= 2 real video frames per train window
        # (t0 reference + first prediction step). Same floor as Ego4D / EgoDex,
        # appropriate because this source is video-only-supervised.
        return self._video_stride + 1

    def _load_prompts(self) -> None:
        """Build the ``task_index -> english_prompt`` lookup from ``meta/tasks.parquet``.

        This shard's ``tasks.parquet`` holds an ``int64 task_index`` column plus
        the plain-English instruction text. The text column's on-disk name
        depends on the writer (see :attr:`TASK_TEXT_COLUMNS` and the module
        docstring); all observed forms are resolved here, and an unrecognized
        layout raises rather than silently mis-mapping prompts.

        Whichever candidate wins is also **dtype-checked**. Resolving the text
        from a non-string column would not fail — ``normalize_prompt`` rejects
        the values, every prompt maps to ``""``, and the load-time consistency
        probe then reports the whole shard as diverged — but the diagnostic would
        point at the wrong thing. Naming the real problem here keeps the error
        actionable, and is what the ``__index_level_0__`` fallback needs to be
        safe (that column is by definition "whatever the row index was", so an
        integer row counter is a plausible future value).

        Each string goes through :func:`normalize_prompt`; a task with no usable
        text maps to ``""`` so the inherited :meth:`_resolve_prompt` raises
        loudly if such an episode ever slips past :meth:`_filter_episodes` (it
        should not — those episodes are dropped at init).

        The lookup is an Arrow-backed :class:`_ArrowPromptMap` rather than a
        ``{int: str}`` dict (fork-CoW; see that class). It duck-types the dict,
        so :meth:`_resolve_prompt` is inherited unchanged.

        Raises:
            DataContractError: on any unusable tasks.parquet layout. Not a plain
                ``KeyError``: in root mode ``build_multibucket`` swallows ordinary
                exceptions into "skip this bucket", which would drop a whole
                shard (~15 M frames) behind one warning.
        """
        tasks_path = self._dataset_dir / "meta" / "tasks.parquet"
        tasks_df = pd.read_parquet(tasks_path)  # FileNotFoundError if missing
        if "task_index" not in tasks_df.columns:
            raise DataContractError(
                f"WorldEngine({self._dataset_id}): meta/tasks.parquet must have a 'task_index' column; "
                f"got {list(tasks_df.columns)}."
            )

        text_col = next((c for c in self.TASK_TEXT_COLUMNS if c in tasks_df.columns), None)
        if text_col is not None:
            series = tasks_df[text_col]
            # infer_dtype(skipna=True), NOT is_string_dtype: on pandas 2.x a
            # string column with even one NULL reads back as object dtype and
            # is_string_dtype value-infers it to False — escalating "1 of 3006
            # tasks is NULL" (normalize_prompt's load-bearing case, handled via
            # n_bad below) into a whole-source DataContractError. skipna keeps
            # NULLs out of the inference; an all-NULL column infers "empty"
            # (pandas 2.x) and maps to all-"" like any other unusable text.
            inferred = pd.api.types.infer_dtype(series, skipna=True)
            if inferred not in ("string", "empty"):
                raise DataContractError(
                    f"WorldEngine({self._dataset_id}): meta/tasks.parquet column {text_col!r} holds "
                    f"{inferred} values, not strings — it is not the instruction text. "
                    f"Available columns: {list(tasks_df.columns)}."
                )
            task_str = series.to_numpy()
        # A writer that emitted pandas metadata makes pandas restore the text as
        # the frame's index (the Ego4D-style layout) instead of a column. Test by
        # "not numeric" rather than "== object": pandas 3 gives a string index the
        # dedicated ``str`` dtype, while a placeholder RangeIndex stays int64.
        elif not pd.api.types.is_numeric_dtype(tasks_df.index):
            task_str = tasks_df.index.to_numpy()
        else:
            raise DataContractError(
                f"WorldEngine({self._dataset_id}): meta/tasks.parquet carries no task-text column "
                f"(looked for {list(self.TASK_TEXT_COLUMNS)}, then a string index); "
                f"got columns {list(tasks_df.columns)} and a {tasks_df.index.dtype} index."
            )

        task_idx = tasks_df["task_index"].to_numpy()
        texts = [normalize_prompt(s) or "" for s in task_str.tolist()]
        n_bad = sum(1 for t in texts if not t)
        if n_bad:
            logger.warning(
                "WorldEngine(%s): %d/%d tasks.parquet entries have no usable English text; "
                "episodes referencing them are dropped by _filter_episodes.",
                self._dataset_id,
                n_bad,
                len(texts),
            )
        self._task_idx_to_text = _ArrowPromptMap(task_idx, texts)

    def _filter_episodes(self, eps_df: pd.DataFrame) -> pd.DataFrame:
        """Drop episodes whose task text is unusable, then prune unread columns.

        Uses the per-episode ``tasks`` column of ``meta/episodes/*.parquet``
        (a ``list<string>`` of length 1) so filtering does not depend on the
        task_index join. Runs before the window index is built, so dropped
        episodes never produce a sample. At the current drop nothing is dropped.
        """
        if "tasks" not in eps_df.columns:
            # Defensive: WorldEngine episodes always carry the tasks column. If a
            # future conversion drops it, fail loud rather than silently keep nulls.
            # DataContractError (not KeyError) so root mode cannot swallow it into
            # a dropped bucket — see :meth:`_load_prompts`.
            raise DataContractError(
                f"WorldEngine({self._dataset_id}): meta/episodes parquet has no 'tasks' column; "
                "cannot filter unusable-prompt episodes."
            )
        keep = eps_df["tasks"].apply(lambda t: normalize_prompt(_episode_tasks_string(t)) is not None)
        n_before = len(eps_df)
        n_drop = int((~keep).to_numpy().sum())
        if n_drop:
            logger.warning(
                "WorldEngine(%s): dropped %d/%d episodes with no usable English prompt",
                self._dataset_id,
                n_drop,
                n_before,
            )
        kept = eps_df[keep.to_numpy()].reset_index(drop=True)

        # ── prune columns the rest of the reader never touches ────────────────
        # The episodes table is dominated by the ``tasks`` list<string> column —
        # one distinct Python list object per episode. ``_getitem_impl`` does
        # ``row = self._eps_df.iloc[ep_local]`` per sample, which INCREFs every
        # object-column cell and dirties its page, so under the fork-based
        # DataLoader each worker privately copies that column. None of these
        # columns are read after this point: the head-camera video offsets and the
        # data-row offset were already materialized by ``_add_episode_offsets``
        # (which runs in ``_build_episode_index`` BEFORE this hook), and the prompt
        # is resolved from the per-frame ``task_index`` (data parquet), not here.
        # ``tasks`` itself is consumed only by the filter just above.
        # Keep only what ``__init__`` (offset extraction / window index) and
        # ``_getitem_impl`` / the scanner read; drop by name so a missing column in
        # a minimal test fixture is a no-op. Alignment is preserved — this only
        # narrows columns, never reorders/drops rows.
        head = self._head_camera
        keep_cols = {
            "length",
            "episode_index",
            "data/chunk_index",
            "data/file_index",
            f"videos/{head}/chunk_index",
            f"videos/{head}/file_index",
            "_data_row_offset",
            self._video_offset_col(head),
        }
        drop_cols = [c for c in kept.columns if c not in keep_cols]
        return kept.drop(columns=drop_cols)

    def _post_init(self, info: dict) -> None:
        super()._post_init(info)
        self._assert_prompt_consistency()

    @staticmethod
    def _sample_probe_shards(all_shards: List[tuple], cap: int) -> List[tuple]:
        """Pick which data shards :meth:`_assert_prompt_consistency` reads.

        ``cap <= 0`` (the default) or ``cap >= len(all_shards)`` → every shard.
        Otherwise a strided walk over the sorted keys, so the sample spreads
        across all chunks instead of clustering at the head of the dataset.

        Split out from the probe so the selection is directly testable: it must
        be a pure function of ``(all_shards, cap)``, with no RNG. A sample that
        varied per launch would make a divergence appear and vanish at random
        across restarts, which is worse than not checking at all.
        """
        n = len(all_shards)
        if cap <= 0 or cap >= n:
            return list(all_shards)
        picks = np.linspace(0, n - 1, num=cap).round().astype(int)
        return [all_shards[i] for i in sorted(set(picks.tolist()))]

    def _assert_prompt_consistency(self) -> None:
        """Load-time guard for the two-source prompt contract.

        Episode FILTERING keeps/drops by the episodes-parquet ``tasks`` cell,
        while the RUNTIME prompt resolves the per-frame ``task_index`` through
        ``meta/tasks.parquet`` — two sources whose agreement is otherwise only
        assumed. If a future data drop diverges (an episode passes the filter but
        its ``task_index`` maps to a missing / unusable-English tasks.parquet
        row), EVERY window of that episode raises in ``__getitem__``; the base
        ``_safe_get`` advances +1 per retry and a WorldEngine episode backs
        ~11,150 windows (≈6 min @ 30 fps) ≫ the 64-retry budget, so the
        DataLoader worker dies mid-training. This pass moves that failure to
        construction time: read the ``task_index`` column of a bounded, evenly
        spread sample of the referenced data shards, require each covered
        episode's ``task_index`` to be CONSTANT across the episode (the runtime
        prompt comes from each *window's* first row —
        ``bases/lerobot_v3_reader.py`` ``_resolve_prompt`` — so a mid-episode
        flip diverges from the first-row value the filter saw), and require that
        constant to map to a non-empty English prompt.

        Coverage is capped by ``PROMPT_CHECK_MAX_SHARDS`` and the achieved
        fraction is logged, never left implicit. **The default (0) is exhaustive**:
        the root's ~33 k column reads are ~13 CPU-minutes serial, but they fan out
        across root mode's 16 concurrent buckets × 8 threads each. Measured wall
        time for a full 50-bucket root build, which is dominated by whether the
        parquet footers are in page cache:

            cold cache   ~55–60 s exhaustive
            warm cache   ~15 s exhaustive   (vs ~1 s at a cap of 8)

        Paying that once per launch buys the whole guarantee, and partial coverage
        buys much less than it looks: a cap of 8 samples a bucket's ~675 data
        shards down to ~17 of its 1,432 episodes (1.2 %), which catches a
        whole-dataset divergence but misses one confined to a single re-converted
        chunk — the likelier failure. Set a positive ``prompt_check_max_shards``
        only when launch latency matters more; note that concurrent ranks
        multiply the IO and are more likely to hit the cold-cache number.

        Environmental problems must not block construction (the reader's
        standing contract): a shard that cannot be read here — a transient NFS
        blip, or corruption that is the integrity scanner's job to blacklist —
        is skipped with a warning. Only a PROVEN prompt divergence raises, and it
        raises :class:`DataContractError` so root mode cannot swallow it into a
        dropped bucket.
        """
        eps = self._eps_df
        if len(eps) == 0:
            return
        chunk = eps["data/chunk_index"].to_numpy().astype(np.int64)
        file_ = eps["data/file_index"].to_numpy().astype(np.int64)
        epi = eps["episode_index"].to_numpy().astype(np.int64)
        length = eps["length"].to_numpy().astype(np.int64)
        row_off = self._ep_data_row_offset

        # Group surviving episodes by their (chunk, file) data shard so each
        # shard's task_index column is read exactly once.
        by_shard: dict[tuple[int, int], list[int]] = {}
        for i in range(len(eps)):
            by_shard.setdefault((int(chunk[i]), int(file_[i])), []).append(i)

        all_shards = sorted(by_shard)
        n_shards = len(all_shards)
        probe_shards = self._sample_probe_shards(all_shards, int(self.PROMPT_CHECK_MAX_SHARDS))
        shard_items = [(k, by_shard[k]) for k in probe_shards]

        def _gather(item):
            (c, f), rows = item
            path = self._dataset_dir / self._data_path_template.format(chunk_index=c, file_index=f)
            try:
                col = pq.read_table(path, columns=["task_index"])["task_index"].combine_chunks().to_numpy()
            except Exception as e:  # noqa: BLE001 — environmental: skip, don't block construction
                return ("skip", (c, f), f"{type(e).__name__}: {e}", len(rows))
            out = []
            for i in rows:
                off = int(row_off[i])
                end = off + int(length[i])
                if not 0 <= off < end <= col.shape[0]:
                    # Episode rows beyond the shard: a truncation-type integrity
                    # problem — the scanner's domain, not a prompt divergence.
                    return ("skip", (c, f), f"episode rows [{off}, {end}) out of range [0, {col.shape[0]})", len(rows))
                seg = col[off:end]
                if not (seg == seg[0]).all():
                    # The whole column is already in memory, so checking every
                    # row (not just the first) is free — and the mid-episode
                    # flip is exactly the single-chunk failure the exhaustive
                    # default exists to catch.
                    return ("varies", (c, f), int(epi[i]))
                out.append((int(epi[i]), int(seg[0])))
            return ("ok", out)

        pairs: list[tuple[int, int]] = []
        varying: list[tuple[tuple[int, int], int]] = []
        n_skipped = 0
        # 8 (not 16) threads: root mode builds up to 16 buckets concurrently, so a
        # wider per-bucket pool only oversubscribes the shared IO path.
        with ThreadPoolExecutor(max_workers=8) as pool:
            for res in pool.map(_gather, shard_items):
                if res[0] == "ok":
                    pairs.extend(res[1])
                elif res[0] == "varies":
                    varying.append((res[1], res[2]))
                else:
                    _, shard, why, n_eps = res
                    n_skipped += n_eps
                    logger.warning(
                        "WorldEngine(%s): prompt-consistency pass skipped data shard %s (%s; %d episodes "
                        "unverified — shard integrity is the scanner's job).",
                        self._dataset_id,
                        shard,
                        why,
                        n_eps,
                    )

        if varying:
            preview = ", ".join(f"episode {e} (data shard {s})" for s, e in varying[:5])
            raise DataContractError(
                f"WorldEngine({self._dataset_id}): task_index varies WITHIN {len(varying)} probed episode(s) "
                f"({preview}{', …' if len(varying) > 5 else ''}). One task per episode is a layout invariant "
                "of this drop, and the runtime prompt is resolved from each window's first row — so windows "
                "past the flip would resolve a different task than the episode declares (or raise, if the "
                "value is missing from tasks.parquet, killing the DataLoader worker mid-run). Fix the data "
                "(or exclude the episodes) before training."
            )

        pmap = self._task_idx_to_text
        bad = [(e, t) for e, t in pairs if t not in pmap or not pmap[t]]
        if bad:
            preview = ", ".join(f"episode {e} → task_index {t}" for e, t in bad[:5])
            raise DataContractError(
                f"WorldEngine({self._dataset_id}): {len(bad)} probed episode(s) resolve to a missing / "
                f"unusable-English tasks.parquet entry ({preview}{', …' if len(bad) > 5 else ''}). "
                "The episodes-parquet 'tasks' cells and meta/tasks.parquet have diverged — every window of "
                "these episodes would raise at runtime and kill the DataLoader worker. Fix the data (or "
                "exclude the episodes) before training."
            )
        logger.info(
            "WorldEngine(%s): prompt consistency verified for %d/%d episodes across %d/%d data shards "
            "(%d episodes skipped with unreadable shards).",
            self._dataset_id,
            len(pairs),
            len(eps),
            len(shard_items),
            n_shards,
            n_skipped,
        )

    @classmethod
    def _multibucket_wrapper(cls):
        return MultiBucketWorldEngineDataset


# ---------------------------------------------------------------------------
# Multi-bucket wrapper (root mode)
# ---------------------------------------------------------------------------


class MultiBucketWorldEngineDataset(MultiLeRobotV3Reader):
    """Aggregate of the WorldEngine root's ``shard_XXX`` buckets.

    Used in root mode: ``WorldEngineDataset.from_config({dataset_dir: <root>})``
    where ``<root>`` (``/mnt/data/wangyuran/pretrain_dataset/WorldEngine``)
    contains 50 shard subdirs, each with its own ``meta/info.json``. All buckets
    share ``action_dim`` and ``normalization_stats=None`` by design contract, so
    the base class's defaults apply directly.
    """

    def __init__(self, buckets: List["WorldEngineDataset"]):
        super().__init__(buckets)
        logger.info(
            "MultiBucketWorldEngineDataset: %d buckets, %d total windows (action_dim=%d)",
            len(self._buckets),
            len(self),
            self.action_dim,
        )


__all__ = ["WorldEngineDataset", "MultiBucketWorldEngineDataset", "normalize_prompt"]
