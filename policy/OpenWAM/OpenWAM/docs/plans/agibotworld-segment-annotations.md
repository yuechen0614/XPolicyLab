# AgiBotWorld Segment Annotation Dataloader Notes

This note documents the local AgiBotWorld dataloader changes for reviewers or
future automated checks.

## Dataset Fields

The AgiBotWorld-Beta LeRobot shards are expected to contain two episode-level
constant fields, stored in both data parquet shards and `meta/episodes` parquet:

- `segment_flag`
- `segment_delta`

`meta/info.json` also declares both fields under `features` and stores a
`segment_annotation` metadata block. `meta/stats.json` contains stats entries for
both fields. The old personal-prefix field names should not appear in the active
dataset schema.

## Flag Semantics

- `segment_flag == 0`: ordinary segment. Keep the full segment unchanged.
- `segment_flag == 1`: first saved segment of a raw episode. Skip the leading
  static prefix before the true motion start.
- `segment_flag == 2`: last saved segment of a raw episode. Skip the trailing
  static suffix after the true motion end.
- `segment_flag == 3`: mostly or fully static segment. Drop the whole segment
  from training.

For flags `0` and `3`, `segment_delta` is `0`.

For flag `1`, `segment_delta` is the number of frames from the original segment
start to the true motion start. For flag `2`, `segment_delta` is the number of
trailing static frames after the true motion end.

## Code Changes

`configs/dataloader/agibotworld.yaml`

- Adds `use_segment_annotations: true`.
- Adds `segment_max_trim_ratio: 0.7`.
- Documents the flag behavior in the config.

`openwam/dataloader/agibotworld.py`

- Adds `use_segment_annotations` and `segment_max_trim_ratio` to
  `AgiBotWorldDataset.CONFIG_KEYS`, so yaml config loading can pass the options
  through.
- Reads `segment_flag` and `segment_delta` in `_filter_episodes`.
- Drops `segment_flag == 3` episodes before window indexing.
- Adds `_valid_start` for `segment_flag == 1`.
- Adds `_valid_end` for `segment_flag == 2`.
- Leaves `segment_flag == 0` unchanged.
- If the fields are missing, logs a warning and falls back to the old full-segment
  behavior.

## Over-trim Episode Drop

`segment_max_trim_ratio` drops an episode outright when the trim would remove at
least that fraction of its raw length (`0.7` → an episode keeping under 30% of
its frames is discarded). `null` disables the rule. Measured against the raw
`length`, so it is independent of which end was trimmed; `segment_flag == 3` rows
are excluded from the count because they are already dropped.

`_validate_trim_ratio` raises on anything outside `(0, 1]` rather than clamping —
a config typo like `70` would otherwise silently disable the rule, and `0` would
drop every episode including the untrimmed ones. Setting the ratio with
`use_segment_annotations: false` logs a warning and has no effect.

Measured on AgiBotWorld-Beta (2026-08-02, 928,722 episodes / 2281.08 h after
trimming), the trimmed fraction per episode is p50 0.057, p90 0.157, p99 0.389,
max 0.897. The rule is therefore a guard against annotation drift, not a material
filter:

| threshold | episodes dropped | hours lost | remaining |
| --------- | ---------------- | ---------- | --------- |
| 0.7       | 47               | 0.06 h     | 2281.03 h |
| 0.5       | 452              | 0.75 h     | 2280.34 h |

At 0.7 the 47 episodes span 13 buckets (577: 16, 741: 11, 390: 4, 377: 3,
740: 3, …), split 31 flag-1 / 16 flag-2.

It does NOT catch short-but-clean episodes: 25,309 episodes are under 33 frames
after trimming, and every training window they yield is partly padded.

`openwam/dataloader/bases/lerobot_v3_reader.py`

- Reads optional `_valid_start` and `_valid_end` columns after episode filtering.
- Builds the window index from `valid_end - valid_start`, not raw episode length.
- Uses the same adjusted `offset` for both parquet rows and video decoding.

This means skipped frames are removed consistently for video, action, proprio,
and masks. The video/action alignment is preserved because both paths use the
same adjusted `offset`.

## Review Checklist (superseded by tests/dataloader/test_agibotworld.py)

- Dataset schema contains `segment_flag` and `segment_delta`.
- Dataset schema does not contain old personal-prefix segment field names.
- `segment_flag == 3` episodes are absent from `AgiBotWorldDataset._eps_df`.
- For `segment_flag == 1`, `_valid_start == segment_delta`.
- For `segment_flag == 2`, `_valid_end == length - segment_delta`.
- In `LeRobotV3Reader._getitem_impl`, the adjusted `offset` is used for both:
  `table.slice(data_row_offset + offset, ...)` and `_decode_window_video(..., offset, ...)`.

## Validation Used

`tests/dataloader/test_agibotworld.py` (31 tests) pins the flag semantics, the
`delta >= length` drop, the disabled / missing-column fallbacks, the
`segment_max_trim_ratio` boundary, and — on a synthetic bucket whose parquet rows
and decoded frames both encode their own index — that no window reads outside its
own episode and that no trimmed frame is ever served.

On the real dataset (`outputs/agibot_probe/`, 2026-08-02, root
`/mnt/data/wangyuran/pretrain_dataset/AgiBotWorld-Beta`):

- 211/211 buckets build; 928,675 episodes / 123,175,420 windows / 2281.03 h with
  `segment_max_trim_ratio: 0.7`.
- 41,820 windows chosen to open every one of the 649 data parquets and all 21,313
  referenced mp4s: 0 failures.
- 480 windows (first + last of 240 episodes, 97 flag-1 / 90 flag-2 / 53 flag-0)
  byte-compared against parquet rows and mp4 frames read independently of the
  reader: 0 mismatches.
- Sample shapes: `video` 9 frames of 384x320, `action` `(32, 80)`,
  `proprio` `(1, 80)`.
- `total_hours: 500.0` also builds cleanly, but yields 500.30 h raw / 492.66 h
  trainable — the budget is spent on raw `length`, so it over-counts the trim.

