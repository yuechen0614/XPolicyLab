#!/usr/bin/env python3
"""Audit a completed LIBERO-plus evaluation and its referenced videos."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import run_10epoch_all_suites as runner

EXPECTED_VIDEO_VIEWS = ("agentview", "model_layout", "left_wrist")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-checkpoint", type=Path)
    parser.add_argument("--expected-total", type=int, default=10_030)
    parser.add_argument("--expected-mujoco-version")
    parser.add_argument("--expected-inference-horizon", type=int)
    parser.add_argument("--expected-seed", type=int)
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Defaults to OUTPUT_DIR/completion_audit.json.",
    )
    return parser.parse_args()


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()


def _load_jobs(libero_path: Path) -> tuple[list[runner.TaskJob], dict]:
    classification_path = libero_path / "libero" / "libero" / "benchmark" / "task_classification.json"
    classification = json.loads(classification_path.read_text(encoding="utf-8"))
    jobs = [runner.TaskJob(suite, int(item["id"]) - 1) for suite, records in classification.items() for item in records]
    metadata = runner._load_task_metadata("plus", libero_path, jobs)
    return jobs, metadata


def _summary_core(summary: dict) -> dict:
    return {
        "overall": summary.get("overall"),
        "suite_results": sorted(summary.get("suite_results", []), key=lambda row: row["suite"]),
        "category_results": sorted(summary.get("category_results", []), key=lambda row: row["category"]),
        "missing_runs": sorted(
            summary.get("missing_runs", []),
            key=lambda row: (
                row["suite"],
                int(row["task_id"]),
                int(row["trial_start"]),
                int(row["trial_stop"]),
            ),
        ),
    }


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_json = (
        args.output_json.expanduser().resolve()
        if args.output_json is not None
        else output_dir / "completion_audit.json"
    )
    errors: list[str] = []

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    if args.expected_checkpoint is not None and not _same_path(checkpoint, args.expected_checkpoint):
        errors.append(f"checkpoint mismatch: {checkpoint} != {args.expected_checkpoint}")
    if manifest.get("flavor") != "plus":
        errors.append(f"flavor mismatch: {manifest.get('flavor')!r} != 'plus'")
    if args.expected_mujoco_version is not None and manifest.get("mujoco_version") != args.expected_mujoco_version:
        errors.append(f"MuJoCo mismatch: {manifest.get('mujoco_version')!r} != {args.expected_mujoco_version!r}")
    if (
        args.expected_inference_horizon is not None
        and manifest.get("inference_horizon") != args.expected_inference_horizon
    ):
        errors.append(
            f"inference horizon mismatch: {manifest.get('inference_horizon')!r} != {args.expected_inference_horizon!r}"
        )
    effective_seed = manifest.get("effective_seed", manifest.get("seed_override"))
    if args.expected_seed is not None and effective_seed != args.expected_seed:
        errors.append(f"seed mismatch: {effective_seed!r} != {args.expected_seed!r}")

    libero_path = Path(manifest["libero_path"]).expanduser().resolve()
    jobs, metadata = _load_jobs(libero_path)
    trial_data = manifest["trial_run"]
    trial_run = runner.TrialRun(int(trial_data["trial_start"]), int(trial_data["num_trials"]))
    summary, complete = runner._summarize(output_dir, jobs, trial_run, metadata, write_files=False)

    if len(jobs) != args.expected_total:
        errors.append(f"expected job count mismatch: {len(jobs)} != {args.expected_total}")
    if len(summary["task_results"]) != args.expected_total:
        errors.append(f"task result row count mismatch: {len(summary['task_results'])} != {args.expected_total}")
    if not complete or summary["missing_runs"]:
        errors.append(f"missing valid task results: {len(summary['missing_runs'])}")

    task_keys = [(row["suite"], int(row["task_id"])) for row in summary["task_results"]]
    if len(set(task_keys)) != len(task_keys):
        errors.append("task result keys are not unique")

    missing_videos: list[str] = []
    empty_videos: list[str] = []
    malformed_video_maps: list[str] = []
    referenced_video_count = 0
    for row in summary["task_results"]:
        if not row["complete"]:
            continue
        result_path = Path(row["result_paths"][0])
        result = json.loads(result_path.read_text(encoding="utf-8"))
        for trial in result["trials"]:
            video_paths = trial.get("video_paths")
            label = f"{row['suite']}/task{row['task_id']}/trial{trial['trial']}"
            if not isinstance(video_paths, dict):
                malformed_video_maps.append(label)
                continue
            missing_views = set(EXPECTED_VIDEO_VIEWS) - set(video_paths)
            if missing_views:
                malformed_video_maps.append(f"{label}: missing views {sorted(missing_views)}")
            for view in EXPECTED_VIDEO_VIEWS:
                raw_path = video_paths.get(view)
                if not raw_path:
                    continue
                referenced_video_count += 1
                video_path = Path(raw_path)
                if not video_path.is_file():
                    missing_videos.append(str(video_path))
                elif video_path.stat().st_size <= 0:
                    empty_videos.append(str(video_path))

    expected_video_count = args.expected_total * trial_run.num_trials * len(EXPECTED_VIDEO_VIEWS)
    if referenced_video_count != expected_video_count:
        errors.append(f"referenced video count mismatch: {referenced_video_count} != {expected_video_count}")
    if malformed_video_maps:
        errors.append(f"malformed video maps: {len(malformed_video_maps)}")
    if missing_videos:
        errors.append(f"missing referenced videos: {len(missing_videos)}")
    if empty_videos:
        errors.append(f"empty referenced videos: {len(empty_videos)}")

    on_disk_summary_path = output_dir / "summary.json"
    if not on_disk_summary_path.is_file():
        errors.append("summary.json is missing")
    else:
        on_disk_summary = json.loads(on_disk_summary_path.read_text(encoding="utf-8"))
        if _summary_core(on_disk_summary) != _summary_core(summary):
            errors.append("summary.json does not match the audited result set")
    if not (output_dir / "summary.csv").is_file():
        errors.append("summary.csv is missing")

    audit = {
        "audited_at": datetime.now().astimezone().isoformat(),
        "passed": not errors,
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "requirements": {
            "expected_total": args.expected_total,
            "mujoco_version": args.expected_mujoco_version,
            "inference_horizon": args.expected_inference_horizon,
            "seed": args.expected_seed,
            "video_views": list(EXPECTED_VIDEO_VIEWS),
        },
        "overall": summary["overall"],
        "suite_results": summary["suite_results"],
        "category_results": summary["category_results"],
        "missing_runs": summary["missing_runs"],
        "referenced_video_count": referenced_video_count,
        "expected_video_count": expected_video_count,
        "malformed_video_maps": malformed_video_maps,
        "missing_videos": missing_videos,
        "empty_videos": empty_videos,
        "errors": errors,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_json.with_suffix(output_json.suffix + ".tmp")
    temporary_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    temporary_path.replace(output_json)
    print(json.dumps(audit, indent=2))
    return 0 if audit["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
