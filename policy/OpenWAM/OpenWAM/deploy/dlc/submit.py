#!/usr/bin/env python3
"""Submit a DLC job from a YAML spec via aliyun pai-dlc create-job.

Usage:
    python deploy/dlc/submit.py deploy/dlc/<spec>.yaml [--dry-run]

The YAML schema mirrors the legacy CreateJob HTTP body (top-level JobName,
JobType, WorkspaceId, ResourceId, JobMaxRunningTimeMinutes, Priority, JobSpecs,
UserVpc, DataSources, Envs, UserCommand). Each top-level key is mapped to the
corresponding --<flag> on `aliyun pai-dlc create-job`. List/object fields are
serialized to JSON.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

KEY_TO_FLAG = {
    "JobName": "display-name",
    "JobType": "job-type",
    "WorkspaceId": "workspace-id",
    "ResourceId": "resource-id",
    "JobMaxRunningTimeMinutes": "job-max-running-time-minutes",
    "Priority": "priority",
    "UserCommand": "user-command",
    "JobSpecs": "job-specs",
    "UserVpc": "user-vpc",
    "DataSources": "data-sources",
    "Envs": "envs",
    "CredentialConfig": "credential-config",
    "Settings": "settings",
    "SuccessPolicy": "success-policy",
    "ElasticSpec": "elastic-spec",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--region",
        help="Aliyun region for the API call (e.g. ap-southeast-1 for SG).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra Envs entries injected at submission time. Use for secrets "
        "(HF_TOKEN, WANDB_API_KEY) so they never enter the YAML/git.",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec.read_text())
    # ResourceType isn't a CLI flag; quota id implies Lingjun. Drop silently.
    spec.pop("ResourceType", None)
    # Region can be embedded in the spec for clarity (ap-southeast-1 etc.).
    # Always pop so it doesn't trip the unknown-YAML-key warning below — even
    # when --region overrides it, the YAML key is consumed.
    region_from_yaml = spec.pop("Region", None)
    region = args.region or region_from_yaml
    if args.env:
        envs = spec.setdefault("Envs", {})
        for kv in args.env:
            if "=" not in kv:
                print(f"error: --env expects KEY=VALUE, got {kv!r}", file=sys.stderr)
                return 2
            k, v = kv.split("=", 1)
            envs[k] = v

    cmd = ["aliyun"]
    if region:
        cmd += ["--region", region]
    cmd += ["pai-dlc", "create-job"]
    for key, value in spec.items():
        flag = KEY_TO_FLAG.get(key)
        if flag is None:
            print(f"warn: unknown YAML key {key!r}, skipping", file=sys.stderr)
            continue
        if isinstance(value, (list, dict)):
            payload = json.dumps(value)
        else:
            payload = str(value)
        cmd.extend([f"--{flag}", payload])

    print("→", " ".join(shlex.quote(p) for p in cmd[:3]),
          "\\\n   " + " \\\n   ".join(
              f"{cmd[i]} {shlex.quote(cmd[i+1])[:120]}"
              for i in range(3, len(cmd), 2)),
          file=sys.stderr)

    if args.dry_run:
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
