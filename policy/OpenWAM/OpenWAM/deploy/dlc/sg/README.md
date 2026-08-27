# SG DLC Job Specs

Files in this directory are Alibaba PAI-DLC job specs for Singapore
(`ap-southeast-1`) runs. They are intentionally verbose because DLC needs the
full cloud submission form in one YAML file:

- workspace, quota, region, and max runtime
- image and per-pod CPU / memory / GPU resources
- VPC, switch, and security group
- CPFS and OSS mounts
- environment variables for HF cache, W&B, NCCL, and commit checks
- preflight checks for clean code, expected commit, dataset buckets, and model
  cache

These YAMLs are the machine execution layer, not the human experiment
entrypoint.

## Current Specs

- `oss_dataset_layout_probe.yaml` is a one-off infrastructure/data check (OSS
  `lerobot_epic` layout probe), not a training recipe.

## Submit

From the repository root:

```bash
COMMIT="$(git rev-parse HEAD)"

python3 deploy/dlc/submit.py \
  deploy/dlc/sg/<spec>.yaml \
  --env "EXPECTED_COMMIT=${COMMIT}" \
  --env "REPO_DIR=/cpfs/<mirror-path>" \
  --env "PYTHONPATH=/cpfs/<mirror-path>"
```

The job fails fast if the remote `REPO_DIR` is dirty or does not match
`EXPECTED_COMMIT`. The `REPO_DIR` path is not created by DLC submission. Sync or
clone the target commit to the remote CPFS mirror before submitting, then pass
that exact path as both `REPO_DIR` and `PYTHONPATH`.
