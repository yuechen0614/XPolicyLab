import importlib.util
import json
import os
import random
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml


def _load_module(repo_root: Path, relative_path: str, name: str):
    module_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_smoke_module(repo_root: Path):
    return _load_module(repo_root, "benchmarks/libero/smoke_libero.py", "libero_smoke")


def _load_interface_module(repo_root: Path):
    return _load_module(repo_root, "benchmarks/libero/openwam2libero_interface.py", "openwam2libero_interface")


def _load_single_eval_module(repo_root: Path, monkeypatch):
    monkeypatch.syspath_prepend(str(repo_root / "benchmarks" / "libero"))
    return _load_module(repo_root, "benchmarks/libero/single_eval.py", "libero_single_eval")


def _load_full_eval_module(repo_root: Path):
    return _load_module(
        repo_root,
        "benchmarks/libero/run_10epoch_all_suites.py",
        "libero_10epoch_full_eval",
    )


def _fake_libero_repo(root: Path):
    package_root = root / "libero" / "libero"
    for name in ["bddl_files", "init_files", "assets"]:
        (package_root / name).mkdir(parents=True, exist_ok=True)
    return package_root


def test_libero_smoke_writes_isolated_configs(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    smoke = _load_smoke_module(repo_root)

    ordinary_repo = tmp_path / "LIBERO"
    plus_repo = tmp_path / "LIBERO-plus"
    ordinary_package_root = _fake_libero_repo(ordinary_repo)
    plus_package_root = _fake_libero_repo(plus_repo)

    ordinary_config_root = tmp_path / "ordinary-config"
    plus_config_root = tmp_path / "plus-config"
    monkeypatch.setenv("LIBERO_PATH", str(ordinary_repo))
    monkeypatch.setenv("LIBERO_PLUS_PATH", str(plus_repo))
    monkeypatch.setenv("LIBERO_CONFIG_ROOT", str(ordinary_config_root))
    monkeypatch.setenv("LIBERO_PLUS_CONFIG_ROOT", str(plus_config_root))

    ordinary = smoke.write_config("ordinary")
    plus = smoke.write_config("plus")

    assert ordinary["benchmark_root"] == str(ordinary_package_root)
    assert plus["benchmark_root"] == str(plus_package_root)
    assert ordinary_config_root != plus_config_root

    ordinary_yaml = yaml.safe_load((ordinary_config_root / "config.yaml").read_text(encoding="utf-8"))
    plus_yaml = yaml.safe_load((plus_config_root / "config.yaml").read_text(encoding="utf-8"))
    assert ordinary_yaml["assets"] == str(ordinary_package_root / "assets")
    assert plus_yaml["assets"] == str(plus_package_root / "assets")


def test_libero_smoke_rejects_missing_repo_env(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    smoke = _load_smoke_module(repo_root)

    monkeypatch.delenv("LIBERO_PATH", raising=False)
    with pytest.raises(SystemExit, match="LIBERO_PATH is not set"):
        smoke.write_config("ordinary")


def test_libero_smoke_config_root_uses_documented_default(tmp_path, monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    smoke = _load_smoke_module(repo_root)
    ordinary_repo = tmp_path / "LIBERO"
    _fake_libero_repo(ordinary_repo)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LIBERO_PATH", str(ordinary_repo))
    monkeypatch.delenv("LIBERO_CONFIG_ROOT", raising=False)
    monkeypatch.delenv("LIBERO_CONFIG_PATH", raising=False)

    smoke.write_config("ordinary")

    assert (tmp_path / "home" / ".libero-openwam" / "config.yaml").is_file()


def test_libero_single_eval_requires_typed_yaml(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    single_eval = _load_single_eval_module(repo_root, monkeypatch)

    with pytest.raises(TypeError, match="YAML boolean"):
        single_eval._require_bool("off", "send_state")
    with pytest.raises(TypeError, match="YAML integer"):
        single_eval._parse_optional_int("8", "state_dim")
    with pytest.raises(TypeError, match="YAML number"):
        single_eval._parse_optional_float("1.0", "action_clip")
    with pytest.raises(ValueError, match="ordinary.*plus"):
        single_eval.run_eval({"flavor": "ordianry"})


def test_libero_single_eval_resolves_contiguous_trial_range(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    single_eval = _load_single_eval_module(repo_root, monkeypatch)

    assert single_eval._resolve_trial_range({}) == (0, 1)
    assert single_eval._resolve_trial_range({"trial_start": 25, "num_trials": 25}) == (25, 50)
    with pytest.raises(ValueError, match="trial_start must be non-negative"):
        single_eval._resolve_trial_range({"trial_start": -1, "num_trials": 25})
    with pytest.raises(ValueError, match="num_trials must be positive"):
        single_eval._resolve_trial_range({"trial_start": 0, "num_trials": 0})
    with pytest.raises(TypeError, match="trial_start must be a YAML integer"):
        single_eval._resolve_trial_range({"trial_start": "25", "num_trials": 25})


def test_libero_single_eval_resolves_suite_specific_max_steps(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    single_eval = _load_single_eval_module(repo_root, monkeypatch)
    cfg = {
        "max_steps": 600,
        "max_steps_by_suite": {
            "libero_spatial": 600,
            "libero_object": 600,
            "libero_goal": 600,
            "libero_10": 700,
        },
    }

    assert single_eval._resolve_max_steps(cfg, "libero_spatial") == 600
    assert single_eval._resolve_max_steps(cfg, "libero_object") == 600
    assert single_eval._resolve_max_steps(cfg, "libero_goal") == 600
    assert single_eval._resolve_max_steps(cfg, "libero_10") == 700


def test_libero_single_eval_retries_placement_failures_in_process(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    single_eval = _load_single_eval_module(repo_root, monkeypatch)

    class RandomizationError(Exception):
        pass

    expected_env = object()
    attempts = []

    def make_env(task, cfg):
        attempts.append((task, cfg))
        if len(attempts) < 3:
            raise RandomizationError("cannot place objects")
        return expected_env

    monkeypatch.setattr(single_eval, "_make_env", make_env)
    cfg = {"rng_mode": "official_global"}

    assert single_eval._make_env_with_randomization_retries("task", cfg) is expected_env
    assert attempts == [("task", cfg), ("task", cfg), ("task", cfg)]


def test_libero_single_eval_does_not_retry_unrelated_env_errors(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    single_eval = _load_single_eval_module(repo_root, monkeypatch)
    attempts = 0

    def make_env(task, cfg):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("unrelated construction failure")

    monkeypatch.setattr(single_eval, "_make_env", make_env)
    with pytest.raises(RuntimeError, match="unrelated construction failure"):
        single_eval._make_env_with_randomization_retries("task", {})
    assert attempts == 1


def test_libero_ordinary_protocol_uses_requested_eval_defaults():
    repo_root = Path(__file__).resolve().parents[1]
    policy = yaml.safe_load((repo_root / "benchmarks" / "libero" / "policy_config.yml").read_text(encoding="utf-8"))

    assert policy["max_steps"] == 600
    assert policy["num_trials"] == 50
    assert policy["max_steps_by_suite"] == {
        "libero_spatial": 600,
        "libero_object": 600,
        "libero_goal": 600,
        "libero_10": 700,
    }
    assert policy["seed"] == 42
    assert policy["rng_mode"] == "environment"
    assert policy["reseed_each_trial"] is False
    assert policy["settle_steps"] == 30
    assert policy["settle_action"] == [0, 0, 0, 0, 0, 0, -1]
    assert policy["camera_height"] == 256
    assert policy["camera_width"] == 256

    full_eval = _load_full_eval_module(repo_root)
    assert full_eval.LIBERO_PROTOCOL_VERSION == "openwam-libero-seed42-settle30-gripm1-h10-v1"
    args = full_eval._build_parser().parse_args([])
    assert args.inference_mode == "sync"
    assert args.inference_horizon == 10


def test_libero_full_eval_maps_long_and_balances_all_tasks():
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)

    suites = full_eval._resolve_suites("spatial,goal,object,long")
    assert suites == ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
    jobs = full_eval._build_jobs(suites, {suite: 10 for suite in suites}, None)
    slots = full_eval._build_replica_slots(list(range(8)), 8920)
    assignments = full_eval._assign_jobs(jobs, slots)

    assert len(jobs) == 40
    assert {job for slot_jobs in assignments.values() for job in slot_jobs} == set(jobs)
    for gpu in range(8):
        gpu_slots = [slot for slot in slots if slot.gpu == gpu]
        assert sum(len(assignments[slot]) for slot in gpu_slots) == 5
        assert sorted(len(assignments[slot]) for slot in gpu_slots) == [2, 3]
        assert assignments[gpu_slots[0]][0] != assignments[gpu_slots[1]][0]


def test_libero_full_eval_sampling_matches_imagewam_per_suite_algorithm():
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    suites = ["libero_spatial", "libero_goal", "libero_object", "libero_10"]
    task_counts = {
        "libero_spatial": 2402,
        "libero_goal": 2591,
        "libero_object": 2518,
        "libero_10": 2519,
    }

    jobs = full_eval._build_jobs(
        suites,
        task_counts,
        None,
        sample_ratio=0.2,
        sample_seed=42,
    )
    actual = {suite: [job.task_id for job in jobs if job.suite == suite] for suite in suites}
    expected = {}
    for suite in suites:
        sample_count = max(1, int(np.ceil(task_counts[suite] * 0.2)))
        rng = random.Random(f"42:{suite}")
        expected[suite] = sorted(rng.sample(range(task_counts[suite]), sample_count))

    assert actual == expected
    assert {suite: len(ids) for suite, ids in actual.items()} == {
        "libero_spatial": 481,
        "libero_goal": 519,
        "libero_object": 504,
        "libero_10": 504,
    }
    assert len(jobs) == 2008


def test_libero_full_eval_uses_default_mujoco_version():
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)

    assert full_eval.REQUIRED_MUJOCO_VERSION == "3.3.2"
    assert full_eval.DEFAULT_CKPT_DIR == Path(
        "/mnt/data/wangyuran/openwam_checkpoints/new-openwam-libero-sft-10epoch-final"
    )
    assert full_eval.DEFAULT_CKPT_NAME == "checkpoint_step_10850.safetensors"
    assert full_eval.DEFAULT_LIBERO_PATH == Path("/mnt/data/wangyuran/LIBERO")
    assert full_eval.DEFAULT_LIBERO_PYTHON == Path("/mnt/data/wangyuran/miniconda3/envs/libero/bin/python")

    environment = yaml.safe_load((repo_root / "benchmarks" / "libero" / "environment.yml").read_text(encoding="utf-8"))
    pip_dependencies = environment["dependencies"][-1]["pip"]
    assert "numpy==1.22.4" in pip_dependencies
    assert "opencv-python==4.6.0.66" in pip_dependencies
    assert "robomimic==0.2.0" in pip_dependencies
    assert "robosuite==1.4.0" in pip_dependencies
    assert "bddl==1.0.1" in pip_dependencies
    assert "mujoco==3.3.2" in pip_dependencies


def test_libero_plus_requested_protocol_is_pinned():
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)

    assert full_eval.DEFAULT_LIBERO_PLUS_PATH == Path("/mnt/data/wangyuran/LIBERO-plus")
    assert full_eval.DEFAULT_LIBERO_PLUS_PYTHON == Path("/mnt/data/wangyuran/miniconda3/envs/libero-plus/bin/python")

    environment = yaml.safe_load(
        (repo_root / "benchmarks" / "libero" / "environment-plus.yml").read_text(encoding="utf-8")
    )
    pip_dependencies = environment["dependencies"][-1]["pip"]
    assert "numpy==1.22.4" in pip_dependencies
    assert "opencv-python==4.6.0.66" in pip_dependencies
    assert "robomimic==0.2.0" in pip_dependencies
    assert "robosuite==1.4.0" in pip_dependencies
    assert "bddl==1.0.1" in pip_dependencies
    assert "wand==0.7.2" in pip_dependencies
    assert "scikit-image==0.19.3" in pip_dependencies
    assert "mujoco==3.3.2" in pip_dependencies

    policy = yaml.safe_load(
        (repo_root / "benchmarks" / "libero" / "policy_config_plus.yml").read_text(encoding="utf-8")
    )
    assert policy["flavor"] == "plus"
    assert policy["num_trials"] == 1
    assert policy["max_steps_by_suite"] == {
        "libero_spatial": 600,
        "libero_object": 600,
        "libero_goal": 600,
        "libero_10": 700,
    }
    assert policy["seed"] == 10000
    assert policy["rng_mode"] == "official_global"
    assert policy["reseed_each_trial"] is False
    assert policy["settle_steps"] == 30
    assert policy["settle_action"] == [0, 0, 0, 0, 0, 0, -1]
    assert policy["camera_height"] == 256
    assert policy["camera_width"] == 256


def test_libero_plus_official_rng_control_is_repeatable(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    single_eval = _load_single_eval_module(repo_root, monkeypatch)

    single_eval._control_seed(10000)
    first = (random.random(), np.random.random())
    single_eval._control_seed(10000)
    second = (random.random(), np.random.random())

    assert first == second


def test_libero_resume_rejects_protocol_mismatch(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    signature = {
        "protocol_version": full_eval.LIBERO_PLUS_PROTOCOL_VERSION,
        "policy_config_sha256": "official-config",
    }
    (tmp_path / "manifest.json").write_text(json.dumps({"resume_signature": signature}), encoding="utf-8")

    full_eval._validate_resume_signature(tmp_path, signature)
    full_eval._validate_resume_signature(
        tmp_path,
        signature | {"render_gpus": [0, 1, 2, 3, 5]},
    )
    with pytest.raises(RuntimeError, match="different evaluation protocol"):
        full_eval._validate_resume_signature(
            tmp_path,
            signature | {"policy_config_sha256": "changed-config"},
        )

    legacy_output = tmp_path / "legacy"
    legacy_output.mkdir()
    (legacy_output / "manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="legacy evaluation"):
        full_eval._validate_resume_signature(legacy_output, signature)


def test_libero_ordinary_launcher_allows_runtime_tuning(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    args = Namespace(
        flavor="ordinary",
        policy_config=repo_root / "benchmarks" / "libero" / "policy_config.yml",
        num_trials=50,
        trial_start=0,
        seed=None,
        smoke=False,
        inference_mode="sync",
        inference_horizon=10,
        denoise_mode="sync",
        denoise_steps=10,
    )

    full_eval._validate_ordinary_protocol(args)
    custom_policy = yaml.safe_load(args.policy_config.read_text(encoding="utf-8"))
    custom_policy["camera_height"] = 128
    custom_policy["camera_width"] = 160
    args.policy_config = tmp_path / "ordinary-custom.yml"
    args.policy_config.write_text(yaml.safe_dump(custom_policy), encoding="utf-8")
    args.inference_mode = "async"
    args.inference_horizon = 32
    args.denoise_mode = "async"
    args.denoise_steps = 5
    full_eval._validate_ordinary_protocol(args)


def test_libero_plus_launcher_keeps_protocol_but_allows_runtime_tuning(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    args = Namespace(
        flavor="plus",
        policy_config=repo_root / "benchmarks" / "libero" / "policy_config_plus.yml",
        num_trials=1,
        trial_start=0,
        seed=None,
        inference_mode="sync",
        inference_horizon=10,
        denoise_mode="sync",
        denoise_steps=10,
    )

    full_eval._validate_plus_protocol(args)
    args.seed = 42
    with pytest.raises(ValueError, match="effective seed must be 10000"):
        full_eval._validate_plus_protocol(args)
    args.seed = None
    custom_policy = yaml.safe_load(args.policy_config.read_text(encoding="utf-8"))
    custom_policy["camera_height"] = 128
    custom_policy["camera_width"] = 160
    args.policy_config = tmp_path / "plus-custom.yml"
    args.policy_config.write_text(yaml.safe_dump(custom_policy), encoding="utf-8")
    args.inference_mode = "async"
    args.inference_horizon = 32
    args.denoise_mode = "async"
    args.denoise_steps = 5
    full_eval._validate_plus_protocol(args)


def test_libero_plus_metadata_is_zero_based_and_category_aware(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    benchmark_root = tmp_path / "libero" / "libero" / "benchmark"
    benchmark_root.mkdir(parents=True)
    (benchmark_root / "task_classification.json").write_text(
        json.dumps(
            {
                "libero_spatial": [
                    {
                        "id": 1,
                        "name": "synthetic_plus_task",
                        "category": "Sensor Noise",
                        "difficulty_level": 2,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    job = full_eval.TaskJob("libero_spatial", 0)

    metadata = full_eval._load_task_metadata("plus", tmp_path, [job])

    assert metadata[job] == full_eval.TaskMetadata(
        name="synthetic_plus_task",
        category="Sensor Noise",
        difficulty_level=2,
    )


def test_libero_plus_summary_aggregates_categories(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    jobs = [
        full_eval.TaskJob("libero_spatial", 0),
        full_eval.TaskJob("libero_spatial", 1),
    ]
    trial_run = full_eval.TrialRun(0, 1)
    metadata = {
        jobs[0]: full_eval.TaskMetadata("task_zero", "Sensor Noise", 1),
        jobs[1]: full_eval.TaskMetadata("task_one", "Objects Layout", 2),
    }
    for job, success in zip(jobs, (1, 0), strict=True):
        run_dir = full_eval._run_output_dir(tmp_path, job, trial_run) / "result"
        run_dir.mkdir(parents=True)
        (run_dir / "results.json").write_text(
            json.dumps(
                {
                    "suite": job.suite,
                    "task_id": job.task_id,
                    "trial_start": 0,
                    "trial_stop": 1,
                    "num_trials": 1,
                    "successes": success,
                    "trials": [{"trial": 0}],
                }
            ),
            encoding="utf-8",
        )

    summary, complete = full_eval._summarize(tmp_path, jobs, trial_run, metadata, write_files=True)

    assert complete
    assert summary["overall"]["success_rate"] == 0.5
    categories = {row["category"]: row for row in summary["category_results"]}
    assert categories["Sensor Noise"]["success_rate"] == 1.0
    assert categories["Objects Layout"]["success_rate"] == 0.0
    csv_text = (tmp_path / "summary.csv").read_text(encoding="utf-8")
    assert "task_zero,Sensor Noise,1" in csv_text


def test_libero_full_eval_uses_two_unique_ports_per_gpu():
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)

    ports = [port for slot in range(8) for port in full_eval._server_ports(8920, slot)]
    assert ports == list(range(8920, 8936))
    assert len(ports) == len(set(ports))


def test_libero_full_eval_uses_one_contiguous_fifty_trial_run():
    repo_root = Path(__file__).resolve().parents[1]
    full_eval = _load_full_eval_module(repo_root)
    trial_run = full_eval.TrialRun(trial_start=0, num_trials=50)

    assert range(trial_run.trial_start, trial_run.trial_stop) == range(0, 50)


def test_libero_policy_reuses_ws_and_rotates_images(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    iface = _load_interface_module(repo_root)

    instances = []

    class FakeWSPolicyClient:
        def __init__(self, *args, **kwargs):
            self.closed = 0
            self.resets = 0
            self.predictions = []
            instances.append(self)

        def ping(self):
            return {"type": iface.transport.PONG}

        def reset(self):
            self.resets += 1
            return {"type": iface.transport.RESET_ACK}

        def predict(self, payload):
            self.predictions.append(payload)
            # Raw EEF10 full pose == the current obs pose (identity rot) -> the
            # bridge must emit a zero OSC delta. The trailing -1 is the trained
            # open-scale gripper meaning CLOSED, which the bridge negates into
            # LIBERO's own +1 = close command.
            return {"action": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0]}

        def close(self):
            self.closed += 1

    encoded_images = []

    def capture_image(image):
        encoded_images.append(np.asarray(image).copy())
        return "encoded"

    monkeypatch.setattr(iface, "WSPolicyClient", FakeWSPolicyClient)
    monkeypatch.setattr(iface.client, "encode_numpy_b64", capture_image)

    policy = iface.OpenWAMLiberoPolicy()
    ws = instances[0]
    policy.reset()
    action = policy.act(
        {
            "agentview_image": np.array(
                [
                    [[1, 0, 0], [2, 0, 0]],
                    [[3, 0, 0], [4, 0, 0]],
                ],
                dtype=np.uint8,
            ),
            "robot0_eye_in_hand_image": np.zeros((2, 2, 3), dtype=np.uint8),
            "robot0_eef_pos": np.zeros(3, dtype=np.float32),
            "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32),
        },
        "pick up the bowl",
    )

    assert action.shape == (7,)
    np.testing.assert_allclose(action[:6], np.zeros(6), atol=1e-5)
    # Trained open-scale -1 (closed) reaches the env as its native +1 (close).
    assert action[6] == 1.0
    assert ws.resets == 1
    assert ws.closed == 0
    assert ws.predictions[0]["images"]["head_camera"] == "encoded"
    assert encoded_images[0].shape == (256, 320, 3)
    raw_head = np.array(
        [
            [[1, 0, 0], [2, 0, 0]],
            [[3, 0, 0], [4, 0, 0]],
        ],
        dtype=np.uint8,
    )
    expected_head = iface.resize_for_lshape_slot(raw_head[::-1, ::-1], "head_camera")
    np.testing.assert_array_equal(encoded_images[0], expected_head)
    # The proprio sent is the raw EEF10 (identity rot6d; qpos [0.04, -0.04] is a
    # fully open hand, which the open-scale renders as +1).
    sent_state = ws.predictions[0]["state"]
    assert len(sent_state) == 10
    np.testing.assert_allclose(sent_state, [0, 0, 0, 1, 0, 0, 0, 1, 0, 1], atol=1e-6)
    policy.close()
    assert ws.closed == 1


def test_libero_shell_scripts_are_valid():
    repo_root = Path(__file__).resolve().parents[1]
    scripts = [
        repo_root / "benchmarks" / "libero" / "run_smoke.sh",
        repo_root / "benchmarks" / "libero" / "single_eval.sh",
        repo_root / "benchmarks" / "libero" / "setup_env.sh",
        repo_root / "benchmarks" / "libero" / "setup_plus_env.sh",
        repo_root / "benchmarks" / "libero" / "run_10epoch.sh",
        repo_root / "benchmarks" / "libero" / "run_plus.sh",
    ]
    result = subprocess.run(
        ["bash", "-n", *map(os.fspath, scripts)],
        cwd=repo_root,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_libero_shell_wrappers_resolve_python_from_path(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    fake_repo = tmp_path / "LIBERO"
    fake_repo.mkdir()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    env = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "LIBERO_PATH": str(fake_repo),
        "LIBERO_PYTHON": "python",
    }

    for script, args in (
        ("run_smoke.sh", ["ordinary", "import"]),
        ("single_eval.sh", ["ordinary", "libero_spatial", "0"]),
    ):
        result = subprocess.run(
            ["bash", str(repo_root / "benchmarks" / "libero" / script), *args],
            cwd=repo_root,
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert str(fake_python) in result.stdout
