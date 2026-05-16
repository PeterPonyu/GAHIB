from __future__ import annotations

from experiments import benchmark_config as bc


def test_get_configured_dataset_dirs_falls_back_to_repo_local_data(tmp_path):
    env = {}
    dirs, env_was_set = bc.get_configured_dataset_dirs(env=env, project_root=str(tmp_path))

    assert dirs == (str(tmp_path / "data"),)
    assert env_was_set is False


def test_get_configured_dataset_dirs_uses_explicit_env_roots():
    env = {bc.DATASET_DIRS_ENV: "/data/a:/data/b"}
    dirs, env_was_set = bc.get_configured_dataset_dirs(env=env, project_root="/unused")

    assert dirs == ("/data/a", "/data/b")
    assert env_was_set is True


def test_full_benchmark_readiness_requires_explicit_acknowledgement():
    env = {}
    readiness = bc.evaluate_full_benchmark_readiness(
        env=env,
        project_root="/repo",
        cuda_available=False,
    )

    assert readiness.ok is False
    assert readiness.inventory.search_dirs == ("/repo/data",)
    assert any(bc.DATASET_DIRS_ENV in reason for reason in readiness.reasons)
    assert any(bc.FULL_BENCHMARK_CONFIRM_ENV in reason for reason in readiness.reasons)
    assert any(bc.FULL_BENCHMARK_ALLOW_CPU_ENV in reason for reason in readiness.reasons)


def test_format_full_benchmark_run_plan_mentions_exact_public_safe_path():
    readiness = bc.evaluate_full_benchmark_readiness(
        env={
            bc.DATASET_DIRS_ENV: "/data/a:/data/b",
            bc.FULL_BENCHMARK_CONFIRM_ENV: "1",
            bc.FULL_BENCHMARK_ALLOW_CPU_ENV: "1",
        },
        project_root="/repo",
        cuda_available=True,
    )

    plan = bc.format_full_benchmark_run_plan(readiness)
    assert "GAHIB_CONFIRM_FULL_BENCHMARK=1" in plan
    assert "bash experiments/run_all_expanded.sh" in plan
    assert "docs/DATASET_MANIFEST.csv" in plan
