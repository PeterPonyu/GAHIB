"""Tests for full benchmark run safety gates and run-plan reporting."""

from __future__ import annotations

from experiments.benchmark_config import (
    DATASET_DIRS_ENV,
    FULL_BENCHMARK_CONFIRM_ENV,
    FULL_BENCHMARK_STEPS,
    SELECTED_DATASETS,
    evaluate_full_benchmark_readiness,
    format_full_benchmark_run_plan,
)


def test_full_benchmark_guard_blocks_without_data_roots_or_confirmation(tmp_path):
    status = evaluate_full_benchmark_readiness(
        env={},
        project_root=str(tmp_path),
        cuda_available=True,
    )

    assert not status.ok
    assert status.inventory.found_count == 0
    assert status.inventory.required_count == 53
    assert any(DATASET_DIRS_ENV in reason for reason in status.reasons)
    assert any(FULL_BENCHMARK_CONFIRM_ENV in reason for reason in status.reasons)

    plan = format_full_benchmark_run_plan(status)
    assert "Full benchmark guard: NOT STARTED" in plan
    assert "Exact run plan when prerequisites are satisfied:" in plan
    assert "bash experiments/run_all_expanded.sh" in plan


def test_full_benchmark_guard_passes_only_when_all_roots_confirmed_and_gpu_ready(tmp_path):
    data_root = tmp_path / "benchmark-data"
    data_root.mkdir()
    for dataset_name in SELECTED_DATASETS:
        (data_root / f"{dataset_name}.h5ad").touch()

    status = evaluate_full_benchmark_readiness(
        env={
            DATASET_DIRS_ENV: str(data_root),
            FULL_BENCHMARK_CONFIRM_ENV: "1",
        },
        project_root=str(tmp_path),
        cuda_available=True,
    )

    assert status.ok
    assert status.reasons == ()
    assert status.inventory.found_count == 53


def test_run_plan_lists_every_expanded_benchmark_step(tmp_path):
    status = evaluate_full_benchmark_readiness(
        env={},
        project_root=str(tmp_path),
        cuda_available=False,
    )
    plan = format_full_benchmark_run_plan(status)

    for label, script in FULL_BENCHMARK_STEPS:
        assert label in plan
        assert script in plan
