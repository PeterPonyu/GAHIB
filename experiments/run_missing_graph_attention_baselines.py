#!/usr/bin/env python3
"""Run only missing reviewer-requested graph-attention-style baselines.

This runner is intentionally narrower than ``run_sc_deeplearning_benchmark.py``.
It preserves existing deep-learning benchmark rows and appends/replaces only the
transparent PyTorch ``scGAC-style``, ``SCEA-style``, and ``scVAG-style`` rows.
All preprocessing, labels, and metric calculation use the shared GAHIB policy.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Iterable

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.benchmark_config import (  # noqa: E402
    DATASET_DIRS_ENV,
    FULL_BENCHMARK_ALLOW_CPU_ENV,
    FULL_BENCHMARK_CONFIRM_ENV,
    evaluate_full_benchmark_readiness,
    format_full_benchmark_run_plan,
)
from experiments.exp_utils import discover_datasets, evaluate_latent, get_labels, load_and_preprocess  # noqa: E402
from experiments.external_baselines import (  # noqa: E402
    ALL_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_SPECS,
    graph_attention_style_training_config_json,
    resolve_online_graph_attention_method,
    train_online_graph_attention,
)
from experiments.external_baselines.online_graph_attention import ExternalBaselineResult  # noqa: E402

EXPERIMENT = "sc_deeplearning_benchmark"
PREFIX = "scdeep"
RESULTS_DIR = PROJECT_ROOT / "GAHIB_results" / EXPERIMENT
TABLES_DIR = RESULTS_DIR / "tables"
LOG_DIR = RESULTS_DIR / "logs"
STYLE_METHODS = tuple(ALL_GRAPH_ATTENTION_METHODS)
CANONICAL_METHOD_ORDER = (
    "scVI",
    "CellBLAST",
    "CLEAR",
    "SCALEX",
    "scDAC",
    "scDeepCluster",
    "scDHMap",
    "scGNN",
    "scGCC",
    "scSMD",
    "siVAE",
    *STYLE_METHODS,
    "GAHIB",
)


def _dataset_name(path: str | Path) -> str:
    return Path(path).name.removesuffix(".h5ad")


def _table_path(dataset_name: str) -> Path:
    return TABLES_DIR / f"{PREFIX}_{dataset_name}_df.csv"


def _status_path(dataset_name: str) -> Path:
    return TABLES_DIR / f"{PREFIX}_{dataset_name}_external_status.csv"


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["method"])
    frame = pd.read_csv(path)
    if "method" not in frame.columns:
        frame = frame.rename(columns={frame.columns[0]: "method"})
    frame["method"] = frame["method"].astype(str)
    return frame


def _read_status(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if "method" in frame.columns:
        frame["method"] = frame["method"].astype(str)
    return frame


def _ordered_methods(methods: Iterable[str]) -> list[str]:
    order = {method: index for index, method in enumerate(CANONICAL_METHOD_ORDER)}
    return sorted(methods, key=lambda method: (order.get(method, len(order)), method))


def _order_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "method" not in frame.columns:
        return frame
    ordered = _ordered_methods(frame["method"].astype(str).tolist())
    rank = {method: index for index, method in enumerate(ordered)}
    return (
        frame.assign(_method_rank=frame["method"].astype(str).map(rank))
        .sort_values("_method_rank", kind="stable")
        .drop(columns=["_method_rank"])
        .reset_index(drop=True)
    )


def missing_style_methods(frame: pd.DataFrame, *, force: bool = False) -> list[str]:
    """Return style baselines that should be run for one existing result table."""

    if force or frame.empty or "method" not in frame.columns:
        return list(STYLE_METHODS)
    present = set(frame["method"].astype(str))
    return [method for method in STYLE_METHODS if method not in present]


def external_status_row(dataset_name: str, result: ExternalBaselineResult) -> dict[str, object]:
    """Public-safe status metadata for one style-baseline attempt."""

    method_key = resolve_online_graph_attention_method(result.method)
    spec = ONLINE_GRAPH_ATTENTION_SPECS[method_key]
    return {
        "dataset": dataset_name,
        "method": result.method,
        "status": result.status,
        "reason": result.reason,
        "repo_url": spec.repo_url,
        "commit": spec.commit,
        "license": spec.license,
        "checkout": result.checkout,
        "command": result.command,
        "training_config": result.training_config,
        "elapsed": result.elapsed,
        "outputs": ";".join(result.output_files),
        "source_note": spec.source_note,
        "equivalence_note": spec.equivalence_note,
    }


def metrics_row(method: str, result: ExternalBaselineResult, labels: np.ndarray) -> dict[str, object]:
    """Build one benchmark table row without fabricating failed metrics."""

    row: dict[str, object] = {"method": method}
    if result.status == "ok" and result.latent is not None:
        row.update(evaluate_latent(result.latent, labels))
        row["train_time"] = result.elapsed
    return row


def merge_rows(existing: pd.DataFrame, new_rows: list[dict[str, object]], *, force: bool) -> pd.DataFrame:
    if force and not existing.empty and "method" in existing.columns:
        existing = existing[~existing["method"].astype(str).isin(STYLE_METHODS)].copy()
    if not new_rows:
        return _order_table(existing)
    merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    merged = merged.drop_duplicates(subset=["method"], keep="last")
    return _order_table(merged)


def merge_status(existing: pd.DataFrame, new_rows: list[dict[str, object]], *, force: bool) -> pd.DataFrame:
    if existing.empty and not new_rows:
        return existing
    if not existing.empty and "method" in existing.columns:
        rerun_methods = set(STYLE_METHODS if force else [row["method"] for row in new_rows])
        existing = existing[~existing["method"].astype(str).isin(rerun_methods)].copy()
    if new_rows:
        existing = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    return existing.reset_index(drop=True)


def validate_readiness(*, allow_partial_data: bool = False) -> None:
    status = evaluate_full_benchmark_readiness()
    if allow_partial_data:
        reasons = tuple(
            reason
            for reason in status.reasons
            if "configured data roots provide" not in reason
        )
    else:
        reasons = status.reasons
    if reasons:
        print(format_full_benchmark_run_plan(status))
        raise SystemExit(2)


def write_run_log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / "missing_graph_attention_latest.status"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")



def sort_datasets_by_size(datasets: list[str]) -> list[str]:
    """Run smaller datasets first so a later oversized file cannot block all progress."""

    return sorted(datasets, key=lambda path: (Path(path).stat().st_size if Path(path).exists() else 0, _dataset_name(path)))


def run_isolated_datasets(datasets: list[str], args: argparse.Namespace) -> int:
    """Run one child Python process per dataset for OOM-resumable execution."""

    print("Isolated dataset mode: one child process per dataset; existing rows are preserved.")
    failures = 0
    for index, path in enumerate(datasets, start=1):
        dataset_name = _dataset_name(path)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset",
            dataset_name,
            "--epochs",
            str(args.epochs),
            "--no-isolate",
        ]
        if args.force:
            cmd.append("--force")
        if args.allow_partial_data:
            cmd.append("--allow-partial-data")
        print(f"\n[{index}/{len(datasets)}] child start: {dataset_name}")
        write_run_log(json.dumps({"event": "child_start", "dataset": dataset_name, "index": index}, sort_keys=True))
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy(), text=True, check=False)
        if proc.returncode != 0:
            failures += 1
            print(f"  child failed: {dataset_name} returncode={proc.returncode}")
            write_run_log(json.dumps({
                "event": "child_failed",
                "dataset": dataset_name,
                "returncode": proc.returncode,
            }, sort_keys=True))
        else:
            write_run_log(json.dumps({"event": "child_ok", "dataset": dataset_name}, sort_keys=True))
    if failures:
        print(f"Isolated run completed with {failures} failed dataset child process(es).")
        return 1
    print("Isolated run completed without child-process failures.")
    return 0

def run_dataset(filepath: str | Path, *, epochs: int, force: bool) -> dict[str, object]:
    dataset_name = _dataset_name(filepath)
    table_path = _table_path(dataset_name)
    status_path = _status_path(dataset_name)
    existing = _read_table(table_path)
    methods_to_run = missing_style_methods(existing, force=force)

    if not methods_to_run:
        print(f"  Skipping {dataset_name}: all style-baseline rows already present")
        return {"dataset": dataset_name, "status": "skipped", "methods": []}

    print(f"\n{'─' * 72}")
    print(f"Dataset: {dataset_name}")
    print(f"Style methods to run: {', '.join(methods_to_run)}")
    print(f"Existing table: {table_path if table_path.exists() else '<new>'}")
    print(f"{'─' * 72}")

    adata = load_and_preprocess(str(filepath))
    labels, _ = get_labels(adata)

    metric_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    dataset_start = time.time()
    for method in methods_to_run:
        print(f"  Training {method} for {epochs} epochs...")
        try:
            result = train_online_graph_attention(
                method,
                adata,
                labels,
                dataset_name=dataset_name,
                epochs=epochs,
            )
        except Exception as exc:  # keep run resumable and visible
            result = ExternalBaselineResult(
                method=method,
                status="not_runnable",
                reason=f"adapter_error: {type(exc).__name__}: {exc}",
                training_config=graph_attention_style_training_config_json(method, epochs=epochs),
            )
            traceback.print_exc()
        status_rows.append(external_status_row(dataset_name, result))
        metric_rows.append(metrics_row(method, result, labels))
        if result.status == "ok" and result.latent is not None:
            ari = metric_rows[-1].get("ARI")
            print(f"    ✓ {method}: ARI={ari if ari is not None else 'NA'}, time={result.elapsed:.1f}s")
        else:
            print(f"    ○ {method}: {result.status} — {result.reason}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    merged = merge_rows(existing, metric_rows, force=force)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(table_path, index=False)

    existing_status = _read_status(status_path)
    merged_status = merge_status(existing_status, status_rows, force=force)
    merged_status.to_csv(status_path, index=False)

    elapsed = time.time() - dataset_start
    print(f"  Saved metrics: {table_path}")
    print(f"  Saved status:  {status_path}")
    print(f"  Dataset elapsed: {elapsed:.1f}s")

    del adata
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"dataset": dataset_name, "status": "ran", "methods": methods_to_run, "elapsed": elapsed}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs per style baseline.")
    parser.add_argument("--limit", type=int, default=None, help="Run at most N datasets after filtering.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset basename to include; repeatable.")
    parser.add_argument("--force", action="store_true", help="Recompute style rows even when present.")
    parser.add_argument("--dry-run", action="store_true", help="Print missing rows and exit without training.")
    parser.add_argument("--no-isolate", action="store_true", help="Run selected datasets in this process instead of child processes.")
    parser.add_argument("--manifest-order", action="store_true", help="Use manifest order instead of sorting datasets by file size.")
    parser.add_argument(
        "--allow-partial-data",
        action="store_true",
        help="Allow configured data roots with fewer than 53 datasets, useful for smoke tests only.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_readiness(allow_partial_data=args.allow_partial_data)
    datasets = discover_datasets()
    if args.dataset:
        wanted = set(args.dataset)
        datasets = [path for path in datasets if _dataset_name(path) in wanted]
    if not args.manifest_order:
        datasets = sort_datasets_by_size(datasets)
    if args.limit is not None:
        datasets = datasets[: args.limit]

    print("Missing graph-attention style baseline runner")
    print(f"Dataset roots: {os.environ.get(DATASET_DIRS_ENV)}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Confirmation: {FULL_BENCHMARK_CONFIRM_ENV}={os.environ.get(FULL_BENCHMARK_CONFIRM_ENV)}")
    print(f"CPU override: {FULL_BENCHMARK_ALLOW_CPU_ENV}={os.environ.get(FULL_BENCHMARK_ALLOW_CPU_ENV, '')}")
    print(f"Epochs: {args.epochs}; force={args.force}; dry_run={args.dry_run}")
    print(f"Datasets selected: {len(datasets)}")

    summary = []
    for path in datasets:
        dataset_name = _dataset_name(path)
        table = _read_table(_table_path(dataset_name))
        methods = missing_style_methods(table, force=args.force)
        if methods:
            summary.append({"dataset": dataset_name, "methods": methods})
    print(f"Datasets needing style rows: {len(summary)}")
    for item in summary[:20]:
        print(f"  {item['dataset']}: {', '.join(item['methods'])}")
    if len(summary) > 20:
        print(f"  ... +{len(summary) - 20} more")

    if args.dry_run:
        return 0

    if not args.no_isolate and len(datasets) > 1:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        write_run_log(json.dumps({"event": "isolated_start", "epochs": args.epochs, "datasets": len(datasets)}, sort_keys=True))
        code = run_isolated_datasets(datasets, args)
        write_run_log(json.dumps({"event": "isolated_complete", "returncode": code}, sort_keys=True))
        return code

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    write_run_log(json.dumps({"event": "start", "epochs": args.epochs, "datasets": len(datasets)}, sort_keys=True))
    results = []
    start = time.time()
    for path in datasets:
        result = run_dataset(path, epochs=args.epochs, force=args.force)
        results.append(result)
        write_run_log(json.dumps(result, sort_keys=True))
    elapsed = time.time() - start
    write_run_log(json.dumps({"event": "complete", "elapsed": elapsed, "datasets": len(results)}, sort_keys=True))
    print(f"\nComplete. Elapsed: {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
