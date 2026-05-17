"""Tests for the resumable missing graph-attention-style baseline runner."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments.external_baselines.online_graph_attention import ExternalBaselineResult
from experiments import run_missing_graph_attention_baselines as runner


def test_missing_style_methods_detects_only_absent_style_rows():
    frame = pd.DataFrame({"method": ["scVI", "scGAC-style", "GAHIB"]})

    assert runner.missing_style_methods(frame) == ["SCEA-style", "scVAG-style"]
    assert runner.missing_style_methods(frame, force=True) == [
        "scGAC-style",
        "SCEA-style",
        "scVAG-style",
    ]


def test_merge_rows_preserves_existing_methods_and_places_style_rows_before_gahib():
    existing = pd.DataFrame(
        {
            "method": ["scVI", "GAHIB"],
            "ARI": [0.1, 0.9],
        }
    )
    merged = runner.merge_rows(
        existing,
        [
            {"method": "scGAC-style", "ARI": 0.2},
            {"method": "SCEA-style", "ARI": 0.3},
            {"method": "scVAG-style", "ARI": 0.4},
        ],
        force=False,
    )

    assert merged["method"].tolist() == [
        "scVI",
        "scGAC-style",
        "SCEA-style",
        "scVAG-style",
        "GAHIB",
    ]
    assert merged.loc[merged["method"] == "scVI", "ARI"].item() == 0.1
    assert merged.loc[merged["method"] == "GAHIB", "ARI"].item() == 0.9


def test_merge_rows_force_replaces_only_style_rows():
    existing = pd.DataFrame(
        {
            "method": ["scVI", "scGAC-style", "SCEA-style", "GAHIB"],
            "ARI": [0.1, -1.0, -2.0, 0.9],
        }
    )
    merged = runner.merge_rows(
        existing,
        [{"method": "scGAC-style", "ARI": 0.2}],
        force=True,
    )

    assert merged["method"].tolist() == ["scVI", "scGAC-style", "GAHIB"]
    assert merged.loc[merged["method"] == "scGAC-style", "ARI"].item() == 0.2
    assert merged.loc[merged["method"] == "scVI", "ARI"].item() == 0.1
    assert merged.loc[merged["method"] == "GAHIB", "ARI"].item() == 0.9


def test_status_row_records_public_safe_training_config():
    training_config = runner.graph_attention_style_training_config_json("scGAC-style", epochs=200)
    result = ExternalBaselineResult(
        method="scGAC-style",
        status="ok",
        reason="completed transparent PyTorch scGAC-style route",
        latent=np.zeros((2, 10), dtype=np.float32),
        elapsed=1.5,
        training_config=training_config,
    )

    row = runner.external_status_row("toy", result)
    assert row["dataset"] == "toy"
    assert row["method"] == "scGAC-style"
    assert row["license"] == "GPL-3.0"
    assert "github.com" in row["repo_url"]
    config = json.loads(row["training_config"])
    assert config["epochs"] == 200
    assert config["latent_dim"] == 10
    assert config["k_neighbors"] == 15
    assert config["label_usage"] == "cluster_count_only"


def test_merge_status_replaces_only_rerun_style_methods():
    existing = pd.DataFrame(
        {
            "method": ["scGAC-style", "SCEA-style", "old_external"],
            "status": ["old", "old", "ok"],
        }
    )
    merged = runner.merge_status(
        existing,
        [{"method": "scGAC-style", "status": "new"}],
        force=False,
    )

    assert merged.set_index("method").loc["scGAC-style", "status"] == "new"
    assert merged.set_index("method").loc["SCEA-style", "status"] == "old"
    assert merged.set_index("method").loc["old_external", "status"] == "ok"
