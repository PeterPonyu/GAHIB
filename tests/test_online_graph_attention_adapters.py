"""Tests for reviewer-requested online graph-attention baseline adapters."""

from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.external_baselines.online_graph_attention import (
    ONLINE_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_SPECS,
    sanitize_dataset_name,
    train_online_graph_attention,
    write_external_data_tsv,
)
from experiments.run_sc_deeplearning_benchmark import get_complete_benchmark_datasets


class FakeAdata:
    def __init__(self):
        self.X = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        self.layers = {"counts": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)}
        self.obs_names = ["cell_a", "cell_b"]
        self.var_names = ["gene_1", "gene_2", "gene_3"]
        self.n_obs = 2


def test_online_registry_has_provenance_for_reviewer_requested_methods():
    assert ONLINE_GRAPH_ATTENTION_METHODS == ("scGAC", "SCEA", "scVAG")
    for method in ONLINE_GRAPH_ATTENTION_METHODS:
        spec = ONLINE_GRAPH_ATTENTION_SPECS[method]
        assert spec.repo_url.startswith("https://github.com/")
        assert len(spec.commit) == 40
        assert spec.license in {"GPL-3.0", "MIT"}
        assert spec.env_var.startswith("GAHIB_")


def test_sanitize_dataset_name_keeps_external_script_safe():
    assert sanitize_dataset_name("GSE 123 / tumor.h5ad") == "GSE_123_tumor.h5ad"
    assert sanitize_dataset_name("***") == "dataset"


def test_write_external_data_tsv_uses_hvg_filtered_counts_with_gene_rows(tmp_path):
    output = write_external_data_tsv(FakeAdata(), tmp_path / "data" / "toy" / "data.tsv")
    frame = pd.read_csv(output, sep="\t", index_col=0)
    assert list(frame.index) == ["gene_1", "gene_2", "gene_3"]
    assert list(frame.columns) == ["cell_a", "cell_b"]
    assert frame.to_numpy().tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_missing_checkout_is_logged_not_runnable_without_metrics(monkeypatch):
    for key in [
        "GAHIB_SCGAC_DIR",
        "GAHIB_SCEA_DIR",
        "GAHIB_SCVAG_DIR",
        "GAHIB_ONLINE_BASELINE_ROOT",
        "GAHIB_ONLINE_BASELINE_PYTHON",
    ]:
        monkeypatch.delenv(key, raising=False)
    result = train_online_graph_attention(
        "scGAC",
        FakeAdata(),
        ["cluster_1", "cluster_2"],
        dataset_name="toy",
        epochs=1,
        env={},
    )
    assert result.status == "not_runnable"
    assert result.latent is None
    assert "external checkout missing" in result.reason


def test_resume_detection_requires_new_online_baseline_rows(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame({"method": ["scVI", "GAHIB"], "ARI": [0.1, 0.2]}).to_csv(
        tables / "scdeep_old_df.csv", index=False
    )
    required = ["scVI", "scGAC", "SCEA", "scVAG", "GAHIB"]
    pd.DataFrame({"method": required, "ARI": [0.1, None, None, None, 0.2]}).to_csv(
        tables / "scdeep_new_df.csv", index=False
    )
    assert get_complete_benchmark_datasets(str(tables), "scdeep", required) == {"new"}
