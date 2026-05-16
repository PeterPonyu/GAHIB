"""Tests for reviewer-requested online graph-attention baseline adapters."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from experiments.external_baselines.online_graph_attention import (
    ONLINE_GRAPH_ATTENTION_METHODS,
    ONLINE_GRAPH_ATTENTION_SPECS,
    PYTORCH_GRAPH_ATTENTION_METHODS,
    sanitize_dataset_name,
    train_external_online_graph_attention,
    train_online_graph_attention,
    train_pytorch_graph_attention_style,
    write_external_data_tsv,
)
from experiments.run_sc_deeplearning_benchmark import external_status_row, get_complete_benchmark_datasets


class FakeAdata:
    def __init__(self):
        self.X = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float32)
        self.layers = {"counts": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)}
        self.obs = {"cell_type": np.array(["secret_a", "secret_b"], dtype=object)}
        self.obs_names = ["cell_a", "cell_b"]
        self.var_names = ["gene_1", "gene_2", "gene_3"]
        self.n_obs = 2


class SingleUseLabels:
    def __init__(self, values):
        self.values = values
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        if self.iterations > 1:
            raise AssertionError("labels were consumed after cluster-count inference")
        return iter(self.values)


def test_online_registry_has_provenance_for_reviewer_requested_style_methods():
    assert ONLINE_GRAPH_ATTENTION_METHODS == ("scGAC-style", "SCEA-style", "scVAG-style")
    assert PYTORCH_GRAPH_ATTENTION_METHODS == ONLINE_GRAPH_ATTENTION_METHODS
    for method in ONLINE_GRAPH_ATTENTION_METHODS:
        spec = ONLINE_GRAPH_ATTENTION_SPECS[method]
        assert spec.name == method
        assert spec.source_name in {"scGAC", "SCEA", "scVAG"}
        assert spec.repo_url.startswith("https://github.com/")
        assert len(spec.commit) == 40
        assert spec.license in {"GPL-3.0", "MIT"}
        assert spec.env_var.startswith("GAHIB_")
        assert "not exact" in spec.equivalence_note


def test_sanitize_dataset_name_keeps_external_script_safe():
    assert sanitize_dataset_name("GSE 123 / tumor.h5ad") == "GSE_123_tumor.h5ad"
    assert sanitize_dataset_name("***") == "dataset"


def test_write_external_data_tsv_uses_hvg_filtered_counts_with_gene_rows(tmp_path):
    output = write_external_data_tsv(FakeAdata(), tmp_path / "data" / "toy" / "data.tsv")
    frame = pd.read_csv(output, sep="\t", index_col=0)
    assert list(frame.index) == ["gene_1", "gene_2", "gene_3"]
    assert list(frame.columns) == ["cell_a", "cell_b"]
    assert frame.to_numpy().tolist() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_write_external_data_tsv_does_not_serialize_obs_labels(tmp_path):
    output = write_external_data_tsv(FakeAdata(), tmp_path / "data" / "toy" / "data.tsv")
    raw = output.read_text()
    assert "secret_a" not in raw
    assert "secret_b" not in raw
    assert "cell_type" not in raw


def test_external_checkout_adapter_logs_missing_checkout_without_metrics(monkeypatch):
    for key in [
        "GAHIB_SCGAC_DIR",
        "GAHIB_SCEA_DIR",
        "GAHIB_SCVAG_DIR",
        "GAHIB_ONLINE_BASELINE_ROOT",
        "GAHIB_ONLINE_BASELINE_PYTHON",
    ]:
        monkeypatch.delenv(key, raising=False)
    result = train_external_online_graph_attention(
        "scGAC",
        FakeAdata(),
        ["cluster_1", "cluster_2"],
        dataset_name="toy",
        epochs=1,
        env={},
    )
    assert result.status == "not_runnable"
    assert result.latent is None
    assert result.method == "scGAC-style"
    assert "external checkout missing" in result.reason


def test_resume_detection_requires_new_online_baseline_rows(tmp_path):
    tables = tmp_path / "tables"
    tables.mkdir()
    pd.DataFrame({"method": ["scVI", "GAHIB"], "ARI": [0.1, 0.2]}).to_csv(
        tables / "scdeep_old_df.csv", index=False
    )
    required = ["scVI", "scGAC-style", "SCEA-style", "scVAG-style", "GAHIB"]
    pd.DataFrame({"method": required, "ARI": [0.1, None, None, None, 0.2]}).to_csv(
        tables / "scdeep_new_df.csv", index=False
    )
    assert get_complete_benchmark_datasets(str(tables), "scdeep", required) == {"new"}


def test_all_pytorch_style_baselines_run_on_tiny_fixture():
    for method in PYTORCH_GRAPH_ATTENTION_METHODS:
        result = train_online_graph_attention(
            method,
            FakeAdata(),
            ["cluster_1", "cluster_2"],
            dataset_name="toy",
            epochs=1,
        )
        assert result.status == "ok"
        assert result.method == method
        assert result.latent is not None
        assert result.latent.shape == (2, 10)
        assert np.isfinite(result.latent).all()
        assert "not an exact upstream execution" in result.reason


def test_external_status_row_records_style_source_provenance():
    result = train_online_graph_attention(
        "scGAC-style",
        FakeAdata(),
        ["cluster_1", "cluster_2"],
        dataset_name="toy",
        epochs=1,
    )
    row = external_status_row("toy", result)
    assert row["method"] == "scGAC-style"
    assert row["repo_url"] == ONLINE_GRAPH_ATTENTION_SPECS["scGAC-style"].repo_url
    assert row["commit"] == ONLINE_GRAPH_ATTENTION_SPECS["scGAC-style"].commit
    assert row["license"] == "GPL-3.0"
    assert "Cheng" in row["source_note"]


def test_labels_are_not_used_as_supervised_targets():
    labels_a = ["celltype_a", "celltype_b"]
    labels_b = ["swapped_name_b", "swapped_name_a"]
    for method in PYTORCH_GRAPH_ATTENTION_METHODS:
        result_a = train_online_graph_attention(method, FakeAdata(), labels_a, dataset_name="toy", epochs=1)
        result_b = train_online_graph_attention(method, FakeAdata(), labels_b, dataset_name="toy", epochs=1)
        assert result_a.latent is not None
        assert result_b.latent is not None
        np.testing.assert_allclose(result_a.latent, result_b.latent, rtol=0.0, atol=0.0)

    source = inspect.getsource(train_pytorch_graph_attention_style)
    assert "cross_entropy" not in source
    assert "labels" not in source.split("n_clusters = _cluster_count(labels)", maxsplit=1)[1]


def test_label_values_are_consumed_once_and_never_serialized():
    for method in PYTORCH_GRAPH_ATTENTION_METHODS:
        labels = SingleUseLabels(["secret_celltype_a", "secret_celltype_b"])
        result = train_online_graph_attention(method, FakeAdata(), labels, dataset_name="toy", epochs=1)
        assert labels.iterations == 1
        exported_fields = " ".join(
            [result.reason, result.command, result.checkout, *result.output_files]
        )
        assert "secret_celltype_a" not in exported_fields
        assert "secret_celltype_b" not in exported_fields
