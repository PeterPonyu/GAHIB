"""PyTorch graph-attention style baselines for reviewer-requested methods.

The reviewer-requested scGAC, SCEA, and scVAG implementations are external
research repositories. Their original code and environment pins are not vendored
into this MIT public repository. This module therefore exposes transparent
``*-style`` baselines implemented from scratch in PyTorch. The routes preserve the
shared GAHIB preprocessing/evaluation policy and record provenance for the source
method that inspired each variant, but they do not claim exact parity with the
external implementations.

For reproducibility audits, the legacy checkout adapter remains available through
``train_external_online_graph_attention``. The benchmark-facing
``train_online_graph_attention`` function intentionally uses the safe PyTorch
style route so missing legacy TensorFlow/Keras stacks do not fabricate or block
metrics.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OnlineBaselineSpec:
    """Metadata and command policy for one external online baseline source."""

    name: str
    source_name: str
    repo_dir: str
    repo_url: str
    commit: str
    license: str
    env_var: str
    scripts: tuple[str, ...]
    requires_old_keras: bool
    source_note: str
    equivalence_note: str


@dataclass(frozen=True)
class TorchStyleConfig:
    """Small implementation choices for one transparent PyTorch style route."""

    hidden_dim: int
    latent_dim: int = 10
    k_neighbors: int = 15
    dropout: float = 0.1
    graph_layers: int = 2
    graph_smoothness_weight: float = 0.02
    cluster_balance_weight: float = 0.0
    kl_weight: float = 0.0
    variational: bool = False
    input_dropout: float = 0.0


@dataclass(frozen=True)
class ExternalBaselineResult:
    """Outcome of one online graph-attention baseline run."""

    method: str
    status: str
    reason: str
    latent: Optional[np.ndarray] = None
    command: str = ""
    checkout: str = ""
    elapsed: float = 0.0
    output_files: tuple[str, ...] = ()


ONLINE_GRAPH_ATTENTION_SPECS: dict[str, OnlineBaselineSpec] = {
    "scGAC-style": OnlineBaselineSpec(
        name="scGAC-style",
        source_name="scGAC",
        repo_dir="scGAC",
        repo_url="https://github.com/Joye9285/scGAC.git",
        commit="893909a7e46e0ed2e52fd1800f2c1606c4cf447a",
        license="GPL-3.0",
        env_var="GAHIB_SCGAC_DIR",
        scripts=("scGAC.py",),
        requires_old_keras=True,
        source_note="Cheng & Ma, Bioinformatics 2022; DOI 10.1093/bioinformatics/btac099.",
        equivalence_note=(
            "PyTorch graph-attention autoencoder with unsupervised cluster-balance regularization; "
            "not exact scGAC parity."
        ),
    ),
    "SCEA-style": OnlineBaselineSpec(
        name="SCEA-style",
        source_name="SCEA",
        repo_dir="SCEA",
        repo_url="https://github.com/SAkbari93/SCEA.git",
        commit="a13ef014f25843a53ba720b969d4b57be06d9416",
        license="GPL-3.0",
        env_var="GAHIB_SCEA_DIR",
        scripts=("SCEA.py",),
        requires_old_keras=True,
        source_note="Akbari Rokn Abadi et al., BMC Genomics 2023; DOI 10.1186/s12864-023-09344-y.",
        equivalence_note=(
            "PyTorch graph-attention denoising autoencoder style route; not exact SCEA parity."
        ),
    ),
    "scVAG-style": OnlineBaselineSpec(
        name="scVAG-style",
        source_name="scVAG",
        repo_dir="scVAG",
        repo_url="https://github.com/pourialaghayee/scVAG.git",
        commit="a0eb8151849e251f11f6859a1e4867a6f5367360",
        license="MIT",
        env_var="GAHIB_SCVAG_DIR",
        scripts=("scVAG_VAE.py", "scGAC.py"),
        requires_old_keras=True,
        source_note="Laghaee et al., Heliyon 2024; DOI 10.1016/j.heliyon.2024.e40732.",
        equivalence_note=(
            "PyTorch variational graph-attention autoencoder style route; not exact scVAG parity."
        ),
    ),
}

ONLINE_GRAPH_ATTENTION_METHODS = tuple(ONLINE_GRAPH_ATTENTION_SPECS)

_METHOD_ALIASES = {
    "scGAC": "scGAC-style",
    "SCEA": "SCEA-style",
    "scVAG": "scVAG-style",
}

_TORCH_STYLE_CONFIGS: dict[str, TorchStyleConfig] = {
    "scGAC-style": TorchStyleConfig(
        hidden_dim=96,
        graph_smoothness_weight=0.03,
        cluster_balance_weight=0.01,
        dropout=0.08,
    ),
    "SCEA-style": TorchStyleConfig(
        hidden_dim=80,
        graph_smoothness_weight=0.05,
        input_dropout=0.08,
        dropout=0.12,
    ),
    "scVAG-style": TorchStyleConfig(
        hidden_dim=96,
        graph_smoothness_weight=0.02,
        kl_weight=1e-3,
        variational=True,
        dropout=0.08,
    ),
}


class ExternalBaselineError(RuntimeError):
    """Raised for adapter-internal validation failures."""


def resolve_online_graph_attention_method(method: str) -> str:
    """Return the transparent benchmark method key, accepting legacy aliases."""

    if method in ONLINE_GRAPH_ATTENTION_SPECS:
        return method
    if method in _METHOD_ALIASES:
        return _METHOD_ALIASES[method]
    raise KeyError(f"unknown online graph-attention baseline: {method}")


def _dense(matrix: object) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix, dtype=np.float32)


def _as_names(values: object, fallback_prefix: str, n: int) -> list[str]:
    if values is None:
        return [f"{fallback_prefix}_{i}" for i in range(n)]
    names = [str(v) for v in list(values)]
    if len(names) != n:
        return [f"{fallback_prefix}_{i}" for i in range(n)]
    return names


def sanitize_dataset_name(name: str) -> str:
    """Return a shell/path-safe dataset id accepted by external scripts/status rows."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return safe or "dataset"


def write_external_data_tsv(adata, destination: Path) -> Path:
    """Write GAHIB-preprocessed data in the external repositories' TSV format.

    The normal GAHIB policy still controls dataset discovery, HVG selection,
    max-cell cap, and seed. The source implementations recommend raw-count
    input, so the adapter exports the already-HVG-filtered ``layers['counts']``
    matrix when present, falling back to ``adata.X`` only for minimal synthetic
    tests.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    matrix = adata.layers["counts"] if hasattr(adata, "layers") and "counts" in adata.layers else adata.X
    x = _dense(matrix)
    if x.ndim != 2:
        raise ExternalBaselineError(f"expected 2-D matrix, got shape {x.shape}")
    cell_names = _as_names(getattr(adata, "obs_names", None), "cell", x.shape[0])
    gene_names = _as_names(getattr(adata, "var_names", None), "gene", x.shape[1])
    # External scripts read genes as rows and cells as columns.
    pd.DataFrame(x.T, index=gene_names, columns=cell_names).to_csv(destination, sep="\t")
    return destination


def _cluster_count(labels: Iterable[object]) -> int:
    labels_arr = np.asarray(list(labels)).astype(str)
    count = len(np.unique(labels_arr))
    if count < 2:
        raise ExternalBaselineError("graph-attention style baselines require at least two label groups for k selection")
    return count


def _torch_feature_matrix(adata) -> np.ndarray:
    matrix = adata.layers["counts"] if hasattr(adata, "layers") and "counts" in adata.layers else adata.X
    x = _dense(matrix)
    if x.ndim != 2:
        raise ExternalBaselineError(f"expected 2-D matrix, got shape {x.shape}")
    if x.shape[0] < 2:
        raise ExternalBaselineError("graph-attention style baselines require at least two cells")
    if x.shape[1] < 1:
        raise ExternalBaselineError("graph-attention style baselines require at least one feature")
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    # Raw-count layers are non-negative; normalize/log for stable full-batch torch training.
    if np.min(x) >= 0.0:
        library = x.sum(axis=1, keepdims=True)
        positive_library = library[library > 0]
        target_library = float(np.median(positive_library)) if positive_library.size else 1.0
        scale = np.divide(target_library, library, out=np.ones_like(library), where=library > 0)
        x = np.log1p(x * scale).astype(np.float32, copy=False)

    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    x = (x - mean) / np.maximum(std, 1e-6)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _knn_edge_index(x: np.ndarray, k_neighbors: int) -> torch.Tensor:
    n_obs = x.shape[0]
    n_neighbors = min(max(1, k_neighbors), n_obs - 1) + 1
    neighbors = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    indices = neighbors.fit(x).kneighbors(return_distance=False)
    edges: set[tuple[int, int]] = set()
    for dst, row in enumerate(indices):
        edges.add((dst, dst))
        for src in row:
            src_i = int(src)
            if src_i == dst:
                continue
            edges.add((src_i, dst))
            edges.add((dst, src_i))
    src_nodes, dst_nodes = zip(*sorted(edges))
    return torch.tensor([src_nodes, dst_nodes], dtype=torch.long)


class _GraphAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)
        self.attn_src = nn.Parameter(torch.empty(output_dim))
        self.attn_dst = nn.Parameter(torch.empty(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.attn_src.view(1, -1))
        nn.init.xavier_uniform_(self.attn_dst.view(1, -1))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        h = self.linear(x)
        logits = (h.index_select(0, src) * self.attn_src).sum(dim=1)
        logits = logits + (h.index_select(0, dst) * self.attn_dst).sum(dim=1)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        # Segment softmax by destination node using only PyTorch primitives.
        weights = torch.exp(torch.clamp(logits, min=-20.0, max=20.0))
        denom = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
        denom.index_add_(0, dst, weights)
        alpha = weights / denom.index_select(0, dst).clamp_min(1e-12)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = torch.zeros_like(h)
        messages = alpha.unsqueeze(1) * h.index_select(0, src)
        out.index_add_(0, dst, messages)
        return out + self.bias


class _GraphAttentionAutoencoder(nn.Module):
    def __init__(self, input_dim: int, n_clusters: int, config: TorchStyleConfig):
        super().__init__()
        layers: list[_GraphAttentionLayer] = []
        current_dim = input_dim
        for _ in range(max(1, config.graph_layers - 1)):
            layers.append(_GraphAttentionLayer(current_dim, config.hidden_dim, config.dropout))
            current_dim = config.hidden_dim
        self.encoder_layers = nn.ModuleList(layers)
        self.variational = config.variational
        if self.variational:
            self.mu_layer = _GraphAttentionLayer(current_dim, config.latent_dim, config.dropout)
            self.logvar_layer = _GraphAttentionLayer(current_dim, config.latent_dim, config.dropout)
        else:
            self.latent_layer = _GraphAttentionLayer(current_dim, config.latent_dim, config.dropout)
        self.decoder = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.ELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, input_dim),
        )
        self.cluster_head = nn.Linear(config.latent_dim, n_clusters) if config.cluster_balance_weight > 0 else None
        self.config = config

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        h = x
        if self.config.input_dropout > 0:
            h = F.dropout(h, p=self.config.input_dropout, training=self.training)
        for layer in self.encoder_layers:
            h = F.elu(layer(h, edge_index))
            h = F.dropout(h, p=self.config.dropout, training=self.training)
        if self.variational:
            mu = self.mu_layer(h, edge_index)
            logvar = torch.clamp(self.logvar_layer(h, edge_index), min=-8.0, max=8.0)
            if self.training:
                eps = torch.randn_like(mu)
                z = mu + eps * torch.exp(0.5 * logvar)
            else:
                z = mu
            return z, mu, logvar
        z = self.latent_layer(h, edge_index)
        return z, z, None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        z, mu, logvar = self.encode(x, edge_index)
        recon = self.decoder(z)
        return recon, z, mu, logvar


def _graph_smoothness(z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    src, dst = edge_index
    return (z.index_select(0, src) - z.index_select(0, dst)).pow(2).mean()


def _cluster_balance_loss(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1).clamp_min(1e-8)
    per_cell_entropy = -(probabilities * probabilities.log()).sum(dim=1).mean()
    average_probability = probabilities.mean(dim=0).clamp_min(1e-8)
    batch_entropy = -(average_probability * average_probability.log()).sum()
    return per_cell_entropy - batch_entropy


def train_pytorch_graph_attention_style(
    method: str,
    adata,
    labels: Iterable[object],
    *,
    dataset_name: str,
    epochs: int,
    seed: int = 42,
    device: Optional[str | torch.device] = None,
) -> ExternalBaselineResult:
    """Train one transparent PyTorch ``*-style`` graph-attention baseline.

    The implementation uses only project dependencies and does not import or
    copy external repositories. Labels are used only to infer the requested
    cluster count for unsupervised style losses, matching the previous adapter's
    cluster-count contract without supervised leakage.
    """

    method_key = resolve_online_graph_attention_method(method)
    spec = ONLINE_GRAPH_ATTENTION_SPECS[method_key]
    config = _TORCH_STYLE_CONFIGS[method_key]
    n_clusters = _cluster_count(labels)
    x_np = _torch_feature_matrix(adata)
    start = time.time()

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    x = torch.as_tensor(x_np, dtype=torch.float32, device=resolved_device)
    edge_index = _knn_edge_index(x_np, config.k_neighbors).to(resolved_device)
    model = _GraphAttentionAutoencoder(x.shape[1], n_clusters=n_clusters, config=config).to(resolved_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    max_epochs = max(1, int(epochs))

    model.train()
    for _ in range(max_epochs):
        optimizer.zero_grad(set_to_none=True)
        recon, z, mu, logvar = model(x, edge_index)
        loss = F.mse_loss(recon, x)
        if config.graph_smoothness_weight:
            loss = loss + config.graph_smoothness_weight * _graph_smoothness(z, edge_index)
        if config.cluster_balance_weight and model.cluster_head is not None:
            loss = loss + config.cluster_balance_weight * _cluster_balance_loss(model.cluster_head(z))
        if config.kl_weight and logvar is not None:
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = loss + config.kl_weight * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

    model.eval()
    with torch.no_grad():
        _, latent_t, _, _ = model(x, edge_index)
    latent = latent_t.detach().cpu().numpy().astype(np.float32, copy=False)
    if latent.shape[0] != x_np.shape[0]:
        raise ExternalBaselineError(f"latent row mismatch: expected {x_np.shape[0]}, got {latent.shape[0]}")
    if not np.all(np.isfinite(latent)):
        raise ExternalBaselineError(f"{method_key} latent contains NaN/Inf")

    elapsed = time.time() - start
    command = (
        f"pytorch_graph_attention_style(method={method_key}, dataset={sanitize_dataset_name(dataset_name)}, "
        f"epochs={max_epochs}, latent_dim={config.latent_dim}, k_neighbors={config.k_neighbors}, seed={seed})"
    )
    return ExternalBaselineResult(
        method=method_key,
        status="ok",
        reason=(
            f"completed transparent PyTorch {method_key} route inspired by {spec.source_name}; "
            f"{spec.equivalence_note} Source: {spec.repo_url}@{spec.commit}"
        ),
        latent=latent,
        command=command,
        elapsed=elapsed,
    )


def train_online_graph_attention(
    method: str,
    adata,
    labels: Iterable[object],
    *,
    dataset_name: str,
    epochs: int,
    timeout: int = 7200,
    env: Optional[Mapping[str, str]] = None,
) -> ExternalBaselineResult:
    """Benchmark-facing route for reviewer-requested online graph baselines.

    ``timeout`` and ``env`` are accepted for API compatibility with the legacy
    checkout adapter. They are intentionally unused by this PyTorch route.
    """

    del timeout, env
    return train_pytorch_graph_attention_style(method, adata, labels, dataset_name=dataset_name, epochs=epochs)


def resolve_checkout(spec: OnlineBaselineSpec, env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    """Find a user-provided checkout without downloading or vendoring it."""

    env = os.environ if env is None else env
    candidates: list[Path] = []
    if env.get(spec.env_var):
        candidates.append(Path(env[spec.env_var]).expanduser())
    if env.get("GAHIB_ONLINE_BASELINE_ROOT"):
        candidates.append(Path(env["GAHIB_ONLINE_BASELINE_ROOT"]).expanduser() / spec.repo_dir)
    for candidate in candidates:
        if candidate.is_dir() and all((candidate / script).is_file() for script in spec.scripts):
            return candidate.resolve()
    return None


def _python_for_external(env: Optional[Mapping[str, str]] = None) -> str:
    env = os.environ if env is None else env
    return env.get("GAHIB_ONLINE_BASELINE_PYTHON") or sys.executable


def _check_old_keras_environment(python_executable: str, timeout: int = 30) -> tuple[bool, str]:
    """Verify the legacy Keras import surface used by source checkouts."""

    code = (
        "import sys\n"
        "try:\n"
        "    import keras, tensorflow\n"
        "    from keras.engine.topology import Layer, InputSpec\n"
        "    from keras.optimizers import Adam\n"
        "except Exception as exc:\n"
        "    print(type(exc).__name__ + ': ' + str(exc))\n"
        "    raise SystemExit(1)\n"
        "print('python=' + sys.version.split()[0])\n"
        "print('keras=' + getattr(keras, '__version__', 'unknown'))\n"
        "print('tensorflow=' + getattr(tensorflow, '__version__', 'unknown'))\n"
    )
    try:
        proc = subprocess.run(
            [python_executable, "-c", code],
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"legacy Keras environment check failed: {type(exc).__name__}: {exc}"
    details = (proc.stdout + proc.stderr).strip()
    if proc.returncode != 0:
        return False, f"legacy Keras/TensorFlow 1.x imports unavailable under {python_executable}: {details}"
    return True, details


def _copy_checkout(source: Path, target: Path) -> None:
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", "logs", "result", "data")
    shutil.copytree(source, target, ignore=ignore)


def _run_command(cmd: list[str], cwd: Path, timeout: int, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout, env=dict(env), check=False)


def _hidden_output_path(workdir: Path, dataset_id: str) -> Path:
    return workdir / "result" / f"hidden_{dataset_id}.tsv"


def _read_hidden(path: Path, expected_rows: int) -> np.ndarray:
    if not path.is_file():
        raise ExternalBaselineError(f"expected hidden output not found: {path}")
    frame = pd.read_csv(path, sep="\t", index_col=0)
    latent = frame.to_numpy(dtype=np.float32)
    if latent.shape[0] != expected_rows:
        raise ExternalBaselineError(
            f"hidden output row mismatch: expected {expected_rows}, got {latent.shape[0]} from {path}"
        )
    if not np.all(np.isfinite(latent)):
        raise ExternalBaselineError(f"hidden output contains NaN/Inf: {path}")
    return latent


def train_external_online_graph_attention(
    method: str,
    adata,
    labels: Iterable[object],
    *,
    dataset_name: str,
    epochs: int,
    timeout: int = 7200,
    env: Optional[Mapping[str, str]] = None,
) -> ExternalBaselineResult:
    """Run one source checkout when a compatible external environment exists.

    This helper is retained for audit/reproduction. The public benchmark route
    uses ``train_online_graph_attention`` above because exact legacy execution
    depends on non-project checkouts and old TensorFlow/Keras packages.
    """

    method_key = resolve_online_graph_attention_method(method)
    spec = ONLINE_GRAPH_ATTENTION_SPECS[method_key]
    env = os.environ if env is None else env
    checkout = resolve_checkout(spec, env)
    if checkout is None:
        return ExternalBaselineResult(
            method=method_key,
            status="not_runnable",
            reason=(
                f"external checkout missing; set {spec.env_var} or "
                f"GAHIB_ONLINE_BASELINE_ROOT/{spec.repo_dir}; source={spec.repo_url}; commit={spec.commit}"
            ),
        )

    python_executable = _python_for_external(env)
    if spec.requires_old_keras:
        ok, details = _check_old_keras_environment(python_executable)
        if not ok:
            return ExternalBaselineResult(
                method=method_key,
                status="not_runnable",
                reason=details,
                checkout=str(checkout),
            )

    dataset_id = sanitize_dataset_name(dataset_name)
    n_clusters = _cluster_count(labels)
    start = time.time()
    commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(prefix=f"gahib_{spec.source_name.lower()}_") as tmp:
        workdir = Path(tmp) / spec.repo_dir
        _copy_checkout(checkout, workdir)
        write_external_data_tsv(adata, workdir / "data" / dataset_id / "data.tsv")
        for script in spec.scripts:
            if script == "scVAG_VAE.py":
                commands.append([python_executable, script, dataset_id])
            else:
                commands.append(
                    [
                        python_executable,
                        script,
                        dataset_id,
                        str(n_clusters),
                        "--pre_epochs",
                        str(epochs),
                        "--epochs",
                        str(epochs),
                    ]
                )

        outputs: list[str] = []
        for cmd in commands:
            proc = _run_command(cmd, workdir, timeout, env)
            outputs.append("$ " + " ".join(cmd) + "\n" + proc.stdout + proc.stderr)
            if proc.returncode != 0:
                elapsed = time.time() - start
                return ExternalBaselineResult(
                    method=method_key,
                    status="not_runnable",
                    reason=f"external command failed with exit {proc.returncode}: {outputs[-1][-2000:]}",
                    command=" && ".join(" ".join(cmd) for cmd in commands),
                    checkout=str(checkout),
                    elapsed=elapsed,
                )
        try:
            hidden_path = _hidden_output_path(workdir, dataset_id)
            latent = _read_hidden(hidden_path, expected_rows=int(getattr(adata, "n_obs", _dense(adata.X).shape[0])))
        except ExternalBaselineError as exc:
            return ExternalBaselineResult(
                method=method_key,
                status="not_runnable",
                reason=str(exc),
                command=" && ".join(" ".join(cmd) for cmd in commands),
                checkout=str(checkout),
                elapsed=time.time() - start,
            )
        elapsed = time.time() - start
        return ExternalBaselineResult(
            method=method_key,
            status="ok",
            reason="completed via external checkout under GAHIB preprocessing policy",
            latent=latent,
            command=" && ".join(" ".join(cmd) for cmd in commands),
            checkout=str(checkout),
            elapsed=elapsed,
            output_files=(str(hidden_path),),
        )
