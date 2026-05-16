"""Online graph-attention baseline adapters for the GAHIB benchmark.

The reviewer-requested scGAC, SCEA, and scVAG implementations are external
research repositories.  Their licenses/dependency pins make vendoring unsafe for
this MIT public repository, so this module provides thin, provenance-recording
subprocess adapters.  They run only when the caller supplies a local checkout;
otherwise they return a structured ``not_runnable`` result that the benchmark
can log without fabricating metrics.
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


@dataclass(frozen=True)
class OnlineBaselineSpec:
    """Metadata and command policy for one external online baseline."""

    name: str
    repo_dir: str
    repo_url: str
    commit: str
    license: str
    env_var: str
    scripts: tuple[str, ...]
    requires_old_keras: bool
    source_note: str


@dataclass(frozen=True)
class ExternalBaselineResult:
    """Outcome of one external baseline run."""

    method: str
    status: str
    reason: str
    latent: Optional[np.ndarray] = None
    command: str = ""
    checkout: str = ""
    elapsed: float = 0.0
    output_files: tuple[str, ...] = ()


ONLINE_GRAPH_ATTENTION_SPECS: dict[str, OnlineBaselineSpec] = {
    "scGAC": OnlineBaselineSpec(
        name="scGAC",
        repo_dir="scGAC",
        repo_url="https://github.com/Joye9285/scGAC.git",
        commit="893909a7e46e0ed2e52fd1800f2c1606c4cf447a",
        license="GPL-3.0",
        env_var="GAHIB_SCGAC_DIR",
        scripts=("scGAC.py",),
        requires_old_keras=True,
        source_note="Cheng & Ma, Bioinformatics 2022; DOI 10.1093/bioinformatics/btac099.",
    ),
    "SCEA": OnlineBaselineSpec(
        name="SCEA",
        repo_dir="SCEA",
        repo_url="https://github.com/SAkbari93/SCEA.git",
        commit="a13ef014f25843a53ba720b969d4b57be06d9416",
        license="GPL-3.0",
        env_var="GAHIB_SCEA_DIR",
        scripts=("SCEA.py",),
        requires_old_keras=True,
        source_note="Akbari Rokn Abadi et al., BMC Genomics 2023; DOI 10.1186/s12864-023-09344-y.",
    ),
    "scVAG": OnlineBaselineSpec(
        name="scVAG",
        repo_dir="scVAG",
        repo_url="https://github.com/pourialaghayee/scVAG.git",
        commit="a0eb8151849e251f11f6859a1e4867a6f5367360",
        license="MIT",
        env_var="GAHIB_SCVAG_DIR",
        scripts=("scVAG_VAE.py", "scGAC.py"),
        requires_old_keras=True,
        source_note="Laghaee et al., Heliyon 2024; DOI 10.1016/j.heliyon.2024.e40732.",
    ),
}

ONLINE_GRAPH_ATTENTION_METHODS = tuple(ONLINE_GRAPH_ATTENTION_SPECS)


class ExternalBaselineError(RuntimeError):
    """Raised for adapter-internal validation failures."""


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
    """Return a shell/path-safe dataset id accepted by the external scripts."""

    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._-")
    return safe or "dataset"


def write_external_data_tsv(adata, destination: Path) -> Path:
    """Write GAHIB-preprocessed data in the external repositories' format.

    The normal GAHIB policy still controls dataset discovery, HVG selection,
    max-cell cap, and seed.  These source implementations explicitly recommend
    raw-count input, so the adapter exports the already-HVG-filtered
    ``layers['counts']`` matrix when present, falling back to ``adata.X`` only
    for minimal synthetic tests.
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
    """Verify the legacy Keras import surface used by scGAC/SCEA/scVAG."""

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


def _cluster_count(labels: Iterable[object]) -> int:
    labels_arr = np.asarray(list(labels)).astype(str)
    count = len(np.unique(labels_arr))
    if count < 2:
        raise ExternalBaselineError("external graph-attention baselines require at least two clusters")
    return count


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
    """Run one online graph-attention baseline when a compatible checkout exists.

    Returns ``status='ok'`` with ``latent`` on success, otherwise
    ``status='not_runnable'`` with a precise reason.  Missing checkouts or
    incompatible TensorFlow/Keras stacks are expected outcomes, not exceptions,
    because the benchmark must not fabricate metrics.
    """

    if method not in ONLINE_GRAPH_ATTENTION_SPECS:
        raise KeyError(f"unknown online graph-attention baseline: {method}")
    spec = ONLINE_GRAPH_ATTENTION_SPECS[method]
    env = os.environ if env is None else env
    checkout = resolve_checkout(spec, env)
    if checkout is None:
        return ExternalBaselineResult(
            method=method,
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
                method=method,
                status="not_runnable",
                reason=details,
                checkout=str(checkout),
            )

    dataset_id = sanitize_dataset_name(dataset_name)
    n_clusters = _cluster_count(labels)
    start = time.time()
    commands: list[list[str]] = []
    with tempfile.TemporaryDirectory(prefix=f"gahib_{method.lower()}_") as tmp:
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
                    method=method,
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
                method=method,
                status="not_runnable",
                reason=str(exc),
                command=" && ".join(" ".join(cmd) for cmd in commands),
                checkout=str(checkout),
                elapsed=time.time() - start,
            )
        elapsed = time.time() - start
        return ExternalBaselineResult(
            method=method,
            status="ok",
            reason="completed via external checkout under GAHIB preprocessing policy",
            latent=latent,
            command=" && ".join(" ".join(cmd) for cmd in commands),
            checkout=str(checkout),
            elapsed=elapsed,
            output_files=(str(hidden_path),),
        )
