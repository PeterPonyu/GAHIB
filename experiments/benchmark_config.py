"""Benchmark dataset configuration and full-run safety guard helpers."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import os
from typing import Dict, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIRS_ENV = "GAHIB_DATASET_DIRS"
FULL_BENCHMARK_CONFIRM_ENV = "GAHIB_CONFIRM_FULL_BENCHMARK"
FULL_BENCHMARK_ALLOW_CPU_ENV = "GAHIB_ALLOW_CPU_FULL_BENCHMARK"

EXCLUDE = ['GSE120575_melanomaHmCancer', 'GSE225948_bloodMmStrokeDev']

SELECTED_DATASETS = [
    # ── Cancer (27) ──────────────────────────────────────────────────
    # CancerDatasets/
    'GSE123813_bccHmCancer',
    'GSE123813_sccHmCancer',
    'GSE123902_LungAdreHmCancer',
    'GSE132509_acutelymluekPBMCHmCancer',
    'GSE143423_lbm_CancerBrainHm',
    'GSE143423_tnbc_CancerBrainHm',
    'GSE148218_bmALLHmCancer',
    'GSE155109_bcECHmCancer',
    'GSE155109_bcStromaHmCancer',
    'GSE183904_GastricHmCancer',
    'GSE222002_TcellsHmCancer',
    'GSE222369_NKsLymphomaHmCancer',
    'GSE225600_breast_CancerHm',
    'GSE235787_bcellsALLHmCancer',
    'GSE262288_breastMetasisHmCancer',
    'GSE98638_TcellLiverHmCancer',
    # CancerDatasets2/
    'GSE117988_MCCPBMCCancer',
    'GSE117988_MCCTumorCancer',
    'GSE124310_MMHmCancer',
    'GSE138709_LiverCancer',
    'GSE149655_CAHmCancer',
    'GSE163558_stomachHmCancer',
    'GSE168181_BreastHmCancer',
    'GSE189357_lungAdreHmCancer',
    'GSE225857_liverColonMetasisHmCancer',
    'GSE228499_breastHmCancer',
    'GSE283205_hepatoblastomaCancer',
    # ── Development (26) ─────────────────────────────────────────────
    # DevelopmentDatasets/
    'GSE120505_bloodAged',
    'GSE148215_hESCHSPCD8Hm',
    'GSE165844_LSKMmBatch',
    'GSE167597_spineMm',
    'GSE192857_hESCHmTimes',
    'GSE226131_HSCMmAged',
    'GSE253355_bmNicheHm',
    'bm_GSE120446',
    'dentate',
    'endo',
    'hESC_GSE144024',
    'hemato',
    'ifnHSPC_GSE226824',
    'lung',
    'setty',
    # DevelopmentDatasets2/
    'GSE115571_LPSMmDev',
    'GSE130148_LungHmDev',
    'GSE142653pitHmDev',
    'GSE145929_ProgastinMmDev',
    'GSE145929_UrineMmDev',
    'GSE165784_RetinaHmDev',
    'GSE189070_astrocytesSCIMmDev',
    'GSE213740_ADHm',
    'GSE247719_PanSci_05_Muscle_adata',
    'GSE247719_PanSci_T_cell_adata',
    'GSE275119_TeethMmDev',
]

FULL_BENCHMARK_STEPS: Tuple[Tuple[str, str], ...] = (
    ("[1/11] Ablation Study (5 variants)", "experiments/run_ablation.py"),
    ("[2/11] SC Deep Learning Benchmark (8 methods)", "experiments/run_sc_deeplearning_benchmark.py"),
    ("[3/11] Classical DR Benchmark (6 methods)", "experiments/run_classical_benchmark.py"),
    ("[4/11] GM-VAE Benchmark (6 methods)", "experiments/run_gmvae_benchmark.py"),
    ("[5/11] Disentanglement Comparison (6 methods)", "experiments/run_disentanglement.py"),
    ("[6/11] Encoder Architecture Comparison (3 methods)", "experiments/run_encoder_comparison.py"),
    ("[7/11] Graph Convolution Sweep (6 methods)", "experiments/run_graph_conv_sweep.py"),
    ("[8/11] Hyperparameter Sensitivity (4 sweeps x 5 values)", "experiments/run_hyperparam_sensitivity.py"),
    ("[9/11] Latent Dimension Ablation (5 dimensions)", "experiments/run_latent_dim_ablation.py"),
    ("[10/11] Multi-Seed Robustness (5 seeds)", "experiments/run_seed_robustness.py"),
    ("[11/11] Computational Cost Analysis (3 methods + scaling)", "experiments/run_computational_cost.py"),
)


@dataclass(frozen=True)
class DatasetInventory:
    """Current local coverage of the public 53-dataset benchmark manifest."""

    search_dirs: Tuple[str, ...]
    dataset_dirs_env_was_set: bool
    found_paths: Mapping[str, str]
    missing: Tuple[str, ...]

    @property
    def required_count(self) -> int:
        return len(SELECTED_DATASETS)

    @property
    def found_count(self) -> int:
        return len(self.found_paths)

    @property
    def is_complete(self) -> bool:
        return not self.missing and self.found_count == self.required_count


@dataclass(frozen=True)
class FullBenchmarkReadiness:
    """Gate state for the expensive full benchmark entrypoint."""

    inventory: DatasetInventory
    confirmation_present: bool
    cuda_available: Optional[bool]
    cpu_override_present: bool
    reasons: Tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.reasons


def get_configured_dataset_dirs(
    env: Optional[Mapping[str, str]] = None,
    project_root: str = PROJECT_ROOT,
) -> Tuple[Tuple[str, ...], bool]:
    """Return dataset search directories and whether they came from env config."""

    env = os.environ if env is None else env
    dataset_dirs_env = env.get(DATASET_DIRS_ENV, "")
    if dataset_dirs_env:
        return tuple(p for p in dataset_dirs_env.split(os.pathsep) if p), True
    return (os.path.join(project_root, "data"),), False


def _all_h5ad_files(search_dirs: Sequence[str]) -> Tuple[str, ...]:
    files = []
    for directory in search_dirs:
        files.extend(glob.glob(os.path.join(directory, "*.h5ad")))
    return tuple(f for f in files if not any(excluded in f for excluded in EXCLUDE))


def match_benchmark_datasets(search_dirs: Sequence[str]) -> Tuple[Dict[str, str], Tuple[str, ...]]:
    """Match configured ``.h5ad`` files to ``SELECTED_DATASETS`` in benchmark order."""

    all_files = _all_h5ad_files(search_dirs)
    found: Dict[str, str] = {}
    missing = []
    for name in SELECTED_DATASETS:
        exact = [
            f for f in all_files
            if os.path.basename(f).replace(".h5ad", "") == name
        ]
        if exact:
            found[name] = exact[0]
            continue
        matches = [
            f for f in all_files
            if name in os.path.basename(f).replace(".h5ad", "")
        ]
        if matches:
            found[name] = matches[0]
        else:
            missing.append(name)
    return found, tuple(missing)


def benchmark_dataset_inventory(
    env: Optional[Mapping[str, str]] = None,
    project_root: str = PROJECT_ROOT,
) -> DatasetInventory:
    """Summarize whether the current environment provides the full dataset set."""

    search_dirs, env_was_set = get_configured_dataset_dirs(env=env, project_root=project_root)
    found, missing = match_benchmark_datasets(search_dirs)
    return DatasetInventory(
        search_dirs=search_dirs,
        dataset_dirs_env_was_set=env_was_set,
        found_paths=found,
        missing=missing,
    )


def detect_cuda_available() -> Optional[bool]:
    """Return CUDA availability without making torch a guard import requirement."""

    try:
        import torch  # type: ignore
    except Exception:
        return None
    return bool(torch.cuda.is_available())


def evaluate_full_benchmark_readiness(
    env: Optional[Mapping[str, str]] = None,
    project_root: str = PROJECT_ROOT,
    cuda_available: Optional[bool] = None,
) -> FullBenchmarkReadiness:
    """Validate data-root, explicit-confirmation, and runtime prerequisites."""

    env = os.environ if env is None else env
    inventory = benchmark_dataset_inventory(env=env, project_root=project_root)
    confirmation_present = env.get(FULL_BENCHMARK_CONFIRM_ENV) == "1"
    cpu_override_present = env.get(FULL_BENCHMARK_ALLOW_CPU_ENV) == "1"
    if cuda_available is None:
        cuda_available = detect_cuda_available()

    reasons = []
    if not inventory.dataset_dirs_env_was_set:
        reasons.append(
            f"{DATASET_DIRS_ENV} is unset; only repo-local data/ would be searched."
        )
    if not inventory.is_complete:
        reasons.append(
            f"configured data roots provide {inventory.found_count}/"
            f"{inventory.required_count} required benchmark .h5ad files."
        )
    if not confirmation_present:
        reasons.append(
            f"{FULL_BENCHMARK_CONFIRM_ENV}=1 is required to acknowledge the full run."
        )
    if cuda_available is not True and not cpu_override_present:
        reasons.append(
            "CUDA GPU is not available/detected; set "
            f"{FULL_BENCHMARK_ALLOW_CPU_ENV}=1 only for an intentionally planned CPU run."
        )

    return FullBenchmarkReadiness(
        inventory=inventory,
        confirmation_present=confirmation_present,
        cuda_available=cuda_available,
        cpu_override_present=cpu_override_present,
        reasons=tuple(reasons),
    )


def format_full_benchmark_run_plan(status: FullBenchmarkReadiness) -> str:
    """Render a public-safe exact run plan instead of starting unsafe full runs."""

    inventory = status.inventory
    lines = [
        "Full benchmark guard: " + ("PASS" if status.ok else "NOT STARTED"),
        f"Dataset roots ({DATASET_DIRS_ENV}): "
        + (os.pathsep.join(inventory.search_dirs) if inventory.search_dirs else "<none>"),
        f"Dataset coverage: {inventory.found_count}/{inventory.required_count} required .h5ad files",
        f"CUDA available: {status.cuda_available if status.cuda_available is not None else 'unknown'}",
        f"Explicit confirmation: {FULL_BENCHMARK_CONFIRM_ENV}="
        + ("1" if status.confirmation_present else "<missing>"),
    ]
    if status.reasons:
        lines.append("")
        lines.append("Reasons the full 53-dataset run was not started:")
        lines.extend(f"- {reason}" for reason in status.reasons)
    if inventory.missing:
        preview = ", ".join(inventory.missing[:10])
        suffix = "" if len(inventory.missing) <= 10 else f", ... (+{len(inventory.missing) - 10} more)"
        lines.append(f"Missing datasets: {preview}{suffix}")

    lines.extend([
        "",
        "Exact run plan when prerequisites are satisfied:",
        "1. Stage authorized benchmark .h5ad files whose basenames match docs/DATASET_MANIFEST.csv.",
        f"2. Export data roots, for example: export {DATASET_DIRS_ENV}="
        '"/path/to/CancerDatasets:/path/to/DevelopmentDatasets"',
        "3. Confirm an appropriate runtime before launching: choose CUDA_VISIBLE_DEVICES=<gpu-id> on a suitable GPU node.",
        f"4. Acknowledge the expensive run: export {FULL_BENCHMARK_CONFIRM_ENV}=1",
        f"   If intentionally running CPU-only, also export {FULL_BENCHMARK_ALLOW_CPU_ENV}=1.",
        "5. Launch exactly: bash experiments/run_all_expanded.sh",
        "6. The runner will execute these commands in order:",
    ])
    for label, script in FULL_BENCHMARK_STEPS:
        lines.append(f"   - {label}: ${{GAHIB_PYTHON:-python}} {script}")
    lines.extend([
        "7. Monitor GAHIB_results/run_expanded.log and per-study GAHIB_results/*/tables outputs.",
        "8. Do not claim new metrics until every per-dataset table exists and has been reviewed.",
    ])
    return "\n".join(lines)


def main() -> int:
    status = evaluate_full_benchmark_readiness()
    print(format_full_benchmark_run_plan(status))
    return 0 if status.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
