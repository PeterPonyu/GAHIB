"""
Shared utilities for GAHIB experiment scripts.
============================================
Follows the training-pipeline skill Phase 2 preprocessing exactly:
  1. Save raw counts in layers['counts']
  2. Normalize (target_sum=1e4) + log1p
  3. Select highly variable genes (n_top_genes=2000)
  4. Subsample to max_cells=3000
  5. Subset to (subsampled cells) × (HVG genes) and .copy()

The resulting adata1 is used by ALL models:
  - GAHIB variants:  GAHIB(adata1, layer='counts', ...)
  - External models: X = adata1.X.toarray()  (normalized HVG)
  - Labels:          get_labels(adata1)
"""

import sys, os, glob, logging
import numpy as np
import pandas as pd
import scanpy as sc

logger = logging.getLogger(__name__)
import scipy.sparse as sp

# Add project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from gahib.metrics import compute_all_metrics

# ── Constants ──
MAX_CELLS = 3000
N_HVG = 2000
SEED = 42

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


def discover_datasets():
    """Find all h5ad files and filter to selected 53.

    Dataset directories are configurable via the GAHIB_DATASET_DIRS environment
    variable (colon-separated list of paths). Falls back to ~/Downloads/ defaults.
    """
    _dataset_dirs_env = os.environ.get("GAHIB_DATASET_DIRS", "")
    if _dataset_dirs_env:
        search_dirs = _dataset_dirs_env.split(os.pathsep)
    else:
        search_dirs = [
            os.path.expanduser("~/Downloads/CancerDatasets2"),
            os.path.expanduser("~/Downloads/CancerDatasets"),
            os.path.expanduser("~/Downloads/DevelopmentDatasets2"),
            os.path.expanduser("~/Downloads/DevelopmentDatasets"),
        ]
    all_files = []
    for d in search_dirs:
        all_files.extend(glob.glob(os.path.join(d, "*.h5ad")))
    all_files = [f for f in all_files if not any(e in f for e in EXCLUDE)]

    selected = []
    for name in SELECTED_DATASETS:
        # Prefer exact filename match, then fall back to substring
        exact = [f for f in all_files
                 if os.path.basename(f).replace('.h5ad', '') == name]
        if exact:
            selected.append(exact[0])
        else:
            matches = [f for f in all_files
                       if name in os.path.basename(f).replace('.h5ad', '')]
            if matches:
                selected.append(matches[0])
            else:
                print(f"⚠ Dataset not found: {name}")
    return selected


def get_labels(adata, resolution=1.0):
    """Compute unsupervised reference labels via Leiden clustering.

    All benchmarking uses Leiden on preprocessed data as the reference
    partition.  Ground-truth cell type annotations are never used,
    ensuring fully unsupervised evaluation.

    Parameters
    ----------
    adata : AnnData
        Preprocessed data (normalized, log-transformed, HVG-selected).
    resolution : float
        Leiden resolution parameter.  Default 1.0.

    Returns
    -------
    labels : ndarray of str
        Leiden cluster assignments.
    n_clusters : int
        Number of Leiden clusters found.
    """
    leiden_key = f'leiden_{resolution}'
    if leiden_key not in adata.obs.columns:
        if 'neighbors' not in adata.uns:
            use_rep = 'X_pca' if 'X_pca' in adata.obsm else None
            sc.pp.neighbors(adata, use_rep=use_rep)
        sc.tl.leiden(adata, resolution=resolution, key_added=leiden_key)
    labels = adata.obs[leiden_key].values.astype(str)
    n_clusters = len(np.unique(labels))
    logger.info("  Leiden (res=%.1f): %d clusters", resolution, n_clusters)
    return labels, n_clusters


def load_and_preprocess(filepath):
    """Load h5ad and apply the MANDATORY preprocessing pipeline.

    Returns
    -------
    adata1 : AnnData
        Preprocessed data with:
        - adata1.X = normalized log-transformed HVG expression
        - adata1.layers['counts'] = raw integer counts (HVG subset)
        - adata1.n_obs <= MAX_CELLS (3000)
        - adata1.n_vars = N_HVG (2000)
    """
    adata = sc.read_h5ad(filepath)
    adata.obs_names_make_unique()
    adata.var_names_make_unique()

    # 1. Ensure sparse
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # 2. Save raw counts BEFORE normalization
    adata.layers['counts'] = adata.X.copy()

    # 3. Normalize + log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 4. Select highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)

    # 5. Subsample cells to fixed size
    rng = np.random.default_rng(SEED)
    if adata.shape[0] > MAX_CELLS:
        idxs = rng.choice(adata.shape[0], MAX_CELLS, replace=False)
    else:
        idxs = rng.permutation(adata.shape[0])

    # 6. Subset to (subsampled cells) x (HVG genes) and COPY
    adata1 = adata[idxs, adata.var['highly_variable']].copy()

    print(f"  Preprocessed: {adata.n_obs} cells -> {adata1.n_obs} cells, "
          f"{adata.n_vars} genes -> {adata1.n_vars} HVGs")

    return adata1


def get_dense_X(adata1):
    """Get dense normalized HVG matrix from preprocessed adata1.
    Use this as input for external models (GM-VAE, etc.)."""
    X = adata1.X
    if sp.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.float32)


def evaluate_latent(latent, labels, dre_k=15):
    """Compute all metrics for a latent embedding."""
    # Encode string labels to integers (compute_all_metrics expects int)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    labels_int = le.fit_transform(np.asarray(labels).astype(str))
    raw = compute_all_metrics(latent, labels_int, dre_k=dre_k)
    return {k: v for k, v in raw.items()
            if not k.startswith('_') and np.isscalar(v)}


def get_done_datasets(tables_dir, prefix):
    """Check which datasets have already been processed (for resume)."""
    done = set()
    for f in glob.glob(os.path.join(tables_dir, f'{prefix}_*_df.csv')):
        name = os.path.basename(f).replace(f'{prefix}_', '').replace('_df.csv', '')
        done.add(name)
    return done
