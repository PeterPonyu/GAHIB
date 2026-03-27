# ============================================================================
# interpretation.py - Post-hoc Interpretation & Biovalidation of GAHIB Models
# ============================================================================
"""
Extraction and analysis routines for interpreting GAHIB model components
and downstream biological validation:

**Model Interpretation (4 axes):**
  1. GAT Attention — edge-level attention → cell-type attention patterns
  2. Information Bottleneck — bottleneck (i_dim=2) structure & retention
  3. Hyperbolic Geometry — Lorentz norms, Poincaré projection, hierarchy
  4. Gene Attribution — Decoder Jacobian → per-dimension gene programs

**Biological Validation (5 axes):**
  5. Gene Program Discovery — enrichment analysis of latent-dimension genes
  6. Stemness–Hierarchy Correlation — Lorentz norms vs differentiation score
  7. Marker Gene Recovery — overlap with known cell-type marker databases
  8. Latent Traversal — in-silico perturbation along latent dimensions
  9. Reconstruction Quality — per-cell / per-gene fidelity analysis
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ============================================================================
# Data containers
# ============================================================================

@dataclass
class InterpretationResult:
    """All interpretation outputs for a single dataset."""
    dataset_name: str
    labels: np.ndarray                          # (n_cells,) str cluster labels
    label_names: Optional[List[str]] = None     # unique sorted label names

    # Latent representations
    q_z: Optional[np.ndarray] = None            # (n_cells, latent_dim)
    q_m: Optional[np.ndarray] = None            # (n_cells, latent_dim) posterior mean
    q_s: Optional[np.ndarray] = None            # (n_cells, latent_dim) posterior logvar

    # Information Bottleneck
    le: Optional[np.ndarray] = None             # (n_cells, i_dim)  bottleneck
    ld: Optional[np.ndarray] = None             # (n_cells, latent_dim) recovered
    ib_retention: Optional[np.ndarray] = None   # (n_cells,) MSE(q_z, ld)

    # Hyperbolic geometry
    z_manifold: Optional[np.ndarray] = None     # (n_cells, latent_dim+1) Lorentz
    lorentz_norms: Optional[np.ndarray] = None  # (n_cells,) distance from origin
    poincare_coords: Optional[np.ndarray] = None  # (n_cells, latent_dim) Poincaré ball

    # GAT attention
    edge_index: Optional[np.ndarray] = None     # (2, n_edges) original graph
    attn_edge_index: Optional[np.ndarray] = None  # (2, n_edges') expanded (with self-loops)
    attention_weights: Optional[List[np.ndarray]] = None  # per-layer attention
    attn_type_matrix: Optional[np.ndarray] = None  # (n_types, n_types) mean attention
    attn_homophily: Optional[float] = None      # fraction same-type in top-K

    # Gene attribution (decoder Jacobian)
    gene_scores: Optional[np.ndarray] = None    # (n_genes, latent_dim) |∂gene/∂z|
    gene_names: Optional[np.ndarray] = None     # (n_genes,) gene name strings
    top_genes_per_dim: Optional[Dict[int, List[str]]] = field(default_factory=dict)

    # Posterior variance analysis
    dim_variance: Optional[np.ndarray] = None   # (latent_dim,) variance per dim

    # --- Biovalidation fields ---

    # Gene program enrichment (per latent dimension)
    enrichment_results: Optional[Dict[int, List[Dict]]] = field(default_factory=dict)

    # Stemness–hierarchy correlation
    stemness_scores: Optional[np.ndarray] = None       # (n_cells,)
    stemness_norm_corr: Optional[float] = None         # Spearman(stemness, lorentz_norm)
    stemness_norm_pval: Optional[float] = None

    # Marker gene recovery
    marker_overlap: Optional[Dict[str, Dict]] = field(default_factory=dict)

    # Latent traversal (gene response to perturbation)
    traversal_responses: Optional[Dict[int, np.ndarray]] = field(default_factory=dict)
    # key=dim, value=(n_steps, n_genes) decoder output along traversal

    # Reconstruction quality
    recon_per_cell: Optional[np.ndarray] = None        # (n_cells,) per-cell loss
    recon_per_gene: Optional[np.ndarray] = None        # (n_genes,) per-gene loss
    recon_per_type: Optional[Dict[str, float]] = field(default_factory=dict)

    # Hyperbolic hierarchy
    hyp_dist_matrix: Optional[np.ndarray] = None       # (n_types, n_types)
    hyp_dist_labels: Optional[List[str]] = None


# ============================================================================
# GAT Attention Extraction
# ============================================================================

def extract_gat_attention(
    model,
    X_norm: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Extract attention weights from GAT layers using return_attention_weights.

    Manually replays the encoder forward pass, calling each GATConv with
    ``return_attention_weights=True`` to capture per-edge attention coefficients.

    Returns
    -------
    attn_list : list of arrays, one per GATConv layer, shape (n_edges', n_heads)
    ei_list : list of arrays, one per GATConv layer, shape (2, n_edges')
        The expanded edge indices (with self-loops) matching each attention array.
    """
    from torch_geometric.nn import GATConv

    device = model.device
    encoder = model.nn.encoder

    x = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device)
    ew = torch.as_tensor(edge_weight, dtype=torch.float32, device=device)

    attn_list = []
    ei_list = []

    with torch.no_grad():
        residual = None
        h = x

        for i, (conv, bn, drop) in enumerate(
            zip(encoder.convs, encoder.bns, encoder.dropouts)
        ):
            if isinstance(conv, GATConv):
                out, (ei_expanded, alpha) = conv(
                    h, ei, return_attention_weights=True
                )
                attn_list.append(alpha.cpu().numpy())
                ei_list.append(ei_expanded.cpu().numpy())
                h = out
            else:
                h = encoder._process_layer(h, conv, ei, ew)

            h = bn(h)
            h = encoder.relu(h)
            h = drop(h)
            if encoder.use_residual if hasattr(encoder, 'use_residual') else (i == 0):
                if i == 0:
                    residual = h

        if residual is not None:
            h = h + residual

        # conv_mean layer
        if hasattr(encoder, 'conv_mean') and isinstance(encoder.conv_mean, GATConv):
            _, (ei_expanded, alpha) = encoder.conv_mean(
                h, ei, return_attention_weights=True
            )
            attn_list.append(alpha.cpu().numpy())
            ei_list.append(ei_expanded.cpu().numpy())

    if not attn_list:
        logger.warning("No GAT attention weights captured. Is graph_type='GAT'?")

    return attn_list, ei_list


def compute_attention_homophily(
    attention_weights: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
    top_k_fraction: float = 0.1,
) -> float:
    """Fraction of top-attended edges connecting same-type cells."""
    if attention_weights.ndim == 2:
        attn = attention_weights.mean(axis=1)  # average over heads
    else:
        attn = attention_weights

    n_top = max(1, int(len(attn) * top_k_fraction))
    top_idx = np.argsort(attn)[-n_top:]

    src = edge_index[0, top_idx]
    dst = edge_index[1, top_idx]
    same = (labels[src] == labels[dst]).sum()
    return float(same / n_top)


def compute_attention_type_matrix(
    attention_weights: np.ndarray,
    edge_index: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Mean attention between cell-type pairs → (n_types, n_types) matrix."""
    if attention_weights.ndim == 2:
        attn = attention_weights.mean(axis=1)
    else:
        attn = attention_weights

    unique_labels = np.unique(labels)
    n_types = len(unique_labels)
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    mat = np.zeros((n_types, n_types), dtype=np.float64)
    counts = np.zeros((n_types, n_types), dtype=np.float64)

    src_labels = labels[edge_index[0]]
    dst_labels = labels[edge_index[1]]

    for e in range(len(attn)):
        i = label_to_idx[src_labels[e]]
        j = label_to_idx[dst_labels[e]]
        mat[i, j] += attn[e]
        counts[i, j] += 1

    counts[counts == 0] = 1
    mat /= counts

    return mat, list(unique_labels)


# ============================================================================
# Information Bottleneck Analysis
# ============================================================================

def extract_bottleneck_representations(
    model,
    X_norm: np.ndarray,
    edge_index: Optional[np.ndarray],
    edge_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract q_z, q_m, q_s, le (bottleneck), ld (recovered), z_manifold.

    Returns (q_z, q_m, q_s, le, ld, z_manifold) all as numpy arrays.
    """
    device = model.device
    nn_module = model.nn

    x = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device) if edge_index is not None else None
    ew = torch.as_tensor(edge_weight, dtype=torch.float32, device=device) if edge_weight is not None else None

    with torch.no_grad():
        outputs = nn_module(x, edge_index=ei, edge_weight=ew)

    return (
        outputs.q_z.cpu().numpy(),
        outputs.q_m.cpu().numpy(),
        outputs.q_s.cpu().numpy(),
        outputs.le.cpu().numpy(),
        outputs.ld.cpu().numpy(),
        outputs.z_manifold.cpu().numpy(),
    )


def compute_ib_retention(q_z: np.ndarray, ld: np.ndarray) -> np.ndarray:
    """Per-cell information retention: MSE between original and recovered latent."""
    return np.mean((q_z - ld) ** 2, axis=1)


# ============================================================================
# Hyperbolic Geometry Analysis
# ============================================================================

def compute_lorentz_norms(z_manifold: np.ndarray) -> np.ndarray:
    """Hyperbolic distance from origin on Lorentz hyperboloid.

    For a point x on H^n, d(o, x) = acosh(x_0) since origin = (1, 0, ..., 0).
    """
    x0 = z_manifold[:, 0]
    x0_clamped = np.clip(x0, a_min=1.0 + 1e-8, a_max=None)
    return np.arccosh(x0_clamped)


def poincare_projection(z_manifold: np.ndarray) -> np.ndarray:
    """Project from Lorentz hyperboloid to Poincaré ball.

    y_i = x_i / (1 + x_0)  for i = 1, ..., n
    """
    x0 = z_manifold[:, 0:1]
    spatial = z_manifold[:, 1:]
    return spatial / (1.0 + x0)


def compute_hyperbolic_distances_between_types(
    z_manifold: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Mean pairwise hyperbolic distance between cell-type centroids.

    Returns (distance_matrix, label_names).
    """
    from .core.utils import lorentzian_product

    unique_labels = np.unique(labels)
    n_types = len(unique_labels)

    # Compute centroids on the hyperboloid (Fréchet mean approximation:
    # average in ambient space then project back)
    centroids = []
    for lbl in unique_labels:
        mask = labels == lbl
        centroid = z_manifold[mask].mean(axis=0)
        # Re-normalize to hyperboloid: x_0 = sqrt(1 + ||x_spatial||^2)
        spatial = centroid[1:]
        centroid[0] = np.sqrt(1.0 + np.sum(spatial ** 2))
        centroids.append(centroid)

    centroids = np.array(centroids)

    # Compute pairwise hyperbolic distances
    dist_mat = np.zeros((n_types, n_types))
    for i in range(n_types):
        for j in range(i + 1, n_types):
            xi, xj = centroids[i], centroids[j]
            # Lorentzian product: -x0*y0 + sum(xi*yi)
            inner = -xi[0] * xj[0] + np.sum(xi[1:] * xj[1:])
            arg = np.clip(-inner, 1.0 + 1e-8, None)
            d = np.arccosh(arg)
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    return dist_mat, list(unique_labels)


# ============================================================================
# Gene Attribution (Decoder Jacobian)
# ============================================================================

def compute_decoder_jacobian(
    model,
    q_z: np.ndarray,
    n_samples: int = 300,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute |∂decoder/∂z| averaged over sampled cells.

    Returns gene_scores of shape (n_genes, latent_dim).
    """
    device = model.device
    nn_module = model.nn
    decoder = nn_module.decoder

    rng = np.random.default_rng(42)
    n_cells = q_z.shape[0]
    sample_idx = rng.choice(n_cells, min(n_samples, n_cells), replace=False)
    z_samples = q_z[sample_idx]

    n_genes = None
    latent_dim = z_samples.shape[1]
    accum = None

    for start in range(0, len(z_samples), batch_size):
        batch = z_samples[start:start + batch_size]
        z = torch.tensor(batch, dtype=torch.float32, device=device, requires_grad=True)

        # Decoder forward (MLP decoder returns (pred_x, dropout_x))
        pred_x, _ = decoder(z)

        if n_genes is None:
            n_genes = pred_x.shape[1]
            accum = np.zeros((n_genes, latent_dim), dtype=np.float64)

        # Compute Jacobian: for each gene, backprop through sum over batch
        for g in range(n_genes):
            if z.grad is not None:
                z.grad.zero_()
            pred_x[:, g].sum().backward(retain_graph=(g < n_genes - 1))
            accum[g] += np.abs(z.grad.cpu().numpy()).sum(axis=0)

        # Clear graph
        del pred_x, z

    accum /= len(z_samples)
    return accum


def compute_decoder_jacobian_fast(
    model,
    q_z: np.ndarray,
    n_samples: int = 200,
) -> np.ndarray:
    """Faster Jacobian via per-dimension perturbation (finite differences).

    Returns gene_scores of shape (n_genes, latent_dim).
    """
    device = model.device
    decoder = model.nn.decoder

    rng = np.random.default_rng(42)
    n_cells = q_z.shape[0]
    sample_idx = rng.choice(n_cells, min(n_samples, n_cells), replace=False)
    z_base = torch.tensor(q_z[sample_idx], dtype=torch.float32, device=device)

    eps = 1e-3
    latent_dim = z_base.shape[1]

    with torch.no_grad():
        pred_base, _ = decoder(z_base)
        n_genes = pred_base.shape[1]
        gene_scores = np.zeros((n_genes, latent_dim), dtype=np.float64)

        for d in range(latent_dim):
            z_perturbed = z_base.clone()
            z_perturbed[:, d] += eps
            pred_perturbed, _ = decoder(z_perturbed)
            diff = (pred_perturbed - pred_base) / eps
            gene_scores[:, d] = np.abs(diff.cpu().numpy()).mean(axis=0)

    return gene_scores


def get_top_genes_per_dimension(
    gene_scores: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 15,
) -> Dict[int, List[str]]:
    """Top-K genes per latent dimension by attribution score."""
    latent_dim = gene_scores.shape[1]
    result = {}
    for d in range(latent_dim):
        idx = np.argsort(gene_scores[:, d])[-top_k:][::-1]
        result[d] = list(gene_names[idx])
    return result


# ============================================================================
# Posterior Variance Analysis
# ============================================================================

def compute_dimension_variance(q_m: np.ndarray) -> np.ndarray:
    """Variance of posterior means across cells, per latent dimension."""
    return np.var(q_m, axis=0)


def compute_dimension_utilization(q_m: np.ndarray, q_s: np.ndarray) -> np.ndarray:
    """Dimension utilization: ratio of inter-cell variance to mean posterior variance.

    High values indicate the dimension encodes meaningful variation.
    Low values indicate the dimension collapsed to the prior (KL inactive).
    """
    inter_var = np.var(q_m, axis=0)
    mean_post_var = np.mean(F.softplus(torch.tensor(q_s)).numpy() ** 2, axis=0)
    return inter_var / (mean_post_var + 1e-8)


# ============================================================================
# Full Interpretation Pipeline
# ============================================================================

def run_interpretation(
    model,
    labels: np.ndarray,
    dataset_name: str,
    gene_names: Optional[np.ndarray] = None,
    n_jacobian_samples: int = 200,
) -> InterpretationResult:
    """Run all interpretation analyses on a trained GAHIB model.

    Parameters
    ----------
    model : GAHIB
        Trained model (after .fit()).
    labels : ndarray of str
        Cluster labels for all cells.
    dataset_name : str
        Name of the dataset for display/storage.
    gene_names : ndarray of str, optional
        Gene names matching model's input features.
    n_jacobian_samples : int
        Number of cells to sample for Jacobian computation.

    Returns
    -------
    InterpretationResult with all computed fields populated.
    """
    result = InterpretationResult(
        dataset_name=dataset_name,
        labels=labels,
        label_names=sorted(np.unique(labels).tolist()),
    )

    logger.info("Extracting representations for %s...", dataset_name)

    # 1. Core representations
    q_z, q_m, q_s, le, ld, z_manifold = extract_bottleneck_representations(
        model, model.X_norm, model.edge_index, model.edge_weight,
    )
    result.q_z = q_z
    result.q_m = q_m
    result.q_s = q_s
    result.le = le
    result.ld = ld
    result.z_manifold = z_manifold

    # 2. Information Bottleneck
    result.ib_retention = compute_ib_retention(q_z, ld)
    logger.info("  IB retention MSE: mean=%.4f, std=%.4f",
                result.ib_retention.mean(), result.ib_retention.std())

    # 3. Hyperbolic geometry
    result.lorentz_norms = compute_lorentz_norms(z_manifold)
    result.poincare_coords = poincare_projection(z_manifold)
    logger.info("  Lorentz norms: mean=%.3f, range=[%.3f, %.3f]",
                result.lorentz_norms.mean(),
                result.lorentz_norms.min(), result.lorentz_norms.max())

    # 4. Posterior variance
    result.dim_variance = compute_dimension_variance(q_m)

    # 5. GAT attention (if graph encoder)
    if model.encoder_type == "graph" and model.edge_index is not None:
        result.edge_index = model.edge_index
        attn_list, ei_list = extract_gat_attention(
            model, model.X_norm, model.edge_index, model.edge_weight,
        )
        result.attention_weights = attn_list

        if attn_list:
            # Use last layer's attention + its expanded edge_index
            last_attn = attn_list[-1]
            last_ei = ei_list[-1]
            result.attn_edge_index = last_ei
            result.attn_homophily = compute_attention_homophily(
                last_attn, last_ei, labels,
            )
            result.attn_type_matrix, _ = compute_attention_type_matrix(
                last_attn, last_ei, labels,
            )
            logger.info("  Attention homophily: %.3f", result.attn_homophily)

    # 6. Gene attribution (decoder Jacobian)
    if gene_names is not None:
        result.gene_names = gene_names
    logger.info("  Computing decoder Jacobian (%d samples)...", n_jacobian_samples)
    result.gene_scores = compute_decoder_jacobian_fast(
        model, q_z, n_samples=n_jacobian_samples,
    )
    if gene_names is not None:
        result.top_genes_per_dim = get_top_genes_per_dimension(
            result.gene_scores, gene_names, top_k=15,
        )

    logger.info("  Interpretation complete for %s", dataset_name)
    return result


# ============================================================================
# 5. Gene Program Discovery (Enrichment Analysis)
# ============================================================================

def compute_gene_enrichment(
    gene_scores: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 100,
) -> Dict[int, List[Dict]]:
    """Run enrichment on top-K attributed genes per latent dimension.

    Uses built-in hypergeometric overlap test against curated pathway gene
    sets (deterministic, no network dependency).

    Returns dict mapping dimension → list of enrichment dicts.
    """
    results = _enrichment_builtin(gene_scores, gene_names, top_k)
    return results


def _enrichment_builtin(
    gene_scores: np.ndarray,
    gene_names: np.ndarray,
    top_k: int = 100,
) -> Dict[int, List[Dict]]:
    """Built-in enrichment using known gene set collections.

    Uses curated pathway signatures from MSigDB hallmark gene sets
    (hardcoded top sets relevant to single-cell biology).
    """
    from scipy.stats import hypergeom

    # Curated pathway gene sets (subset of MSigDB hallmarks relevant to scRNA)
    PATHWAY_SETS = _get_builtin_pathway_sets()

    latent_dim = gene_scores.shape[1]
    gene_set = set(gene_names)
    N = len(gene_names)
    results = {}

    for d in range(latent_dim):
        top_idx = np.argsort(gene_scores[:, d])[-top_k:][::-1]
        query = set(gene_names[top_idx])
        n = len(query)
        hits = []

        for pathway_name, pathway_genes in PATHWAY_SETS.items():
            pathway_in_bg = pathway_genes & gene_set
            K = len(pathway_in_bg)
            if K < 3:
                continue
            overlap = query & pathway_in_bg
            k = len(overlap)
            if k < 2:
                continue
            # P(X >= k) under hypergeometric
            pval = hypergeom.sf(k - 1, N, K, n)
            hits.append({
                "term": pathway_name,
                "pvalue": pval,
                "overlap": k,
                "pathway_size": K,
                "genes": ",".join(sorted(overlap)),
                "source": "builtin",
            })

        # Sort by p-value, keep top 10
        hits.sort(key=lambda x: x["pvalue"])
        results[d] = hits[:10]

    return results


def _get_builtin_pathway_sets() -> Dict[str, set]:
    """Curated pathway gene sets for single-cell enrichment analysis.

    Covers: cell cycle, apoptosis, immune, metabolism, signalling,
    stemness, differentiation — the biology most relevant to scRNA-seq.
    """
    return {
        "Cell Cycle (G1/S)": {
            "MCM2", "MCM3", "MCM4", "MCM5", "MCM6", "MCM7", "PCNA",
            "RPA1", "RPA2", "RPA3", "CDK2", "CDK4", "CCND1", "CCNE1",
            "CCNE2", "E2F1", "E2F2", "CDC6", "CDC45", "ORC1", "RB1",
            "CDKN1A", "CDKN1B", "MYC",
        },
        "Cell Cycle (G2/M)": {
            "CDK1", "CCNB1", "CCNB2", "CCNA2", "CDC20", "CDC25C",
            "AURKA", "AURKB", "PLK1", "TOP2A", "BUB1", "BUB1B",
            "MAD2L1", "NDC80", "KIF11", "CENPE", "CENPF", "UBE2C",
            "BIRC5", "MKI67", "TPX2", "HMGA2",
        },
        "Apoptosis": {
            "BAX", "BAK1", "BCL2", "BCL2L1", "MCL1", "BID", "BAD",
            "CASP3", "CASP7", "CASP8", "CASP9", "CYCS", "APAF1",
            "FADD", "FAS", "TNFRSF10A", "TNFRSF10B", "XIAP", "BIRC2",
            "DIABLO", "TP53", "PMAIP1", "BBC3",
        },
        "Inflammatory Response": {
            "IL1B", "IL6", "TNF", "CXCL8", "CCL2", "CCL3", "CCL4",
            "CCL5", "CXCL1", "CXCL2", "CXCL10", "NFKB1", "RELA",
            "NFKBIA", "PTGS2", "ICAM1", "VCAM1", "SELE", "IL1A",
            "CSF2", "CSF3", "MMP9", "SERPINE1",
        },
        "Interferon Response": {
            "ISG15", "ISG20", "MX1", "MX2", "OAS1", "OAS2", "OAS3",
            "IFIT1", "IFIT2", "IFIT3", "IFITM1", "IFITM3", "IRF7",
            "IRF9", "STAT1", "STAT2", "BST2", "IFI6", "IFI27",
            "IFI35", "IFI44", "IFI44L", "RSAD2", "CXCL10", "CXCL11",
        },
        "Antigen Presentation (MHC)": {
            "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1",
            "HLA-DPA1", "HLA-DPB1", "HLA-DQA1", "HLA-DQB1",
            "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9",
            "CD74", "CIITA", "NLRC5",
        },
        "T Cell Activation": {
            "CD3D", "CD3E", "CD3G", "CD4", "CD8A", "CD8B", "CD28",
            "CTLA4", "PDCD1", "LAG3", "HAVCR2", "TIGIT", "ICOS",
            "GZMB", "GZMA", "PRF1", "IFNG", "IL2", "IL2RA",
            "FOXP3", "TNFRSF9", "CD69", "CD44",
        },
        "Oxidative Phosphorylation": {
            "NDUFA1", "NDUFA2", "NDUFA3", "NDUFA4", "NDUFB1",
            "NDUFB2", "NDUFB3", "NDUFS1", "NDUFS2", "NDUFS3",
            "COX4I1", "COX5A", "COX5B", "COX6A1", "COX6B1",
            "COX7A2", "COX7B", "COX8A", "ATP5F1A", "ATP5F1B",
            "ATP5F1C", "ATP5PO", "UQCRB", "UQCRC1", "UQCRC2",
        },
        "Glycolysis": {
            "HK1", "HK2", "GPI", "PFKL", "PFKM", "ALDOA", "ALDOB",
            "TPI1", "GAPDH", "PGK1", "PGAM1", "ENO1", "ENO2",
            "PKM", "LDHA", "LDHB", "SLC2A1", "SLC2A3", "PFKFB3",
        },
        "Hypoxia": {
            "HIF1A", "VEGFA", "LDHA", "SLC2A1", "PGK1", "ALDOA",
            "ENO1", "PKM", "BNIP3", "BNIP3L", "CA9", "ADM",
            "PDK1", "EGLN1", "EGLN3", "LOX", "P4HA1", "P4HA2",
        },
        "EMT (Epithelial–Mesenchymal)": {
            "CDH1", "CDH2", "VIM", "FN1", "SNAI1", "SNAI2", "TWIST1",
            "TWIST2", "ZEB1", "ZEB2", "MMP2", "MMP3", "MMP9",
            "ACTA2", "TAGLN", "COL1A1", "COL1A2", "COL3A1",
            "SPARC", "TGFB1", "TGFB2", "FOXC2", "SERPINE1",
        },
        "Stemness / Pluripotency": {
            "POU5F1", "SOX2", "NANOG", "KLF4", "MYC", "LIN28A",
            "LIN28B", "SALL4", "DPPA4", "TDGF1", "UTF1", "ZFP42",
            "PODXL", "PROM1", "ALDH1A1", "CD44", "BMI1", "EZH2",
            "TERT", "ABCG2", "THY1", "NES",
        },
        "Ribosome / Translation": {
            "RPL3", "RPL4", "RPL5", "RPL7", "RPL8", "RPL11",
            "RPL13", "RPL15", "RPL18", "RPL23", "RPL27", "RPL35",
            "RPS3", "RPS4X", "RPS5", "RPS6", "RPS8", "RPS12",
            "RPS14", "RPS18", "RPS19", "RPS24", "RPS27", "EEF1A1",
            "EIF4A1", "EIF4G1",
        },
        "Stress / Heat Shock": {
            "HSPA1A", "HSPA1B", "HSPA5", "HSPA6", "HSPA8",
            "HSP90AA1", "HSP90AB1", "HSP90B1", "HSPB1", "HSPD1",
            "HSPE1", "DNAJB1", "DNAJB4", "DNAJB6", "BAG3",
            "ATF3", "ATF4", "DDIT3", "XBP1", "GADD45A",
        },
    }


# ============================================================================
# 6. Stemness–Hierarchy Correlation
# ============================================================================

def compute_stemness_scores(X_norm: np.ndarray) -> np.ndarray:
    """Compute a CytoTRACE-like stemness score from gene expression.

    Uses gene expression entropy as a proxy for differentiation state:
    - Higher entropy → less differentiated (more stem-like)
    - Lower entropy → more differentiated (committed)

    This is a simplified version of CytoTRACE (Gulati et al., Science 2020)
    that uses Shannon entropy over non-zero gene expression as the core signal.
    """
    # Number of genes expressed per cell (gene count)
    if hasattr(X_norm, 'toarray'):
        X_dense = X_norm.toarray()
    else:
        X_dense = np.asarray(X_norm)

    # Gene counts per cell (proxy for differentiation — CytoTRACE core)
    gene_counts = np.sum(X_dense > 0, axis=1).astype(np.float64)

    # Shannon entropy per cell
    X_pos = np.maximum(X_dense, 0)
    row_sums = X_pos.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    p = X_pos / row_sums
    p_safe = np.where(p > 0, p, 1.0)
    entropy = -np.sum(p * np.log2(p_safe), axis=1)

    # Combine: higher gene count + higher entropy = more stem-like
    from scipy.stats import rankdata
    rank_gc = rankdata(gene_counts) / len(gene_counts)
    rank_ent = rankdata(entropy) / len(entropy)
    stemness = (rank_gc + rank_ent) / 2.0

    return stemness


def correlate_stemness_hierarchy(
    stemness: np.ndarray,
    lorentz_norms: np.ndarray,
) -> Tuple[float, float]:
    """Spearman correlation between stemness score and Lorentz norm.

    Hypothesis: stem cells cluster near the hyperboloid origin (low norm),
    differentiated cells at the periphery (high norm) → negative correlation.
    """
    from scipy.stats import spearmanr
    corr, pval = spearmanr(stemness, lorentz_norms)
    return float(corr), float(pval)


# ============================================================================
# 7. Marker Gene Recovery
# ============================================================================

def compute_marker_overlap(
    gene_scores: np.ndarray,
    gene_names: np.ndarray,
    labels: np.ndarray,
    adata=None,
    top_k: int = 50,
) -> Dict[str, Dict]:
    """Compare decoder-attributed top genes against DE marker genes per cluster.

    Uses scanpy.tl.rank_genes_groups (Wilcoxon) to find empirical markers,
    then measures overlap with decoder Jacobian top-K genes per dimension.
    """
    import scanpy as sc
    import anndata as ad

    if adata is None:
        logger.warning("  Marker recovery: no adata provided, skipping.")
        return {}

    # Run DE to get marker genes per cluster
    adata_copy = adata.copy()
    adata_copy.obs["cluster"] = labels.astype(str)
    sc.tl.rank_genes_groups(adata_copy, groupby="cluster", method="wilcoxon")

    unique_labels = np.unique(labels)
    results = {}
    gene_name_set = set(gene_names)
    latent_dim = gene_scores.shape[1]

    # Aggregate top genes per dimension
    jacobian_top = set()
    for d in range(latent_dim):
        idx = np.argsort(gene_scores[:, d])[-top_k:]
        jacobian_top.update(gene_names[idx])

    for lbl in unique_labels:
        # DE marker genes for this cluster
        de_genes = adata_copy.uns["rank_genes_groups"]["names"][lbl]
        de_top = set(de_genes[:top_k]) & gene_name_set

        if len(de_top) == 0:
            continue

        # Overlap with Jacobian top genes
        overlap = de_top & jacobian_top
        # Best matching dimension
        best_dim = -1
        best_overlap = 0
        for d in range(latent_dim):
            idx = np.argsort(gene_scores[:, d])[-top_k:]
            dim_genes = set(gene_names[idx])
            ov = len(de_top & dim_genes)
            if ov > best_overlap:
                best_overlap = ov
                best_dim = d

        results[str(lbl)] = {
            "n_de_markers": len(de_top),
            "n_jacobian_top": len(jacobian_top),
            "overlap_count": len(overlap),
            "overlap_fraction": len(overlap) / max(len(de_top), 1),
            "overlap_genes": sorted(overlap),
            "best_matching_dim": best_dim,
            "best_dim_overlap": best_overlap,
        }

    del adata_copy
    return results


# ============================================================================
# 8. Latent Traversal (In-Silico Perturbation)
# ============================================================================

def compute_latent_traversal(
    model,
    q_z: np.ndarray,
    n_steps: int = 11,
    range_sigma: float = 2.0,
) -> Dict[int, np.ndarray]:
    """Traverse each latent dimension and record decoder output changes.

    For each dimension d:
      - Fix all other dims at the population mean
      - Sweep dim d from mean - range_sigma*std to mean + range_sigma*std
      - Record decoder output at each step

    Returns dict mapping dim → (n_steps, n_genes) array of gene expression.
    """
    device = model.device
    decoder = model.nn.decoder
    latent_dim = q_z.shape[1]

    z_mean = q_z.mean(axis=0)
    z_std = q_z.std(axis=0)

    results = {}

    with torch.no_grad():
        for d in range(latent_dim):
            if z_std[d] < 1e-6:
                continue

            steps = np.linspace(
                z_mean[d] - range_sigma * z_std[d],
                z_mean[d] + range_sigma * z_std[d],
                n_steps,
            )

            z_sweep = np.tile(z_mean, (n_steps, 1))
            z_sweep[:, d] = steps

            z_t = torch.tensor(z_sweep, dtype=torch.float32, device=device)
            pred_x, _ = decoder(z_t)
            results[d] = pred_x.cpu().numpy()

    return results


def identify_responsive_genes(
    traversal_responses: Dict[int, np.ndarray],
    gene_names: np.ndarray,
    top_k: int = 20,
) -> Dict[int, List[Tuple[str, float]]]:
    """For each dimension, find genes with highest response range during traversal.

    Returns dict mapping dim → list of (gene_name, response_range).
    """
    results = {}
    for d, response in traversal_responses.items():
        # response: (n_steps, n_genes)
        gene_range = response.max(axis=0) - response.min(axis=0)
        top_idx = np.argsort(gene_range)[-top_k:][::-1]
        results[d] = [
            (gene_names[i], float(gene_range[i]))
            for i in top_idx
        ]
    return results


# ============================================================================
# 9. Reconstruction Quality Analysis
# ============================================================================

def compute_reconstruction_quality(
    model,
    X_norm: np.ndarray,
    X_raw: np.ndarray,
    edge_index: Optional[np.ndarray],
    edge_weight: Optional[np.ndarray],
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Per-cell and per-gene reconstruction error.

    Returns (per_cell_mse, per_gene_mse, per_type_dict).
    """
    device = model.device
    nn_module = model.nn

    x = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
    ei = torch.as_tensor(edge_index, dtype=torch.long, device=device) if edge_index is not None else None
    ew = torch.as_tensor(edge_weight, dtype=torch.float32, device=device) if edge_weight is not None else None

    with torch.no_grad():
        outputs = nn_module(x, edge_index=ei, edge_weight=ew)
        pred = outputs.pred_x.cpu().numpy()

    # Scale prediction by library size for comparison
    if hasattr(X_raw, 'toarray'):
        raw_dense = X_raw.toarray().astype(np.float64)
    else:
        raw_dense = np.asarray(X_raw, dtype=np.float64)

    lib_sizes = raw_dense.sum(axis=1, keepdims=True)
    lib_sizes[lib_sizes == 0] = 1.0
    pred_scaled = pred * lib_sizes

    # Compute on normalized scale for fair comparison
    pred_norm = np.log1p(pred_scaled / lib_sizes * 1e4)
    x_norm_np = np.asarray(X_norm)

    residuals = (x_norm_np - pred_norm) ** 2

    per_cell = residuals.mean(axis=1)
    per_gene = residuals.mean(axis=0)

    per_type = {}
    for lbl in np.unique(labels):
        mask = labels == lbl
        per_type[str(lbl)] = float(per_cell[mask].mean())

    return per_cell, per_gene, per_type


# ============================================================================
# 10. Hyperbolic Hierarchy Validation
# ============================================================================

def build_hyperbolic_hierarchy(
    z_manifold: np.ndarray,
    labels: np.ndarray,
    lorentz_norms: np.ndarray,
) -> Dict:
    """Analyse the hyperbolic hierarchy learned by the Lorentz geometry.

    Returns a dict with:
      - type_ordering: cell types ordered by mean Lorentz norm (root → leaf)
      - depth_spread: std of norms within each type (homogeneity)
      - hierarchy_score: ratio of between-type norm variance to within-type
    """
    unique_labels = np.unique(labels)
    type_stats = []

    for lbl in unique_labels:
        mask = labels == lbl
        norms = lorentz_norms[mask]
        type_stats.append({
            "type": str(lbl),
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "n_cells": int(mask.sum()),
        })

    type_stats.sort(key=lambda x: x["mean_norm"])

    # Hierarchy score: between-type variance / within-type variance
    all_means = np.array([t["mean_norm"] for t in type_stats])
    all_stds = np.array([t["std_norm"] for t in type_stats])
    between_var = np.var(all_means)
    within_var = np.mean(all_stds ** 2)
    hierarchy_score = float(between_var / (within_var + 1e-8))

    return {
        "type_ordering": type_stats,
        "hierarchy_score": hierarchy_score,
        "between_var": float(between_var),
        "within_var": float(within_var),
    }


# ============================================================================
# Extended Full Pipeline (Biovalidation)
# ============================================================================

def run_biovalidation(
    model,
    adata,
    labels: np.ndarray,
    result: InterpretationResult,
    run_enrichment: bool = True,
    run_traversal: bool = True,
) -> InterpretationResult:
    """Extend an InterpretationResult with downstream biovalidation analyses.

    Parameters
    ----------
    model : GAHIB
        Trained model.
    adata : AnnData
        Preprocessed AnnData (with .X = normalized, .layers['counts'] = raw).
    labels : ndarray of str
        Cluster labels.
    result : InterpretationResult
        Previously computed interpretation result (from run_interpretation).
    run_enrichment : bool
        Whether to run gene enrichment analysis (can be slow).
    run_traversal : bool
        Whether to run latent traversal analysis.

    Returns
    -------
    result : InterpretationResult with biovalidation fields populated.
    """
    import scipy.sparse as sp

    logger.info("Running biovalidation for %s...", result.dataset_name)

    # 5. Gene enrichment
    if run_enrichment and result.gene_scores is not None and result.gene_names is not None:
        logger.info("  Computing gene enrichment...")
        result.enrichment_results = compute_gene_enrichment(
            result.gene_scores, result.gene_names, top_k=100,
        )
        n_enriched = sum(len(v) for v in result.enrichment_results.values())
        logger.info("  Enrichment: %d total terms across %d dimensions",
                     n_enriched, len(result.enrichment_results))

    # 6. Stemness–hierarchy correlation
    logger.info("  Computing stemness scores...")
    result.stemness_scores = compute_stemness_scores(model.X_norm)
    if result.lorentz_norms is not None:
        corr, pval = correlate_stemness_hierarchy(
            result.stemness_scores, result.lorentz_norms,
        )
        result.stemness_norm_corr = corr
        result.stemness_norm_pval = pval
        logger.info("  Stemness–norm correlation: r=%.3f, p=%.2e", corr, pval)

    # 7. Marker gene recovery
    if result.gene_scores is not None:
        logger.info("  Computing marker gene overlap...")
        result.marker_overlap = compute_marker_overlap(
            result.gene_scores, result.gene_names, labels, adata=adata,
        )
        if result.marker_overlap:
            mean_frac = np.mean([
                v["overlap_fraction"] for v in result.marker_overlap.values()
            ])
            logger.info("  Marker overlap: mean fraction=%.3f", mean_frac)

    # 8. Latent traversal
    if run_traversal and result.q_z is not None:
        logger.info("  Computing latent traversals...")
        result.traversal_responses = compute_latent_traversal(
            model, result.q_z, n_steps=11, range_sigma=2.0,
        )

    # 9. Reconstruction quality
    logger.info("  Computing reconstruction quality...")
    X_raw = adata.layers.get("counts", adata.X)
    per_cell, per_gene, per_type = compute_reconstruction_quality(
        model, model.X_norm, X_raw,
        model.edge_index, model.edge_weight, labels,
    )
    result.recon_per_cell = per_cell
    result.recon_per_gene = per_gene
    result.recon_per_type = per_type
    logger.info("  Reconstruction MSE: mean=%.4f, std=%.4f",
                per_cell.mean(), per_cell.std())

    # 10. Hyperbolic hierarchy
    if result.z_manifold is not None:
        hierarchy = build_hyperbolic_hierarchy(
            result.z_manifold, labels, result.lorentz_norms,
        )
        result.hyp_dist_matrix, result.hyp_dist_labels = \
            compute_hyperbolic_distances_between_types(result.z_manifold, labels)
        logger.info("  Hierarchy score: %.3f (between/within norm variance)",
                     hierarchy["hierarchy_score"])

    logger.info("  Biovalidation complete for %s", result.dataset_name)
    return result
