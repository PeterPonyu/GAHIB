"""
Centralized style configuration for GAHIB publication figures.

Matches the MoCoO article style (IEEE J-BHI format):
- 17 x 21 cm page, 300 DPI, Arial/Liberation Sans font
- Wong colorblind-safe palette adapted for GAHIB configurations
- Absolute-geometry layout (no tight_layout)

All figure scripts import from here to ensure visual consistency.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt

# ── Arial font registration (bundled in gahib/viz/fonts/) ────────────────
_FONT_DIR = Path(__file__).resolve().parent / "fonts"
_LAB_DIR = Path.home() / "LAB"
# Arial Italic / Arial Bold Italic from ~/LAB are known to have corrupt
# glyf tables on some systems; only register Regular + Bold.
_ARIAL_SAFE = {"Arial.ttf", "Arial Bold.ttf"}
for _d in (_FONT_DIR, _LAB_DIR):
    if _d.exists():
        for _ttf in _d.glob("Arial*.ttf"):
            if _ttf.name not in _ARIAL_SAFE:
                continue
            try:
                _fm.fontManager.addfont(str(_ttf))
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Figure geometry (17 cm x 21 cm, matching MoCoO / IEEE J-BHI column width)
# ---------------------------------------------------------------------------
FIG_WIDTH_CM = 17.0
FIG_HEIGHT_CM = 21.0
FIG_WIDTH_IN = FIG_WIDTH_CM / 2.54
FIG_HEIGHT_IN = FIG_HEIGHT_CM / 2.54
DPI = 300
DPI_PNG = 150  # lower DPI for PNG to keep file size < 20 MB

SAVEFIG_KW = dict(dpi=DPI, bbox_inches=None, pad_inches=0.0)

# Absolute-geometry layout rects [left, bottom, width, height]
RECT_BOXPLOT_ROW = [0.05, 0.22, 0.93, 0.68]   # bottom margin for rotated x-labels
RECT_HEATMAP = [0.16, 0.18, 0.68, 0.70]        # left margin for y-labels
RECT_TITLE_Y = 0.95                              # figure title y-position
GAP_BOXPLOT = 0.035                               # inter-panel gap

# Heatmap styling
HEATMAP_DARK_THRESHOLD = 0.45
HEATMAP_CMAP = "YlOrRd"

# Accent colours
ACCENT_POSITIVE = "#2ca02c"
ACCENT_NEGATIVE = "#d62728"
ACCENT_BEST = "crimson"

# ---------------------------------------------------------------------------
# Font sizes (calibrated for 17 x 21 cm canvas)
# ---------------------------------------------------------------------------
FS_LABEL = 20
FS_TITLE = 13
FS_AXIS = 11
FS_TICK = 10
FS_LEGEND = 9
FS_SMALL = 8

# ---------------------------------------------------------------------------
# GAHIB model configurations: canonical order and display names
# ---------------------------------------------------------------------------
_CONFIG_ORDER: List[str] = [
    "Base VAE",
    "VAE+IB",
    "VAE+Hyp",
    "VAE+IB+Hyp",
    "GAHIB",
]

_PALETTE: List[str] = [
    "#0072B2",  # Base VAE     — blue (Wong)
    "#E69F00",  # VAE+IB       — orange (Wong)
    "#009E73",  # VAE+Hyp      — bluish green (Wong)
    "#CC79A7",  # VAE+IB+Hyp   — reddish purple (Wong)
    "#D55E00",  # GAHIB          — vermilion (Wong)
]

_CONFIG_COLORS: Dict[str, str] = OrderedDict(
    zip(_CONFIG_ORDER, _PALETTE)
)

# Display name mapping (includes legacy aliases for old CSV files)
_DISPLAY_NAMES: Dict[str, str] = {
    "Base VAE": "Base VAE",
    "VAE+IB": "VAE+IB",
    "VAE+Hyp": "VAE+Hyp",
    "VAE+IB+Hyp": "VAE+IB+Hyp",
    "GAHIB": "GAHIB",
    # Legacy aliases (old CSV files)
    "VAE": "Base VAE",
    "IRecon-VAE": "VAE+IB",
    "Lorentz-VAE": "VAE+Hyp",
    "GM-VAE": "VAE+IB+Hyp",
    "GAHIB (Full)": "GAHIB",
}

# Short names for tight x-tick labels
_SHORT_NAMES: Dict[str, str] = {
    "Base VAE": "Base",
    "VAE+IB": "+IB",
    "VAE+Hyp": "+Hyp",
    "VAE+IB+Hyp": "+IB+Hyp",
    "GAHIB": "GAHIB",
}

# Legacy name mapping for backward compatibility with existing CSV files
LEGACY_NAME_MAP: Dict[str, str] = {
    "VAE": "Base VAE",
    "IRecon-VAE": "VAE+IB",
    "Lorentz-VAE": "VAE+Hyp",
    "GM-VAE": "VAE+IB+Hyp",
    "GAHIB (Full)": "GAHIB",
    "GAT+IB+Lorentz": "GAT+IB+Hyp",
}

# Extended configs (benchmark comparison with external methods)
_BENCHMARK_ORDER: List[str] = [
    "MLP",
    "GAT",
    "GAT+IB",
    "GAT+IB+Hyp",
    "GAHIB",
]

_BENCHMARK_PALETTE: List[str] = [
    "#0072B2",  # MLP
    "#56B4E9",  # GAT
    "#E69F00",  # GAT+IB
    "#009E73",  # GAT+IB+Hyp
    "#D55E00",  # GAHIB
]

_BENCHMARK_COLORS: Dict[str, str] = OrderedDict(
    zip(_BENCHMARK_ORDER, _BENCHMARK_PALETTE)
)

# ---------------------------------------------------------------------------
# Metric definitions — DRE and LSE series
# ---------------------------------------------------------------------------
METRICS_CLUSTERING = ["NMI", "ARI", "ASW", "DAV", "CAL", "COR"]

METRICS_DRE_UMAP = [
    "DRE_umap_distance_correlation",
    "DRE_umap_Q_local",
    "DRE_umap_Q_global",
    "DRE_umap_overall_quality",
]

METRICS_DRE_TSNE = [
    "DRE_tsne_distance_correlation",
    "DRE_tsne_Q_local",
    "DRE_tsne_Q_global",
    "DRE_tsne_overall_quality",
]

METRICS_LSE = [
    "LSE_manifold_dimensionality",
    "LSE_spectral_decay_rate",
    "LSE_participation_ratio",
    "LSE_anisotropy_score",
    "LSE_noise_resilience",
    "LSE_overall_quality",
]

METRICS_DREX = [
    "DREX_trustworthiness",
    "DREX_continuity",
    "DREX_distance_spearman",
    "DREX_distance_pearson",
    "DREX_local_scale_quality",
    "DREX_neighborhood_symmetry",
    "DREX_overall_quality",
]

METRICS_LSEX = [
    "LSEX_two_hop_connectivity",
    "LSEX_radial_concentration",
    "LSEX_local_curvature",
    "LSEX_entropy_stability",
    "LSEX_overall_quality",
]

ALL_DRE_SERIES = METRICS_DRE_UMAP + METRICS_DRE_TSNE
ALL_LSE_SERIES = METRICS_LSE
ALL_METRICS = METRICS_CLUSTERING + ALL_DRE_SERIES + ALL_LSE_SERIES

# Proposed summary metrics (matching MoCoO's proposed set)
PROPOSED_CLUSTERING = ["NMI", "ARI", "ASW", "DAV"]
PROPOSED_QUALITY = [
    "DRE_umap_overall_quality",
    "LSE_overall_quality",
]
PROPOSED_METRICS = PROPOSED_CLUSTERING + PROPOSED_QUALITY

# Display labels
METRIC_DISPLAY: Dict[str, str] = {
    "NMI": "NMI", "ARI": "ARI", "ASW": "ASW",
    "CAL": "CAL", "DAV": "DAV", "COR": "COR",
    "DRE_umap_distance_correlation": "DC (UMAP)",
    "DRE_umap_Q_local": "QL (UMAP)",
    "DRE_umap_Q_global": "QG (UMAP)",
    "DRE_umap_overall_quality": "DRE (UMAP)",
    "DRE_tsne_distance_correlation": "DC (t-SNE)",
    "DRE_tsne_Q_local": "QL (t-SNE)",
    "DRE_tsne_Q_global": "QG (t-SNE)",
    "DRE_tsne_overall_quality": "DRE (t-SNE)",
    "LSE_manifold_dimensionality": "Man. dim.",
    "LSE_spectral_decay_rate": "Spec. decay",
    "LSE_participation_ratio": "Part. ratio",
    "LSE_anisotropy_score": "Anisotropy",
    "LSE_noise_resilience": "Noise res.",
    "LSE_core_quality": "Core qual.",
    "LSE_overall_quality": "LSE overall",
    "DREX_trustworthiness": "Trust.",
    "DREX_continuity": "Contin.",
    "DREX_distance_spearman": "Spearman",
    "DREX_distance_pearson": "Pearson",
    "DREX_local_scale_quality": "Local scale",
    "DREX_neighborhood_symmetry": "Neigh. sym.",
    "DREX_overall_quality": "DREX overall",
    "LSEX_two_hop_connectivity": "2-hop conn.",
    "LSEX_radial_concentration": "Radial conc.",
    "LSEX_local_curvature": "Curvature",
    "LSEX_entropy_stability": "Ent. stab.",
    "LSEX_overall_quality": "LSEX overall",
}

# Direction (True = higher is better)
METRIC_DIRECTION: Dict[str, bool] = {
    "NMI": True, "ARI": True, "ASW": True,
    "CAL": True, "DAV": False, "COR": True,
    "DRE_umap_distance_correlation": True,
    "DRE_umap_Q_local": True,
    "DRE_umap_Q_global": True,
    "DRE_umap_overall_quality": True,
    "DRE_tsne_distance_correlation": True,
    "DRE_tsne_Q_local": True,
    "DRE_tsne_Q_global": True,
    "DRE_tsne_overall_quality": True,
    "LSE_manifold_dimensionality": True,
    "LSE_spectral_decay_rate": True,
    "LSE_participation_ratio": True,
    "LSE_anisotropy_score": True,
    "LSE_noise_resilience": True,
    "LSE_core_quality": True,
    "LSE_overall_quality": True,
    "DREX_trustworthiness": True,
    "DREX_continuity": True,
    "DREX_distance_spearman": True,
    "DREX_distance_pearson": True,
    "DREX_local_scale_quality": True,
    "DREX_neighborhood_symmetry": True,
    "DREX_overall_quality": True,
    "LSEX_two_hop_connectivity": True,
    "LSEX_radial_concentration": True,
    "LSEX_local_curvature": True,
    "LSEX_entropy_stability": True,
    "LSEX_overall_quality": True,
}

# Metric glossary
METRIC_GLOSSARY: Dict[str, str] = {
    "ARI": "Adj. Rand Index",
    "NMI": "Norm. Mutual Info.",
    "ASW": "Avg. Silhouette Width",
    "CAL": "Calinski\u2013Harabasz",
    "DAV": "Davies\u2013Bouldin",
    "COR": "Pearson Corr.",
    "DRE": "Dim. Red. Eval.",
    "DREX": "Ext. Dim. Red. Eval.",
    "LSE": "Latent Space Eval.",
    "LSEX": "Ext. Latent Space Eval.",
}

# Decimal format constants
FMT_SCORE = ".3f"
FMT_SCORE_SHORT = ".2f"
FMT_LARGE = ".1f"
FMT_DELTA = "+.3f"

# Line styles per config
_LINE_STYLES: Dict[str, object] = {
    "Base VAE": "-",
    "VAE+IB": "--",
    "VAE+Hyp": "-.",
    "VAE+IB+Hyp": ":",
    "GAHIB": (0, (5, 1)),
}

_LINE_WIDTHS: Dict[str, float] = {
    c: (2.2 if c == "GAHIB" else 1.4) for c in _CONFIG_ORDER
}


# ---------------------------------------------------------------------------
# Layout helpers (matching MoCoO's absolute-geometry approach)
# ---------------------------------------------------------------------------

def place_axes(fig, rect):
    """Create an axes at an exact position (left, bottom, width, height)."""
    return fig.add_axes(rect)


def row_of_axes(fig, n, rect, gap=0.04, widths=None):
    """Distribute n axes horizontally inside rect."""
    left, bottom, total_w, height = rect
    if widths is None:
        widths = [1.0] * n
    wsum = sum(widths)
    usable = total_w - gap * (n - 1)
    axes = []
    x = left
    for w in widths:
        aw = usable * (w / wsum)
        axes.append(fig.add_axes([x, bottom, aw, height]))
        x += aw + gap
    return axes


def col_of_axes(fig, n, rect, gap=0.04, heights=None):
    """Distribute n axes vertically inside rect (top to bottom)."""
    left, bottom, width, total_h = rect
    if heights is None:
        heights = [1.0] * n
    hsum = sum(heights)
    usable = total_h - gap * (n - 1)
    axes = []
    y = bottom + total_h
    for h in heights:
        ah = usable * (h / hsum)
        y -= ah
        axes.append(fig.add_axes([left, y, width, ah]))
        y -= gap
    return axes


def grid_of_axes(fig, nrows, ncols, rect, hgap=0.04, wgap=0.04,
                 heights=None, widths=None):
    """Create a nrows x ncols grid of axes inside rect."""
    left, bottom, total_w, total_h = rect
    if heights is None:
        heights = [1.0] * nrows
    if widths is None:
        widths = [1.0] * ncols
    hsum = sum(heights)
    wsum = sum(widths)
    usable_h = total_h - hgap * (nrows - 1)
    usable_w = total_w - wgap * (ncols - 1)

    row_bottoms, row_heights = [], []
    y = bottom + total_h
    for rh in heights:
        ah = usable_h * (rh / hsum)
        y -= ah
        row_bottoms.append(y)
        row_heights.append(ah)
        y -= hgap

    col_lefts, col_widths_out = [], []
    x = left
    for cw in widths:
        aw = usable_w * (cw / wsum)
        col_lefts.append(x)
        col_widths_out.append(aw)
        x += aw + wgap

    axes = []
    for ri in range(nrows):
        row = []
        for ci in range(ncols):
            ax = fig.add_axes([col_lefts[ci], row_bottoms[ri],
                               col_widths_out[ci], row_heights[ri]])
            row.append(ax)
        axes.append(row)
    return axes


def enforce_canvas_ratio(fig_w, fig_h, min_ratio=21.0 / 17.0):
    """Ensure fig_h / fig_w >= min_ratio (default 17:21 page).

    Returns adjusted (fig_w, fig_h).
    """
    if fig_h / fig_w < min_ratio:
        fig_h = fig_w * min_ratio
    return fig_w, fig_h


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_style() -> None:
    """Apply publication-quality matplotlib rcParams matching MoCoO style."""
    params = {
        "figure.figsize": (FIG_WIDTH_IN, FIG_HEIGHT_IN),
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": None,
        "savefig.pad_inches": 0.0,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
        "savefig.transparent": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": FS_AXIS,
        "font.weight": "normal",
        "font.style": "normal",
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "normal",
        "axes.titlepad": 4.0,
        "axes.labelsize": FS_AXIS,
        "axes.labelweight": "normal",
        "axes.labelpad": 3.0,
        "axes.linewidth": 0.5,
        "axes.grid": True,
        "axes.grid.which": "major",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "figure.edgecolor": "none",
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
        "grid.alpha": 0.22,
        "grid.linestyle": "--",
        "grid.linewidth": 0.4,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": True,
        "legend.framealpha": 0.65,
        "legend.borderpad": 0.3,
        "lines.linewidth": 1.4,
        "lines.markersize": 3,
        "patch.linewidth": 0.4,
        "mathtext.default": "regular",
    }

    # Arial / Liberation Sans (metric-compatible) font setup
    params["font.sans-serif"] = [
        "Arial", "Liberation Sans", "Nimbus Sans L", "DejaVu Sans"
    ]
    params["pdf.fonttype"] = 42   # TrueType embedded, editable in Illustrator
    params["ps.fonttype"]  = 42

    try:
        import matplotlib.font_manager as fm
        available = {f.name for f in fm.fontManager.ttflist}
        preferred = [n for n in ("Arial", "Liberation Sans", "Nimbus Sans")
                     if n in available]
        if preferred:
            params["font.sans-serif"] = preferred + [
                "DejaVu Sans"
            ]
    except Exception:
        pass

    matplotlib.rcParams.update(params)


def save_figure(fig, path, **extra_kw):
    """Save figure as both PNG and PDF for publication.

    Uses fully manual geometry — no bbox_inches, no pad_inches.
    """
    from pathlib import Path as _Path

    p = _Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # PNG: lower DPI to keep file size manageable
    png_path = p.with_suffix(".png")
    fig.savefig(str(png_path), dpi=DPI_PNG)

    # PDF: no dpi argument (vector)
    pdf_path = p.with_suffix(".pdf")
    fig.savefig(str(pdf_path))

    return str(png_path), str(pdf_path)


def add_panel_label(ax, label, x=-0.08, y=1.06, fontsize=None,
                    fontweight="bold"):
    """Place a panel label (a), (b), ... outside ax. Bold by convention."""
    import matplotlib.patheffects as pe
    fs = fontsize or FS_LABEL
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        fontsize=fs, fontweight=fontweight, va="bottom", ha="left",
        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")],
    )


def get_config_colors() -> Dict[str, str]:
    return OrderedDict(_CONFIG_COLORS)


def get_config_order() -> List[str]:
    return list(_CONFIG_ORDER)


def get_display_name(config: str) -> str:
    return _DISPLAY_NAMES.get(config, config)


def get_short_name(config: str) -> str:
    return _SHORT_NAMES.get(config, config)


def get_line_style(config: str):
    return _LINE_STYLES.get(config, "-")


def get_line_width(config: str) -> float:
    return _LINE_WIDTHS.get(config, 1.4)


def metric_label(abbrev: str, include_direction: bool = True) -> str:
    """Return display label like 'ARI (Adj. Rand Index) up-arrow'."""
    arrow = ""
    if include_direction and abbrev in METRIC_DIRECTION:
        arrow = " \u2191" if METRIC_DIRECTION[abbrev] else " \u2193"
    full = METRIC_DISPLAY.get(abbrev, abbrev)
    return f"{full}{arrow}"


def metric_title(abbrev: str) -> str:
    """Short title like 'ARI up-arrow' for subplot titles."""
    arrow = ""
    if abbrev in METRIC_DIRECTION:
        arrow = " \u2191" if METRIC_DIRECTION[abbrev] else " \u2193"
    display = METRIC_DISPLAY.get(abbrev, abbrev)
    return f"{display}{arrow}"


# ---------------------------------------------------------------------------
# Layout / primitives style constants
# (requires primitives.py; imported lazily to avoid circular import)
# ---------------------------------------------------------------------------

def _make_text_style(**kw):
    """Return a TextStyle without importing at module level (avoids circulars)."""
    from .primitives import TextStyle
    return TextStyle(**kw)


def _make_box_style(**kw):
    from .primitives import BoxStyle
    return BoxStyle(**kw)


# ── Pipeline box colours keyed by stage name ──────────────────────────────
PIPELINE_COLORS: Dict[str, str] = {
    "input":    "#AED6F1",   # scRNA-seq light blue
    "preproc":  "#BDC3C7",   # kNN / preprocessing grey
    "encoder":  "#1A9E8F",   # GAT encoder teal
    "ib":       "#7D3C98",   # IB compression purple
    "lorentz":  "#4A235A",   # Lorentz loss indigo
    "decoder":  "#E67E22",   # NB decoder amber
    "latent":   "#0E6655",   # Latent embedding dark teal
}

# ── Architecture column colours (header, body) ────────────────────────────
ARCH_COLORS: Dict[str, tuple] = {
    "INPUT":                  ("#a8cef0", "#daeeff"),
    "PREPROCESSING":          ("#b0b0b0", "#ebebeb"),
    "GAT ENCODER":            ("#35978f", "#cce9e6"),
    "INFORMATION BOTTLENECK": ("#756bb1", "#e3dff2"),
    "DECODER":                ("#fdae61", "#feeacb"),
}

# ── Text styles ────────────────────────────────────────────────────────────
# (Instantiated on first access via property-like callables)

def TITLE_TEXT_STYLE() -> "TextStyle":  # type: ignore[misc]
    return _make_text_style(size=16, color="#1B2A6B", weight="bold")


def BODY_TEXT_STYLE() -> "TextStyle":  # type: ignore[misc]
    return _make_text_style(size=8.5, color="#222222")


def CAPTION_TEXT_STYLE() -> "TextStyle":  # type: ignore[misc]
    return _make_text_style(size=7.0, color="#444444", italic=True)


def HEADER_TEXT_STYLE() -> "TextStyle":  # type: ignore[misc]
    return _make_text_style(size=9.5, color="#111111", weight="bold")


def SMALL_TEXT_STYLE() -> "TextStyle":  # type: ignore[misc]
    return _make_text_style(size=7.5, color="#333333")


def WHITE_TEXT_STYLE(size: float = 8.5) -> "TextStyle":  # type: ignore[misc]
    return _make_text_style(size=size, color="white", weight="bold")


# ── Box styles ─────────────────────────────────────────────────────────────

def DEFAULT_BOX_STYLE() -> "BoxStyle":  # type: ignore[misc]
    return _make_box_style(face="#ffffff", edge="#333333", lw=0.8)


def PANEL_BOX_STYLE() -> "BoxStyle":  # type: ignore[misc]
    return _make_box_style(face="#EEF2F5", edge="#9FB3C8", lw=1.2, rounding=0.015)
