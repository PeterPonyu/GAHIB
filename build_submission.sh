#!/usr/bin/env bash
# ============================================================================
# build_submission.sh — assemble the GAHIB journal-submission bundles
#
# Sources (editable, kept locally):
#   paper/              — master manuscript (with author information)
#   paper_blind/        — double-anonymous version for review
#   cover_letter/       — cover letter (retargetable; see cover_letter.tex)
#   highlights/         — research highlights
#                         (uses \ifblind toggle for anonymous vs. authored)
#
# Outputs (regenerated each run — safe to delete):
#   GAHIB_submission/            blind bundle (double-anonymous review copy)
#   GAHIB_submission_authored/   non-blind bundle (camera-ready copy)
#
# Both bundles renumber figures to figure1.pdf … figureN.pdf and rewrite
# the \includegraphics{...} references in the .tex accordingly, so no
# internal figure filenames (e.g. gahib/viz/*.py-derived names) leak.
#
# Usage:
#   ./build_submission.sh                 build both bundles
#   ./build_submission.sh --blind         build only the anonymous bundle
#   ./build_submission.sh --authored      build only the non-blind bundle
#   ./build_submission.sh --clean         wipe LaTeX artefacts first
#   ./build_submission.sh --zip           also zip each bundle
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLIND_BUNDLE="$ROOT/GAHIB_submission"
AUTHORED_BUNDLE="$ROOT/GAHIB_submission_authored"

# Figure ordering reflects the appearance order in the manuscript and
# drives the figure1 … figureN rename.
FIG_NAMES=(
  overview                     # 1
  architecture                 # 2
  dataset_taxonomy             # 3
  ablation_all_metrics         # 4
  scdeep_all_metrics           # 5
  classical_all_metrics        # 6
  gmvae_all_metrics            # 7
  disent_all_metrics           # 8
  encoder_all_metrics          # 9
  gconv_all_metrics            # 10
  interp_embedding_poincare    # 11
  interp_bottleneck            # 12
  interp_hyperbolic            # 13
  fig_downstream_analysis      # 14
  interp_summary               # 15
  latent_dim_and_seed          # 16
  computational_cost           # 17
  hyperparam_sensitivity       # 18
)

DO_BLIND=1
DO_AUTHORED=1
DO_CLEAN=0
DO_ZIP=0
for arg in "$@"; do
  case "$arg" in
    --blind)    DO_AUTHORED=0 ;;
    --authored) DO_BLIND=0 ;;
    --clean)    DO_CLEAN=1 ;;
    --zip)      DO_ZIP=1 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

ARTEFACTS=(-name '*.aux' -o -name '*.log' -o -name '*.fls' \
           -o -name '*.fdb_latexmk' -o -name '*.out' -o -name '*.spl' \
           -o -name '*.bbl' -o -name '*.blg' -o -name '*.synctex.gz')

clean_artefacts() {
  find "$ROOT/paper" "$ROOT/paper_blind" "$ROOT/cover_letter" \
       "$ROOT/highlights" "$BLIND_BUNDLE" "$AUTHORED_BUNDLE" \
       \( "${ARTEFACTS[@]}" \) -delete 2>/dev/null || true
}

# Run latexmk quietly, escalating to verbose on error.
latexmk_build() {
  local dir="$1" main="$2"
  (cd "$dir" && latexmk -pdf -interaction=nonstopmode -quiet "$main" \
       >/dev/null 2>&1) || \
  (cd "$dir" && latexmk -pdf -interaction=nonstopmode "$main")
}

# Copy a figure file (pdf/png) from src_dir to dst_dir under a new name.
copy_figure() {
  local src_dir="$1" src_name="$2" dst_dir="$3" dst_name="$4"
  for ext in pdf png; do
    if [[ -f "$src_dir/$src_name.$ext" ]]; then
      cp "$src_dir/$src_name.$ext" "$dst_dir/$dst_name.$ext"
      return 0
    fi
  done
  echo "warn: figure $src_name not found in $src_dir" >&2
  return 1
}

# Rewrite \includegraphics{foo} → \includegraphics{figureN} inside tex_file.
renumber_figures_in_tex() {
  local tex_file="$1"
  local i name tmp
  tmp="$(mktemp)"
  cp "$tex_file" "$tmp"
  for i in "${!FIG_NAMES[@]}"; do
    name="${FIG_NAMES[$i]}"
    local n=$((i + 1))
    # Match \includegraphics[...]{name} or \includegraphics{name} exactly.
    sed -i "s#\\\\includegraphics\\(\\[[^]]*\\]\\)\\?{${name}}#\\\\includegraphics\\1{figure${n}}#g" "$tmp"
  done
  mv "$tmp" "$tex_file"
}

build_bundle() {
  # $1 = mode (blind|authored)
  # $2 = paper source dir (paper_blind or paper)
  # $3 = paper source tex (gahib_paper_blind.tex or gahib_paper.tex)
  # $4 = bundle output dir
  local mode="$1" src_dir="$2" src_tex="$3" bundle="$4"

  echo "[bundle:$mode] $bundle"

  rm -rf "$bundle"
  mkdir -p "$bundle/manuscript" "$bundle/cover_letter" "$bundle/highlights"

  # 1. Paper: copy tex + references, renumber figures, copy renumbered PDFs.
  cp "$src_dir/$src_tex"     "$bundle/manuscript/$src_tex"
  cp "$src_dir/references.bib" "$bundle/manuscript/references.bib" 2>/dev/null \
    || cp "$ROOT/paper/references.bib" "$bundle/manuscript/references.bib"

  # Figures live in paper/figures for the authored copy and directly in
  # paper_blind/ for the blind copy — try both.
  local fig_src
  if [[ "$mode" == "authored" ]]; then
    fig_src="$ROOT/paper/figures"
  else
    fig_src="$ROOT/paper_blind"
  fi

  local i
  for i in "${!FIG_NAMES[@]}"; do
    local n=$((i + 1))
    copy_figure "$fig_src" "${FIG_NAMES[$i]}" "$bundle/manuscript" "figure${n}"
  done

  renumber_figures_in_tex "$bundle/manuscript/$src_tex"
  # Force \graphicspath to look in the current directory only.
  sed -i 's#\\graphicspath{{figures/}}#\\graphicspath{{./}}#' \
    "$bundle/manuscript/$src_tex"

  latexmk_build "$bundle/manuscript" "$src_tex"

  # 2. Cover letter.
  cp "$ROOT/cover_letter/cover_letter.tex" "$bundle/cover_letter/"
  [[ -f "$ROOT/cover_letter/graphical_abstract.pdf" ]] && \
    cp "$ROOT/cover_letter/graphical_abstract.pdf" "$bundle/cover_letter/"
  latexmk_build "$bundle/cover_letter" "cover_letter.tex"

  # 3. Highlights — toggle \blindfalse → \blindtrue when building blind.
  cp "$ROOT/highlights/highlights.tex" "$bundle/highlights/"
  if [[ "$mode" == "blind" ]]; then
    sed -i 's#\\blindfalse#\\blindtrue#' "$bundle/highlights/highlights.tex"
  fi
  latexmk_build "$bundle/highlights" "highlights.tex"

  # 4. Strip LaTeX build artefacts from the bundle; keep only tex/pdf/bib.
  find "$bundle" \( "${ARTEFACTS[@]}" \) -delete 2>/dev/null || true

  # 5. Optional zip.
  if [[ "$DO_ZIP" -eq 1 ]]; then
    (cd "$ROOT" && rm -f "${bundle}.zip" && \
     zip -r "${bundle}.zip" "$(basename "$bundle")" >/dev/null)
    echo "[zip] ${bundle}.zip"
  fi
}

[[ "$DO_CLEAN" -eq 1 ]] && { echo "[clean]"; clean_artefacts; }

if [[ "$DO_BLIND" -eq 1 ]]; then
  build_bundle blind    "$ROOT/paper_blind" gahib_paper_blind.tex \
    "$BLIND_BUNDLE"
fi

if [[ "$DO_AUTHORED" -eq 1 ]]; then
  build_bundle authored "$ROOT/paper"       gahib_paper.tex \
    "$AUTHORED_BUNDLE"
fi

echo "[done]"
