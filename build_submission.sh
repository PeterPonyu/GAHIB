#!/usr/bin/env bash
# ============================================================================
# build_submission.sh — rebuild the GAHIB journal-submission bundle
#
# Sources:
#   paper/              — master manuscript (with author info)
#   paper_blind/        — double-anonymous version for review
#   cover_letter/       — cover letter (retargetable; see cover_letter.tex)
#   highlights/         — research highlights / key questions
#
# Output:
#   GAHIB_submission/   — packaged copy (tex + pdf) ready for journal upload
#   GAHIB_submission/manuscript_blind.zip (optional, pass --zip)
#
# Usage:
#   ./build_submission.sh           rebuild PDFs and sync GAHIB_submission/
#   ./build_submission.sh --clean   remove LaTeX build artefacts first
#   ./build_submission.sh --zip     also produce manuscript_blind.zip
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE="$ROOT/GAHIB_submission"

CLEAN=0
ZIP=0
for arg in "$@"; do
  case "$arg" in
    --clean) CLEAN=1 ;;
    --zip)   ZIP=1 ;;
    -h|--help)
      sed -n '2,17p' "$0"
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
       "$ROOT/highlights" "$BUNDLE" \
       \( "${ARTEFACTS[@]}" \) -delete 2>/dev/null || true
}

build() {
  local dir="$1" main="$2"
  echo "[build] $dir/$main"
  (cd "$dir" && latexmk -pdf -interaction=nonstopmode -quiet "$main" \
       >/dev/null 2>&1 || latexmk -pdf -interaction=nonstopmode "$main")
}

if [[ "$CLEAN" -eq 1 ]]; then
  echo "[clean] removing LaTeX build artefacts"
  clean_artefacts
fi

# 1. Build the four sources (paper/ is authored master; the others get
#    synced into the bundle).
build "$ROOT/paper"         gahib_paper.tex
build "$ROOT/paper_blind"   gahib_paper_blind.tex
build "$ROOT/cover_letter"  cover_letter.tex
build "$ROOT/highlights"    highlights.tex

# 2. Sync sources + PDFs into the bundle.
mkdir -p "$BUNDLE/manuscript_blind" "$BUNDLE/cover_letter" "$BUNDLE/highlights"

cp "$ROOT/paper_blind/gahib_paper_blind.tex" \
   "$ROOT/paper_blind/gahib_paper_blind.pdf" \
   "$ROOT/paper_blind/references.bib" \
   "$BUNDLE/manuscript_blind/" 2>/dev/null || true

# Copy blind-paper figures (pdfs only, skip LaTeX build artefacts).
find "$ROOT/paper_blind" -maxdepth 1 -type f -name '*.pdf' \
     ! -name 'gahib_paper_blind.pdf' -exec cp {} "$BUNDLE/manuscript_blind/" \;

cp "$ROOT/cover_letter/cover_letter.tex" \
   "$ROOT/cover_letter/cover_letter.pdf" \
   "$BUNDLE/cover_letter/"
# Include graphical abstract if present.
[[ -f "$ROOT/cover_letter/graphical_abstract.pdf" ]] && \
  cp "$ROOT/cover_letter/graphical_abstract.pdf" "$BUNDLE/cover_letter/"

cp "$ROOT/highlights/highlights.tex" \
   "$ROOT/highlights/highlights.pdf" \
   "$BUNDLE/highlights/"

# 3. Optional: zip the blind-manuscript folder for one-click upload.
if [[ "$ZIP" -eq 1 ]]; then
  (cd "$BUNDLE" && rm -f manuscript_blind.zip && \
   zip -r manuscript_blind.zip manuscript_blind >/dev/null)
  echo "[zip] $BUNDLE/manuscript_blind.zip"
fi

# 4. Cleanup LaTeX build artefacts in the bundle (only keep tex+pdf+bib).
find "$BUNDLE" \( "${ARTEFACTS[@]}" \) -delete 2>/dev/null || true

echo "[done] submission bundle at: $BUNDLE"
