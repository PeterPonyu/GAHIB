# GAHIB Public-Code Revision Scope for Frontiers Formal Revision

Created: 2026-05-15  
Purpose: keep `GAHIB/` focused on publication-ready public code while formal reviewer-response assets remain in `GAHIB-assets/`.

## Boundary

`GAHIB/` should be the public reproducibility/code-facing folder. It should not contain raw/local datasets, confidential reviewer correspondence, private generated assets, or placeholder final submission PDFs.

Formal reviewer comment register and point-by-point response work should live in:

- `../GAHIB-assets/revision/99_notes/frontiers_actual_reviews_2026-05-15.md`
- `../GAHIB-assets/revision/99_notes/frontiers_revision_action_matrix_2026-05-15.md`

## Public deliverables to harden

1. Installation and quick-start instructions.
2. Public-safe dataset/accession manifest and preprocessing description.
3. Reproducibility commands for headline benchmark/case-study outputs, using public or synthetic/minimal data where possible.
4. Environment/dependency lock.
5. Tests and smoke checks.
6. DOI/archive checklist for a public code release.
7. Clear statement that full generated figures/manuscript/reviewer assets are maintained in `GAHIB-assets/`.

## Reviewer comments that affect public code

- R2 asks for release/code/accessions discipline to remain clear and complete.
- R2 asks for scGAC/SCEA/scVAG lineage and ideally additional baselines; if benchmarked, scripts/configs should be public-safe.
- R1/R2 dataset provenance, QC, units, and normalization details should be reflected in public docs where they do not expose private/local data.
- Dropout robustness, k sensitivity, and cost ANOVA scripts should be made reproducible or summarized with public-safe commands once finalized.

## Verification before public release

```bash
git status --short
pytest -q
python -m compileall gahib experiments
rg -n "/ho[m]e/|Downloa[d]s|passwor[d]|toke[n]|secre[t]|PLACEHOLDER|TODO final" .
```

Any expected local paths in documentation must be marked as examples or replaced with accession-based instructions.

## 2026-05-15 public hardening notes

Worker-4 public-code lane changes should remain limited to public docs,
reproducibility commands, and executable scripts that must not contain private
workstation paths.

Current public-safe additions:

- `docs/DATA.md` documents `.h5ad` input expectations, `GAHIB_DATASET_DIRS`,
  and release-time data/accession checks.
- `docs/REPRODUCIBILITY.md` collects installation, smoke-test, full-test, and
  benchmark commands using repo-relative paths.
- `experiments/run_parallel_group_A.sh`, `experiments/run_parallel_group_B.sh`,
  and `experiments/run_all_expanded.sh` use their own script location plus
  `GAHIB_PYTHON`/`python` rather than hardcoded private interpreter or repo
  paths.

Recommended audit command remains:

```bash
rg -n --hidden -g '!*.pdf' -g '!*.png' -g '!*.jpg' -g '!.git/**' -g '!.venv/**' \
  '/ho[m]e/|/Us[e]rs/|Downloa[d]s|passwor[d]|toke[n]|secre[t]|confiden[t]ial|frontiers_actual_reviews' .
```

Expected hits should be limited to this public scope note, documented example
placeholders, or intentionally segregated revision placeholders. Public runner
scripts should have zero hardcoded `/home/...` paths.
