# Potential Reviewer Comments for GAHIB Manuscript

Created: 2026-04-29
Source reviewed: `GAHIB_submission_authored/manuscript/gahib_paper.tex`
Status: anticipated reviewer concerns only. These are **not** actual received reviews and **not** drafted rebuttal answers.

## Executive triage

### Highest-priority likely major comments

1. **Potential circularity / proxy-label bias in evaluation.** The manuscript evaluates clustering against Leiden labels derived from the same PCA/neighbor graph family used during preprocessing, while GAHIB itself uses a graph encoder built from a PCA-derived kNN graph. Reviewers may ask whether graph-aware methods are being advantaged by the evaluation target.
2. **Biological validation is still partly indirect.** The paper uses many metrics and a large dataset count, but manually curated cell-type/trajectory annotations appear limited or secondary. Reviewers may request stronger validation on curated annotations, known trajectories, or external biological labels.
3. **Benchmark fairness and baseline configuration.** The statement that no per-method hyperparameter search was performed may invite questions about whether baseline methods were under-tuned, especially because some published deep methods appear to perform extremely poorly.
4. **Scalability evidence is incomplete for atlas-level claims.** The evaluation caps datasets at 3,000 cells and scaling tests only reach 3,000 cells, while the manuscript discusses atlas-scale extension and flat runtime. Reviewers may ask for larger-cell experiments or more restrained claims.
5. **Some method/result descriptions appear internally inconsistent.** For example, the method section describes DPT-based validation, while the results section says pseudotime is validated directly from gene expression rather than against DPT. The GAT description says single-head attention, while the limitations mention multi-head GAT message passing.

## Potential major reviewer comments

### 1. Evaluation target may be circular or biased toward graph-based methods

**Potential comment:** The primary clustering metrics compare K-means assignments in latent space against Leiden clusters that are themselves computed from PCA-to-neighbor/UMAP-style preprocessing. Because GAHIB's encoder also consumes a kNN graph derived from PCA/neighborhood structure, this may partially reward recovery of the preprocessing graph rather than independent biological truth.

**Evidence in manuscript:**

- GAHIB constructs the graph from Scanpy neighborhoods on a 50-D PCA embedding with Euclidean distance and UMAP-style connectivity weights (`gahib_paper.tex` lines 424-427).
- The 53-dataset reference partition is Leiden clustering on PCA-to-UMAP neighbors (`lines 615-623`).
- K-means clusters in latent space are compared against the Leiden reference (`lines 632-636`).
- The limitations acknowledge that Leiden labels may not perfectly capture validated cell types (`lines 2000-2005`).

**Why a reviewer may care:** A graph encoder may be advantaged if the target labels and input graph share construction assumptions. The reviewer may ask for evaluation against curated labels where available, or sensitivity analyses using alternative reference partitions.

### 2. Baselines may be under-tuned or unevenly configured

**Potential comment:** The manuscript reports many baselines but states that no per-method hyperparameter search was performed. Reviewers may question whether default hyperparameters are fair across diverse datasets and whether poor performance of some methods reflects configuration mismatch rather than true inferiority.

**Evidence in manuscript:**

- All baselines use default hyperparameters as published and no per-method hyperparameter search was performed (`lines 689-697`).
- Some deep-learning baselines report very low aggregate NMI/ARI, e.g. CellBLAST and SCALEX in the deep-learning benchmark (`lines 870-897`).
- scVI is not significantly different from GAHIB on NMI and ARI (`lines 849-852`), despite broader language about GAHIB ranking first.

**Likely reviewer request:** Include a tuning sensitivity for the strongest baselines, at least scVI and scDHMap; explain why CellBLAST/SCALEX/scGNN configurations are valid; temper claims where differences are not significant.

### 3. The biological ground truth is not strong enough for all claimed biological conclusions

**Potential comment:** The manuscript argues that GAHIB learns biologically structured and hierarchical representations, but many validations depend on Leiden clusters, stemness proxies, GO enrichment, or a subset of datasets. Reviewers may request manually curated cell-type labels, known lineage annotations, perturbation/trajectory benchmarks, or comparison to established pseudotime methods.

**Evidence in manuscript:**

- Stemness score correlation can use external developmental annotations when available or Leiden cluster ordering (`lines 521-527`).
- Biological interpretation is described on representative datasets for which annotations are available (`lines 314-319`).
- GO analysis is performed on four representative datasets (`lines 1436-1445`).
- The limitations acknowledge the need for manually curated annotations (`lines 2000-2005`).

**Likely reviewer request:** Separate results based on true biological annotations from results based on Leiden-derived proxies; report curated-label performance where possible; include an explicit table of which datasets have external labels/lineage annotations.

### 4. Pseudotime validation appears inconsistent between methods and results

**Potential comment:** The method section defines Lorentz-norm pseudotime validation against Scanpy diffusion pseudotime, but the results section says the authors validate directly from original gene expression rather than against an algorithmic reference. This makes the validation protocol unclear.

**Evidence in manuscript:**

- Method: Lorentz norm is validated against DPT via Spearman and Kendall correlations, with Euclidean PCA/UMAP baselines (`lines 541-550`).
- Results: The manuscript says it does not evaluate against DPT and instead validates via gene-expression correlations (`lines 1500-1521`).

**Likely reviewer request:** Resolve the discrepancy, report both validation strategies if both were run, or remove the unused protocol.

### 5. Scalability claims may exceed the tested regime

**Potential comment:** The paper evaluates large biological diversity but caps each dataset at 3,000 cells and tests scaling only up to 3,000 cells. Claims about headroom for atlas-size datasets may be seen as speculative without experiments at tens or hundreds of thousands of cells.

**Evidence in manuscript:**

- Datasets are subsampled to at most 3,000 cells (`lines 615-617`).
- Cell-count scaling tests use 500, 1,000, 2,000, and 3,000 cells (`lines 1675-1678`).
- Limitations say atlas-level scaling would require future graph-sampling strategies (`lines 1980-1989`).
- Discussion says there is headroom to scale to atlas-size datasets (`lines 1961-1968`).

**Likely reviewer request:** Add a larger-cell pilot experiment, reduce atlas-scale language, or explicitly frame atlas scaling as future work only.

### 6. Statistical testing may not fully address dependence and multiple-testing structure

**Potential comment:** The manuscript uses paired Wilcoxon signed-rank tests and BH correction within each table, but the evaluation includes many metrics, datasets from related studies, and repeated method comparisons. Reviewers may ask whether the statistical unit and correction family are appropriate.

**Evidence in manuscript:**

- Wilcoxon testing is across 53 datasets (`lines 662-670`).
- BH correction is applied within each results table, typically 30-100 hypotheses (`lines 675-687`).
- The authors acknowledge that some datasets share parent studies and are not strictly independent (`lines 699-708`).

**Likely reviewer request:** Include effect sizes and confidence intervals, nested/mixed-effect sensitivity analyses, or dataset-family-aware aggregation.

### 7. The interpretation pipeline needs clearer controls

**Potential comment:** GO enrichment, decoder Jacobian attribution, and attention homophily are promising but may need stronger controls. Reviewers may ask whether GO terms remain significant with appropriate gene-universe selection, permutation/null latent dimensions, label shuffling, or comparisons to other attribution methods.

**Evidence in manuscript:**

- GO uses top-100 genes per latent dimension and a curated built-in panel with hypergeometric testing (`lines 552-564`, `1436-1445`).
- Nineteen of 53 datasets produce no enriched terms at BH-adjusted p<0.05 (`lines 1550-1554`).
- Attention homophily uses Leiden cluster labels (`lines 566-573`).

**Likely reviewer request:** Define the gene universe/background, add permutation controls, show negative controls, and avoid over-claiming tissue specificity from only four GO case studies.

### 8. Some claims should be tempered where GAHIB is balanced rather than uniformly superior

**Potential comment:** The paper's strongest claim is that GAHIB provides a balanced embedding, not that it dominates every metric. Reviewers may push back on broad superiority language because scVI is comparable on NMI/ARI, scDHMap is comparable on DRE, PCA can be competitive on linear datasets, and smaller latent dimension may maximize some clustering metrics.

**Evidence in manuscript:**

- scVI is not significantly different from GAHIB on NMI and ARI (`lines 849-852`).
- scDHMap is not significantly different on DRE-UMAP (`lines 849-852`).
- PCA is acknowledged as near-identical on several linear datasets (`lines 1991-1998`).
- Latent dimension d=3 peaks for NMI/ASW, but d=10 is retained for downstream capacity (`lines 1630-1639`).

**Likely reviewer request:** Rephrase the central contribution around multi-objective balance and interpretability, and make metric-specific caveats more visible in the abstract/discussion.

### 9. Architecture details need reconciliation and more implementation specificity

**Potential comment:** Some architectural descriptions may be underspecified or inconsistent. Reviewers may ask for exact graph construction, normalization, loss scaling, numerical stability of Lorentz mapping, and whether the GAT is single-head or multi-head.

**Evidence in manuscript:**

- GAT paragraph specifies a single attention head (`lines 424-430`).
- Limitations refer to multi-head GAT message passing (`lines 1972-1975`).
- Hyperbolic loss weight is high relative to other terms (`lines 476-490`), so reviewers may ask about loss magnitudes and normalization.

**Likely reviewer request:** Add implementation details or supplement: exact PyTorch/PyG modules, graph recomputation policy, normalization of losses, numerical clipping for hyperbolic operations, and ablation of loss weights.

### 10. Reproducibility package may be considered incomplete

**Potential comment:** The code availability section says the library, runners, and tests are public, but visualization pipeline and manuscript sources are available on request. Reviewers may ask that all scripts needed to reproduce figures/tables be released or archived.

**Evidence in manuscript:**

- Core library, experiment runners, and tests are public, while visualization pipeline and manuscript sources are maintained locally and available on request (`lines 2052-2058`).

**Likely reviewer request:** Release figure-generation scripts, exact environment files, processed metadata manifests, and result tables used for all manuscript figures.

## Potential minor reviewer comments

1. **Clarify title terminology.** The title combines GAT, VAE, hyperbolic, and information bottleneck; reviewers may ask for a shorter title or a clearer first occurrence of GAHIB in the abstract.
2. **Reduce density of acronyms and metric families.** The manuscript uses many metric abbreviations (DRE, LSE, ASW, DAV, CAL, COR). A reviewer may ask for a summary table defining each metric and its directionality.
3. **Figure overload.** The submission has many figures/tables. Reviewers may suggest moving some benchmark sweeps or all-20-metric plots to supplementary material.
4. **Report confidence intervals or effect sizes.** Mean±std plus p-values may not be enough for practical significance.
5. **Make dataset provenance easier to audit.** Add a table mapping all 53 datasets to accession, tissue, species, cancer/development category, cell count before/after subsampling, and annotation availability.
6. **Clarify whether labels are used during training.** The manuscript says unsupervised, but interpretation metrics use Leiden/cell-type labels post hoc. Make this boundary explicit.
7. **Check wording around 'disentangled'.** The manuscript critiques disentanglement regularizers but also says latent dimensions are functionally specialised/disentangled. Reviewers may ask for careful terminology to avoid contradiction.
8. **Explain why IB dimension is fixed at 2.** Since hyperbolic visualization and bottleneck geometry depend on the 2D coordinate, reviewers may ask whether 2D is chosen for visualization convenience or validated empirically.
9. **Clarify library-size handling in the NB decoder.** The method says NB captures library-size variation, but reviewers may ask for the exact offset/size-factor parameterization.
10. **Strengthen limitations.** The limitations already mention labels and scalability; reviewers may appreciate explicit discussion of graph-construction bias, preprocessing dependence, and benchmark default-parameter limitations.

## Potential submission-package / revision-package comments

1. **Tracked-changes requirement.** The revision package should eventually include a real tracked-changes PDF, not a placeholder. The current workspace correctly uses a placeholder note only.
2. **Cover letter vs rebuttal separation.** Keep the revision cover letter high-level and put detailed responses in the PP rebuttal/response-to-reviewers file.
3. **Baseline provenance.** The baseline snapshot should remain immutable; include the original submission date/manuscript ID once known.
4. **Filename clarity.** Use final filenames that distinguish clean revised manuscript, tracked-changes manuscript, cover letter, response letter, and source package.
5. **Journal compliance.** Before upload, check whether the venue wants highlights, graphical abstract, source files, anonymized/blind files, line numbers, or separate figure files.

## Suggested revision priorities before drafting rebuttal answers

1. Resolve internal inconsistencies: DPT vs gene-expression pseudotime validation; single-head vs multi-head GAT language.
2. Add or foreground curated-label validation where available.
3. Add a table that distinguishes external annotations from Leiden-derived proxies.
4. Temper claims where differences versus scVI/scDHMap/PCA are non-significant or context-dependent.
5. Add baseline-tuning and/or sensitivity justification for the strongest competing methods.
6. Clarify reproducibility assets and release figure-generation/result aggregation scripts if possible.
7. Add stronger negative/null controls for interpretation analyses.
8. Reframe scalability claims around the actually tested 3,000-cell regime unless a larger pilot is added.
