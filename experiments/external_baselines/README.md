# External online baseline adapters

This directory intentionally contains **adapters only**, not vendored third-party
baseline code.

Reviewer-requested graph-attention baselines:

| Method | Primary source checkout | Commit verified 2026-05-16 | License | Run policy |
| --- | --- | --- | --- | --- |
| scGAC | `https://github.com/Joye9285/scGAC.git` | `893909a7e46e0ed2e52fd1800f2c1606c4cf447a` | GPL-3.0 | Set `GAHIB_SCGAC_DIR` or `GAHIB_ONLINE_BASELINE_ROOT/scGAC`. |
| SCEA | `https://github.com/SAkbari93/SCEA.git` | `a13ef014f25843a53ba720b969d4b57be06d9416` | GPL-3.0 | Set `GAHIB_SCEA_DIR` or `GAHIB_ONLINE_BASELINE_ROOT/SCEA`. |
| scVAG | `https://github.com/pourialaghayee/scVAG.git` | `a0eb8151849e251f11f6859a1e4867a6f5367360` | MIT | Set `GAHIB_SCVAG_DIR` or `GAHIB_ONLINE_BASELINE_ROOT/scVAG`. |

The adapters preserve the normal GAHIB benchmark policy for dataset discovery,
HVG filtering, max-cell cap, random seed, Leiden-derived evaluation labels, and
CSV output schema. They export the already-filtered raw-count layer because the
source implementations request raw-count `data.tsv` input. Cell-type or other
`.obs` label annotations are not serialized into the export TSV; labels stay
in-memory and are only used for the benchmark's unsupervised cluster-count and
evaluation steps.

## Label-use / leakage policy

Benchmark labels are post-hoc Leiden/evaluation labels, not supervised training
targets. The transparent PyTorch `*-style` routes consume the provided label
iterable only to infer the requested number of groups for unsupervised
cluster-count-compatible losses. Label strings and per-cell assignments are not
exported to the external data TSV, not used for graph construction, and not used
in a classification or cross-entropy objective. The regression tests in
`tests/test_online_graph_attention_adapters.py` verify latent invariance under
label-name/per-cell reassignment and assert that sentinel label values are not
consumed after cluster-count inference or serialized in adapter outputs.

If a checkout or compatible legacy TensorFlow/Keras stack is unavailable, the
benchmark writes a `*_external_status.csv` row with `status=not_runnable` and an
exact reason. Empty metric rows in `*_df.csv` must therefore be interpreted as
missing/not-runnable, not as failed performance values.
