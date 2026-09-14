# locusblend (development)

`locusblend` is the package refactor of LocusBlend: the single-file Streamlit
application is being split into a reusable Python package (`src/locusblend/`)
designed to power both the web app and a standalone Python API. This repository is
**development-stage software**: the API exists and works for the internal 1000G
workflow, but it is not published on PyPI, has no archival (Zenodo) release,
and its public surface is not stable yet.

What is intentionally unchanged: algorithms, plotting behavior, LD logic, PLINK
handling, SNP matching, `COLOR_MAPPING`, chromosome X handling, input parsing
and auto-index selection all follow `app.2.8.12.py`.

## Status

| Pass | Scope | State |
| --- | --- | --- |
| 1 | Extract Streamlit-free modules (io, reference, variants, ld, genes, plotting, compare, export, config, models) | done |
| 2 | Standalone pipeline + public `locusblend.plot` for the internal local 1000G reference | done |
| later | Streamlit app rewritten on top of the package; uploaded-LD workflow in the public API; publication-page export | not started |

## Immutable baseline

`app.2.8.12.py` is the current stable LocusBlend Streamlit application and the
**sole source of truth for existing behavior**. It must not be modified,
cleaned up, reformatted, renamed or patched; the extracted modules are compared
against it and `tests/test_baseline.py` pins its exact content. The Streamlit
app has *not* been converted to use the package yet.

## Installation (development)

```bash
git clone <your-fork-or-clone-of-this-repository>
cd locusblend_package
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[export]"   # pillow + kaleido for PNG/PDF rendering
python -m pip install -e ".[genes]"    # pyarrow, only for parquet gene tables
python -m pip install -e ".[recomb]"   # pyBigWig, only for the recombination BigWig
python -m pip install -e ".[dev]"      # pytest
```

Streamlit is deliberately **not** a dependency of the package.

## Input requirements

* **Genome build: GRCh38 / hg38.** No liftover is performed; hg19/hg37 or hg18
  coordinates must be lifted over before calling the API.
* **Required columns:** `CHR`, `BP`, `P`, `A1`, `A2`.
* **Recommended columns:** `rsid`, `BETA`, `SE`, `A1FREQ`, `N`.
* **Accepted file formats:** `.csv`, `.tsv`, `.txt`, `.csv.gz`, `.tsv.gz`,
  `.txt.gz` (same detection/parsing as the web app). Files may also be passed
  as already-loaded `pandas.DataFrame` objects.
* **Accepted column aliases** (case-insensitive): chromosome `CHR`, `chr`,
  `#chr`, `chrom`, `chromosome`; position `BP`, `bp`, `pos`, `position`,
  `base_pair_location`; p-value `P`, `p`, `pval`, `pvalue`, `p_value`,
  `P-value`, `P_VALUE`; variant id `rsid`, `RSID`, `SNP`, `MarkerName`, `ID`;
  effect allele `A1`, `EA`, `effect_allele`, `ALLELE1`; other allele `A2`,
  `NEA`, `other_allele`, `non_effect_allele`, `ALLELE0`; effect size `BETA`,
  `beta`, `Effect`, `estimate`; standard error `SE`, `StdErr`, `stderr`,
  `standard_error`; frequency `A1FREQ`, `EAF`, `MAF`, `freq`; sample size `N`,
  `n`, `N_incl`, `n_incl`, `samplesize`, `sample_size`.
* **Chromosomes:** 1-22 and X. Chromosome X may be given as `X`, `chrX`, `23`
  or `chr23`; it is normalized to `X` internally, and X allele ids use the same
  `CHR:BP:A1:A2` pattern as autosomes.

## External reference data (never bundled)

Large reference files are not part of this repository and are never committed.
They live outside it and are resolved centrally by
`locusblend.reference.ReferenceManager`:

```text
reference_dir/
├── 1000g/
│   ├── AFR/
│   ├── AMR/
│   ├── EAS/
│   ├── EUR/
│   └── SAS/
├── gencode/
└── recombination/
```

The file names inside those directories follow the baseline app's naming
(`1000g_<ANCESTRY>_hg38_high_coverage_..._ch<chrom>.bed/.bim/.fam`,
`gencode.v49.annotation.chr<chrom>.gtf.gz`,
`gencode.v49.annotation.gtf.gz`, `recomb1000GAvg.bw`); the naming templates are
class attributes of `ReferenceManager` so they can be adjusted in one place.
`LOCUSBLEND_REFERENCE_DIR` supplies the default reference directory.

```python
from locusblend import ReferenceManager

ref = ReferenceManager(reference_dir="D:/locusblend_reference", ancestry="EUR")
bfile_prefix = ref.get_bfile_prefix("14")      # verifies .bed/.bim/.fam
gtf_path = ref.get_gtf_path("14")              # chrX falls back to the genome GTF
bw_path = ref.get_recombination_bw_path()      # None when the file is missing
```

## PLINK requirement

PLINK is required for LD computation and for automatic index-variant selection
against the internal 1000G reference. It is **not bundled**. Resolution order:

1. an explicit `plink_path=...`
2. the `LOCUSBLEND_PLINK` environment variable
3. the system `PATH` (the executable must be named `plink`)

If PLINK cannot be found, the API raises a `FileNotFoundError` that names these
options.

## Python API

```python
import locusblend

result = locusblend.plot(
    dataset1="trait1.tsv.gz",
    dataset2="trait2.tsv.gz",
    reference_dir=r"D:\locusblend_reference",
    ancestry="EUR",
    mode="three",
    output="locusblend.png",
)

result.locus_figure      # combined three-row locus figure (dataset1 / dataset2 / GENCODE)
result.compare_figure    # locus compare figure (separate panels per active index variant)
result.index_variants    # index variant labels used for the plot
result.ld_status         # LD source, per-variant reference status, caption
result.dataset1_processed
result.metadata
```

`locusblend.plot` reads both datasets with the web app's format detection,
infers `chrom`/`center_bp` when they are not supplied, resolves the 1000G PLINK
bfile / GENCODE annotation / recombination track through `ReferenceManager`,
reference-matches both datasets (two-pass forward / allele-flip matching),
computes LD with PLINK inside the locus window, builds both locus panels plus
the GENCODE gene track, assembles the same three-row combined figure and the
locus compare figure as `app.2.8.12.py`, and optionally writes the combined
locus figure to a PNG.

### Modes

| `mode` | Baseline mode | Index variants |
| --- | --- | --- |
| `"standard"` | Standard locus zoom | 1 |
| `"two"` | Two-index LocusBlend | 2 |
| `"three"` (default) | Three-index LocusBlend | 3 |

Any other value raises a `ValueError`.

### Index variants

With `index_variants=None` (the default) index variants are **auto-selected by
PLINK LD clumping** at `clump_r2=0.01` within the locus window, from
`auto_index_source="dataset1"` (or `"dataset2"`). The number selected matches
the mode.

```python
result = locusblend.plot(
    dataset1, dataset2,
    reference_dir=REFERENCE_DIR,
    mode="two",
    auto_index_source="dataset2",   # run clumping on dataset 2 instead
)
```

Manual index variants are matched with the baseline's index-variant resolution
(`DISPLAY_ID` -> `rsid` -> `SNP` -> `REF_MATCH` -> `uniqueid` -> `QUERY_UID`)
and must match the mode's count exactly:

```python
result = locusblend.plot(
    dataset1, dataset2,
    reference_dir=REFERENCE_DIR,
    mode="three",
    index_variants=["rs11159021", "rs3742825", "rs74986264"],
)
```

### Output

`output=None` (default) writes nothing. `output="locusblend.png"` writes the
main combined locus figure (parent directories are created). Static rendering
uses Plotly's kaleido; if it is unavailable the existing clear
`RuntimeError` mentioning kaleido/Chrome is raised and no partial file is
written. The full 8.5 x 11 publication-page composition is not part of this
pass.

## Package layout

| Module | Responsibility |
| --- | --- |
| `api.py` | public `plot()` pipeline for the internal 1000G workflow |
| `config.py` | mode/source constants, `LocusBlendConfig`, normalization helpers |
| `models.py` | `LocusBlendResult`, `IndexVariant` |
| `io.py` | summary-statistic parsing, column aliases, chromosome helpers, locus inference helpers |
| `reference.py` | `ReferenceManager`: central external-reference path resolution |
| `variants.py` | BIM UID construction, two-pass allele matching, index resolution, clumping |
| `ld.py` | PLINK discovery/bfile handling, PLINK LD, uploaded-LD parsing/annotation |
| `genes.py` | GTF parsing, gene filtering, Plotly gene track |
| `plotting.py` | `get_plotly_locus_py`, `build_combined_locus_figure`, tooltips, theme, cloning |
| `compare.py` | compare data/panels/triptych/blended compare, autoscale |
| `export.py` | Plotly-to-PNG rendering, Pillow helpers, image composition |
| `colors.py` | `COLOR_MAPPING` (verbatim from the baseline) |

Everything under `src/locusblend/` stays Streamlit-free: no `import streamlit`,
no session state, no widgets, no Streamlit cache decorators, no server-specific
absolute paths, and no assumption that reference files live next to the
package.

## Not implemented yet

* Uploaded-LD orchestration in the public API (the underlying helpers remain in
  `locusblend.ld` for a later pass).
* The 8.5 x 11 publication-page export.
* Remote reference downloading, Zenodo logic, bundled reference data or PLINK.
* A compare-mode argument (the API uses the baseline default, three separate
  compare panels) and re-exposing Streamlit-only display controls.
* Rewriting the Streamlit application to use this package.

## Development

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q src/locusblend examples tests
```

Examples: `examples/python_api.py` (public API) and `examples/quickstart.py`
(component level).
