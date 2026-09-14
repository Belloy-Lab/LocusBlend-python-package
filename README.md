# locusblend (package refactor)

This repository is the **package refactor of LocusBlend**: the single-file
Streamlit application is being split into a reusable Python package
(`src/locusblend/`) that can power both the Streamlit web application and a
standalone Python API that generates LocusBlend figures.

## Status: early package refactor

This is an architectural pass only. Reusable, non-UI logic is being moved out
of the baseline application, but **the Streamlit app has not been converted to
use the package yet** and the **final public API is not stable**. Modules,
function signatures and the shape of `LocusBlendConfig` / `LocusBlendResult`
will change in later passes.

What is intentionally unchanged in this pass: algorithms, plotting behavior,
LD logic, PLINK handling, SNP matching, color mappings, chromosome X handling,
input parsing, and auto-index selection.

## Immutable baseline

`app.2.8.12.py` is the current stable LocusBlend Streamlit application and is
the **sole source of truth for existing behavior**. It is an immutable
baseline used for regression comparison and must not be modified, cleaned up,
reformatted, renamed or patched. The extracted modules are compared against it.

## Reference data are external

Large reference data are **never** committed to this repository and are not
bundled in the package:

* 1000 Genomes PLINK panels (`.bed` / `.bim` / `.fam`)
* GENCODE GTF annotations (`.gtf.gz`)
* recombination-rate BigWig tracks (`.bw`)
* PLINK binaries

They live outside the Git repository and are resolved centrally by
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

```python
from locusblend import ReferenceManager

ref = ReferenceManager(reference_dir="D:/locusblend_reference", ancestry="EUR")
bfile_prefix = ref.get_bfile_prefix("14")   # verifies .bed/.bim/.fam
gtf_path = ref.get_gtf_path("14")
bw_path = ref.get_recombination_bw_path()
```

`LOCUSBLEND_REFERENCE_DIR` supplies the default reference directory.
`PLINK` is not bundled either: it is taken from an explicit path, from
`LOCUSBLEND_PLINK`, or discovered on the system `PATH`.

## Package layout

| Module | Responsibility |
| --- | --- |
| `colors.py` | `COLOR_MAPPING` (three-index LD group colors, verbatim) |
| `io.py` | column/alias normalization, summary-statistic parsing, chromosome helpers, dataset summaries |
| `reference.py` | `ReferenceManager`: central path resolution for 1000G/GENCODE/recombination data |
| `variants.py` | BIM UID construction, two-pass allele matching, index-variant resolution, clumping, auto-index selection |
| `ld.py` | PLINK discovery/bfile handling, PLINK LD, uploaded LD parsing/matching, LD annotation |
| `genes.py` | GTF parsing, gene filtering, gene rows, Plotly gene track |
| `plotting.py` | core locus plotting (`get_plotly_locus_py`), tooltips, theme/clone helpers |
| `compare.py` | locus compare construction, panels, triptych, blended compare, autoscale |
| `export.py` | Plotly-to-PNG rendering, Pillow helpers, publication-page composition |
| `config.py` | configuration dataclasses and mode/source constants |
| `models.py` | result models (`LocusBlendResult`, `IndexVariant`) |
| `api.py` | future public API surface (placeholders only in this pass) |

Everything under `src/locusblend/` must stay Streamlit-free: no `import
streamlit`, no `st.session_state`, no widgets, no Streamlit cache decorators,
no server-specific absolute paths, and no assumption that reference files live
next to the package.

## Development

```bash
python -m pip install -e .
python -m pytest
python -m compileall src/locusblend
```

`examples/quickstart.py` shows the current (low-level) way to point the package
at external reference data and describe a locus configuration.

