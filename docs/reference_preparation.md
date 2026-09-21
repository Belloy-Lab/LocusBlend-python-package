# Preparing the LocusBlend reference directory

LocusBlend does **not** download, host or redistribute reference data. The
package ships no 1000 Genomes, GENCODE or recombination files, and
`locusblend.plot()` never fetches anything: you prepare the reference directory
yourself and point LocusBlend at it with `reference_dir=...` or the
`LOCUSBLEND_REFERENCE_DIR` environment variable.

This page documents the workflow that was verified against the historical
LocusBlend reference bundle, so the same layout can be reproduced from your own
copy of the upstream data.

## Upstream input

* a user-prepared **1000 Genomes GRCh38 PLINK2 PGEN** dataset
  (`.pgen`, `.pvar`, `.psam`);
* our historical source was the 1000 Genomes high-coverage Illumina integrated
  phased panel containing SNV, INDEL and SV data for 3,202 samples;
* citation: Byrska-Bishop M, et al. *High-coverage whole-genome sequencing of
  the expanded 1000 Genomes Project cohort including 602 trios.* Cell. 2022.
  PMID: 36055201.

Obtain the upstream dataset from the official 1000 Genomes Project resources and
under their terms. LocusBlend does not provide download links, mirrors,
checksums or redistribution rights.

## Expected layout

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

* `1000g/<ANCESTRY>/` holds PLINK `.bed`, `.bim` and `.fam` files for
  chromosomes **1-22 and X**;
* chromosome X must be written as **`chX`**, not the historical `ch23`;
* `gencode/` holds the chromosome-level GENCODE annotation (`.gtf.gz`);
* `recombination/` holds the recombination BigWig (`.bw`) and is **optional** -
  when it is missing, LocusBlend simply omits the recombination overlay.

### Required file names

`ReferenceManager.bfile_prefix_template` in `src/locusblend/reference.py` is the
single source of truth for the PLINK prefix:

```text
1000g_{ancestry}_hg38_high_coverage_Illumina.filtered.SNV_INDEL_SV_phased_panel_include_MHC_ch{chrom}
```

Chromosome 14 of EUR is therefore, for example:

```text
1000g/EUR/1000g_EUR_hg38_high_coverage_Illumina.filtered.SNV_INDEL_SV_phased_panel_include_MHC_ch14.bed
```

(plus the matching `.bim` and `.fam`).

### GENCODE and recombination

* `gencode/` must contain the chromosome-level annotation named by
  `ReferenceManager.gtf_chrom_template`
  (`gencode.v49.annotation.chr<chrom>.gtf.gz`); for chromosome X LocusBlend
  also accepts the whole-genome file
  (`gencode.v49.annotation.gtf.gz`, `ReferenceManager.gtf_genome_template`);
* `recombination/` must contain the BigWig named by
  `ReferenceManager.recombination_bw_template` (`recomb1000GAvg.bw`).

## Verified PLINK2 transformation

For each ancestry and each chromosome (replace the placeholders):

```bash
plink2 \
  --pfile <PGEN_PREFIX> vzs \
  --keep <ANCESTRY_KEEP_FILE> \
  --remove <RELATED_SAMPLE_REMOVE_FILE> \
  --chr <CHR> \
  --set-missing-var-ids '@:#$r:$a' \
  --new-id-max-allele-len 1000 \
  --make-bed \
  --out <OUTPUT_PREFIX>
```

Notes on the flags:

* `--pfile <PGEN_PREFIX> vzs` reads the PGEN dataset with a zstd-compressed
  `.pvar`;
* `--keep <ANCESTRY_KEEP_FILE>` uses the per-ancestry sample list
  (`AFR.txt`, `AMR.txt`, `EAS.txt`, `EUR.txt`, `SAS.txt`);
* `--remove <RELATED_SAMPLE_REMOVE_FILE>` is the related-sample removal list
  used in the historical preparation; it is optional from LocusBlend's
  perspective;
* `--set-missing-var-ids '@:#$r:$a'` together with
  `--new-id-max-allele-len 1000` assigns position/allele based variant IDs;
* **do not add `--rm-dup force-first`**: it was not part of the final BED
  workflow that LocusBlend's reference matching was verified against.

The chromosome loop must cover `1 2 ... 22 X` and write `chX` for chromosome X.

## Automated helper

`scripts/prepare_1000g_reference.sh` runs exactly the command above, loops over
the five ancestries and chromosomes 1-22 + X, derives the output file names from
`src/locusblend/reference.py` and writes directly into
`<OUTPUT_ROOT>/1000g/<ANCESTRY>/`:

```bash
bash scripts/prepare_1000g_reference.sh \
  /path/to/1000g_grch38_pgen \
  /path/to/keep_lists \
  /path/to/locusblend_reference \
  /path/to/related_samples_to_remove.txt   # optional

bash scripts/prepare_1000g_reference.sh --print-template   # show expected names
```

Requirements: `plink2` on `PATH`, and `AFR.txt`, `AMR.txt`, `EAS.txt`,
`EUR.txt`, `SAS.txt` inside `KEEP_DIR`. The script downloads nothing.

## Sample selection and QC policy

LocusBlend does not define the upstream QC or sample-selection policy. The keep
lists, the related-sample removal list and any upstream filtering decisions come
from you (or from whichever source published the dataset you use). The
historical LocusBlend reference is one instantiation of this workflow.

## Pointing LocusBlend at the result

```python
import locusblend

status = locusblend.reference_status(reference_dir="/path/to/locusblend_reference")
print(status.describe())   # available ancestries, complete/partial chromosomes

result = locusblend.plot(
    dataset1="trait1.tsv.gz",
    dataset2="trait2.tsv.gz",
    reference_dir="/path/to/locusblend_reference",
    ancestry="EUR",
    mode="three",
)
```

`LOCUSBLEND_REFERENCE_DIR` may be used instead of `reference_dir=...`. GENCODE
and recombination files are shared across ancestries.