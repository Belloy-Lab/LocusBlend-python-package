"""LocusBlend public Python API example (internal local 1000G reference).

This is the expected real-world call. It needs:

* an external reference directory (never part of this repository)::

      reference_dir/
      ├── 1000g/
      │   ├── AFR/  AMR/  EAS/  EUR/  SAS/
      ├── gencode/
      └── recombination/

* PLINK available on the system PATH, via the ``LOCUSBLEND_PLINK`` environment
  variable, or via ``plink_path=...`` (PLINK is not bundled);
* the optional export extra for ``output="..."`` (``pip install -e ".[export]"``
  - Pillow + kaleido; kaleido>=1 additionally needs Chrome/Chromium);
* GRCh38/hg38 summary statistics with at least CHR, BP, P, A1, A2
  (rsid, BETA, SE, A1FREQ, N are recommended). Files may be .csv, .tsv, .txt
  or their .gz variants. No liftover is performed.

The two dataset paths below are placeholders: point them at your own files.
Nothing is downloaded by the package.

Run with::

    python examples/python_api.py
"""

import locusblend

REFERENCE_DIR = r"D:\locusblend_reference"


def basic_example():
    """Auto-select index variants by LD clumping and write the locus PNG."""
    result = locusblend.plot(
        dataset1="trait1.tsv.gz",
        dataset2="trait2.tsv.gz",
        reference_dir=REFERENCE_DIR,
        ancestry="EUR",
        mode="three",
        output="locusblend.png",
    )

    print(result.ld_status["caption"])
    print(result.metadata["summary_text"])
    print("index variants:", result.index_variants)
    print("index reference SNPs:", result.index_reference_snps)
    print("locus figure rows: 3 (dataset 1, dataset 2, GENCODE gene track)")
    return result


def manual_and_pinned_locus_example():
    """Pin the locus and index variants explicitly (no clumping)."""
    return locusblend.plot(
        dataset1="trait1.tsv.gz",
        dataset2="trait2.tsv.gz",
        reference_dir=REFERENCE_DIR,
        ancestry="EUR",
        mode="two",                      # "standard" | "two" | "three"
        chrom="14",
        center_bp=73238768,
        window_kb=500,
        index_variants=["rs11159021", "rs3742825"],
        title1="Alzheimer's disease Female GWAS",
        title2="Monocyte PSEN1 eQTL",
        highlight_genes="PSEN1, PAPLN, HEATR4",
        output=None,                     # no file written
    )


if __name__ == "__main__":
    # Both helpers require real reference data + PLINK; the calls above are the
    # documented usage. Comment the one you do not want to run.
    basic_example()
    # manual_and_pinned_locus_example()
