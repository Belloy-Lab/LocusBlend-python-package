"""Low-level LocusBlend quickstart (component level).

This example shows what the refactored package can already do in this first
pass: point LocusBlend at external reference data, describe a locus run, and
access the component functions. For the high-level pipeline use
``locusblend.plot`` instead (see ``examples/python_api.py``).

Run it with::

    python examples/quickstart.py
"""

from locusblend import LocusBlendConfig, ReferenceManager
from locusblend.config import parse_highlighted_genes

# Reference data live outside the Git repository (never commit them).
REFERENCE_DIR = "D:/locusblend_reference"


def main():
    ref = ReferenceManager(reference_dir=REFERENCE_DIR, ancestry="EUR")
    print(ref.describe())

    for chrom in ("14", "X"):
        try:
            print(f"chr{chrom} PLINK bfile prefix: {ref.get_bfile_prefix(chrom)}")
        except FileNotFoundError as exc:
            # Expected until the external reference collection is in place.
            print(f"chr{chrom} PLINK bfile prefix unavailable:\n{exc}")

    try:
        print(f"GENCODE annotation: {ref.get_gtf_path('14')}")
    except FileNotFoundError as exc:
        print(f"GENCODE annotation unavailable: {exc}")

    print(f"Recombination track: {ref.get_recombination_bw_path()}")

    config = LocusBlendConfig(
        mode="Three-index LocusBlend",
        ancestry="EUR",
        chromosome="14",
        center_bp=73238768,
        window_kb=500,
        index_variants=("rs11159021", "rs3742825", "rs74986264"),
        highlighted_genes="PSEN1, PAPLN, HEATR4",
        title_top="Alzheimer's disease Female GWAS",
        title_bottom="Monocyte PSEN1 eQTL",
    )

    print()
    print(f"mode: {config.mode}")
    print(f"requires {config.required_n_indices} index variant(s)")
    print(f"locus: chr{config.chromosome}:{config.center_bp} +/- {config.window_kb} kb")
    print(f"window_bp: {config.window_bp}")
    print(f"highlighted genes: {sorted(parse_highlighted_genes(config.highlighted_genes))}")


if __name__ == "__main__":
    main()
